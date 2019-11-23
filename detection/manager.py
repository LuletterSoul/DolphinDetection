#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: manager.py
@time: 2019/11/16 13:22
@version 1.0
@desc:
"""

from multiprocessing import Manager, Pool, Queue, cpu_count
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, ProcessPoolExecutor
import interface as I
from pathlib import Path
from config import VideoConfig
from utils import *
from typing import List
from utils import clean_dir, logger
from detection import Detector, DetectionResult
import threading
import cv2
import imutils
import traceback
import time


# Monitor will build multiple video stream receivers according the video configuration
class DetectionMonitor(object):

    def __init__(self, video_config_path: Path, stream_path: Path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__()
        # self.cfgs = I.load_video_config(video_config_path)[-1:]
        self.cfgs = I.load_video_config(video_config_path)
        self.cfgs = [c for c in self.cfgs if c.enable]

        # Communication Pipe between detector and stream receiver
        self.pipes = [Manager().Queue() for c in self.cfgs]
        self.stream_path = stream_path
        self.region_path = region_path
        self.offline_path = offline_path
        self.process_pool = Pool(processes=cpu_count() - 1)
        self.thread_pool = ThreadPoolExecutor()

    def monitor(self):
        self.clean()
        self.call()
        self.wait()

    def call(self):
        for i, cfg in enumerate(self.cfgs):
            # clean all legacy streams and candidates files before initialization
            self.init_stream_receiver(cfg, i)
            self.init_detection(cfg, i)

    def wait(self):
        logger.info('Wait processes done.')
        self.process_pool.close()
        self.process_pool.join()
        logger.info('Closed Pool')

    def init_detection(self, cfg, i):
        self.process_pool.apply_async(I.detect,
                                      (self.stream_path / str(cfg.index), self.region_path / str(cfg.index),
                                       self.pipes[i], cfg,))

    def init_stream_receiver(self, cfg, i):
        self.process_pool.apply_async(I.read_stream, (self.stream_path / str(cfg.index), cfg, self.pipes[i],))

    def clean(self):
        clean_dir(self.stream_path)
        clean_dir(self.region_path)


# Base class embedded controllers of detector
# Each video has a detector controller
# But a controller will manager [row*col] concurrency threads or processes
# row and col are definied in video configuration
class EmbeddingControlMonitor(DetectionMonitor):
    def __init__(self, video_config_path: Path, stream_path: Path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, region_path, offline_path)
        self.caps_queue = [Manager().Queue(maxsize=500) for c in self.cfgs]
        self.caps = []
        for idx, c in enumerate(self.cfgs):
            if c.online:
                self.caps.append(
                    VideoCaptureThreading(self.stream_path / str(c.index), self.pipes[idx], self.caps_queue[idx], c,
                                          c.sample_rate))
            else:
                self.caps.append(
                    VideoOfflineCapture(self.stream_path / str(c.index), self.pipes[idx], self.caps_queue[idx], c,
                                        offline_path / str(c.index),
                                        c.sample_rate, delete_post=False))

        self.controllers = [
            DetectorController(cfg, self.stream_path / str(cfg.index), self.region_path, self.caps_queue[idx],
                               self.pipes[idx]
                               ) for
            idx, cfg in enumerate(self.cfgs)]

    def call(self):
        # Init stream receiver firstly, ensures video index that is arrived before detectors begin detection..
        for i, cfg in enumerate(self.cfgs):
            self.init_stream_receiver(cfg, i)

        # Run video capture from stream
        for i in range(len(self.cfgs)):
            self.caps[i].read()

        # Init detector controller
        self.init_controller()

    def init_controller(self):
        pass


# Concurrency based multi processes
class EmbeddingControlBasedProcessMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, region_path, offline_path)
        self.controllers = [
            ProcessBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                           self.caps_queue[idx],
                                           self.pipes[idx]
                                           ) for
            idx, cfg in enumerate(self.cfgs)]

    def init_controller(self):
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            res, detect_proc = self.controllers[i].start(self.process_pool)
            logger.info('Done init detector controller [{}]....'.format(cfg.index))


# Concurrency based multi threads
class EmbeddingControlBasedThreadMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, region_path: Path, offline_path=None) -> None:
        super().__init__(video_config_path, stream_path, region_path, offline_path)
        self.controllers = [
            ThreadBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                          self.caps_queue[idx],
                                          self.pipes[idx]
                                          ) for
            idx, cfg in enumerate(self.cfgs)]

    def init_controller(self):
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            self.thread_res = self.controllers[i].start(self.thread_pool)
            logger.info('Done init detector controller [{}]....'.format(cfg.index))

    def wait(self):
        super().wait()
        wait(self.thread_res, return_when=ALL_COMPLETED)


# Concurrency based multiple threads and multiple processes
class EmbeddingControlBasedThreadAndProcessMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, region_path, offline_path)
        self.controllers = [
            ProcessAndThreadBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                                    self.caps_queue[idx],
                                                    self.pipes[idx]
                                                    ) for
            idx, cfg in enumerate(self.cfgs)]

    def init_controller(self):
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            pool_res, thread_res = self.controllers[i].start([self.process_pool, self.thread_pool])
            self.thread_res = thread_res
            # logger.info(res.get())
            logger.info('Done init detector controller [{}]....'.format(cfg.index))

    def wait(self):
        super().wait()
        # wait all threads canceled in Thread Pool Executor
        wait(self.thread_res, return_when=ALL_COMPLETED)


class DetectorController(object):
    def __init__(self, cfg: VideoConfig, stream_path: Path, region_path: Path, frame_queue: Queue,
                 index_pool: Queue) -> None:
        super().__init__()
        self.cfg = cfg
        self.stream_path = stream_path
        self.region_path = region_path
        self.result_cnt = 0
        # self.process_pool = process_pool
        self.index_pool = index_pool
        self.col = cfg.routine['col']
        self.raw = cfg.routine['row']

        self.send_pipes = [Manager().Queue() for i in range(self.col * self.raw)]
        self.receive_pipes = [Manager().Queue() for i in range(self.col * self.raw)]
        self.frame_queue = frame_queue
        self.result_queue = Manager().Queue()

        self.quit = False

        # def __getstate__(self):

    #     self_dict = self.__dict__.copy()
    #     del self_dict['process_pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        frame = self.frame_queue.get()
        frame, original_frame = self.preprocess(frame)
        self.col_step = int(frame.shape[0] / self.col)
        self.row_step = int(frame.shape[1] / self.raw)

    def init_detectors(self):
        logger.info('Init total [{}] detectors....'.format(self.col * self.raw))
        self.detectors = []
        for i in range(self.col):
            for j in range(self.raw):
                region_detector_path = self.region_path / str(self.cfg.index) / (str(i) + '-' + str(j))
                index = self.col * i + j
                logger.info(index)
                self.detectors.append(
                    Detector(self.col_step, self.row_step, i, j, self.cfg, self.send_pipes[index],
                             self.receive_pipes[index],
                             region_detector_path))
        self.result_path = self.region_path / str(self.cfg.index) / 'frames'
        self.result_path.mkdir(exist_ok=True, parents=True)
        logger.info('Detectors init done....')

    def preprocess(self, frame):
        original_frame = frame.copy()
        if self.cfg.resize['scale'] != -1:
            frame = cv2.resize(frame, (0, 0), fx=self.cfg.resize['scale'], fy=self.cfg.resize['scale'])
        elif self.cfg.resize['width'] != -1:
            frame = imutils.resize(frame, width=self.cfg.resize['width'])
        elif self.cfg.resize['height'] != -1:
            frame = imutils.resize(frame, height=self.cfg.resize['height'])
        frame = crop_by_roi(frame, self.cfg.roi)
        # frame = imutils.resize(frame, width=1000)
        # frame = frame[340:, :, :]
        # frame = frame[170:, :, :]
        frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        return frame, original_frame

    # def start(self):
    #     self.process_pool.apply_async(self.control, (self,))

    def start(self, pool):
        self.init_control_range()
        self.init_detectors()
        return None

    def write_work(self):
        while True:
            if self.quit:
                break
            r = self.result_queue.get()
            self.result_cnt += 1
            filename = str(self.result_path / (str(self.result_cnt) + '.png'))
            cv2.imwrite(filename, r)
        return True

    def collect_and_reconstruct(self):
        logger.info('Detection controller [{}] start collect and construct'.format(self.cfg.index))
        cnt = 0
        start = time.time()
        while True:
            if self.quit:
                break
            results = self.collect()
            # logger.info('Done collected from detectors.....')
            # logger.info('Constructing sub-frames into a original frame....')
            frame, binary, thresh = self.construct(results)
            cnt += 1
            if cnt % 100 == 0:
                end = time.time() - start
                logger.info(
                    'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
                        self.cfg.index, round(end, 2), round(end / 100, 2)))
                start = time.time()
            if self.cfg.draw_boundary:
                frame = self.draw_boundary(frame)
            # logger.info('Done constructing of sub-frames into a original frame....')
            cv2.imshow('Reconstructed Frame', frame)
            cv2.imshow('Reconstructed Binary', binary)
            cv2.imshow('Reconstructed Thresh', thresh)
            cv2.waitKey(1)
        return True

    def draw_boundary(self, frame):
        shape = frame.shape
        for i in range(self.col - 1):
            start = (0, self.col_step * (i + 1))
            end = (shape[1] - 1, self.col_step * (i + 1))
            cv2.line(frame, start, end, (0, 0, 255), thickness=1)
        for j in range(self.raw - 1):
            start = (self.row_step * (j + 1), 0)
            end = (self.row_step * (j + 1), shape[0] - 1)
            cv2.line(frame, start, end, (0, 0, 255), thickness=1)
        return frame

    def dispatch(self):
        logger.info('Dispatch frame to all detectors....')
        while True:
            if self.quit:
                break
            frame = self.frame_queue.get()
            frame, original_frame = self.preprocess(frame)
            for sp in self.send_pipes:
                sp.put(frame)
        return True

    def collect(self):
        res = []
        for rp in self.receive_pipes:
            res.append(rp.get())
        return res

    def construct(self, results: List[DetectionResult]):
        sub_frames = [r.frame for r in results]
        sub_binary = [r.binary for r in results]
        sub_thresh = [r.thresh for r in results]
        constructed_frame = self.construct_rgb(sub_frames)
        constructed_binary = self.construct_gray(sub_binary)
        constructed_thresh = self.construct_gray(sub_thresh)
        for r in results:
            if len(r.regions):
                self.result_queue.put(constructed_frame)
        return constructed_frame, constructed_binary, constructed_thresh

    def construct_rgb(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.col, self.raw, self.col_step, self.row_step, 3))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3, 4))
        constructed_frame = np.reshape(sub_frames, (self.col * self.col_step, self.raw * self.row_step, 3))
        return constructed_frame

    def construct_gray(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.col, self.raw, self.col_step, self.row_step))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3))
        constructed_frame = np.reshape(sub_frames, (self.col * self.col_step, self.raw * self.row_step))
        return constructed_frame


class ProcessBasedDetectorController(DetectorController):

    def start(self, pool: Pool):
        super().start(pool)
        res = pool.apply_async(self.collect_and_reconstruct, ())
        pool.apply_async(self.dispatch, ())
        logger.info('Running detectors.......')
        detect_proc_res = []
        for idx, d in enumerate(self.detectors):
            logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.raw_index))
            detect_proc_res.append(pool.apply_async(d.detect, ()))
            # detect_proc_res.append(pool.submit(d.detect, ()))
            logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.raw_index))
        return res, detect_proc_res
        # self.monitor.wait_pool()
        # self.loop_work()


class ThreadBasedDetectorController(DetectorController):

    def start(self, pool: ThreadPoolExecutor):
        super().start(pool)
        thread_res = []
        try:
            thread_res.append(pool.submit(self.collect_and_reconstruct))
            thread_res.append(pool.submit(self.dispatch))
            logger.info('Running detectors.......')
            for idx, d in enumerate(self.detectors):
                logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.raw_index))
                thread_res.append(pool.submit(d.detect))
                # detect_proc_res.append(pool.submit(d.detect, ()))
                logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.raw_index))
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        return thread_res
        # self.monitor.wait_pool()
        # self.loop_work()


class ProcessAndThreadBasedDetectorController(DetectorController):

    def start(self, pool):
        process_pool = pool[0]
        thread_pool = pool[1]
        pool_res = []
        thread_res = []
        super().start(process_pool)
        # collect child frames and reconstruct frames from detectors asynchronously
        pr1 = process_pool.apply_async(self.collect_and_reconstruct, ())
        pool_res.append(pr1)
        # dispatch child frames to detector asynchronously
        pr2 = process_pool.apply_async(self.dispatch, ())
        # write detection result asynchronously
        thread_res.append(thread_pool.submit(self.write_work))
        pool_res.append(pr2)
        logger.info('Running detectors.......')
        for idx, d in enumerate(self.detectors):
            logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.raw_index))
            thread_res.append(thread_pool.submit(d.detect))
            # detect_proc_res.append(pool.submit(d.detect, ()))
            logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.raw_index))
        return pool_res, thread_res
        # self.monitor.wait_pool()
        # self.loop_work()


class VideoCaptureThreading:
    def __init__(self, video_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig, sample_rate=5,
                 width=640,
                 height=480,
                 delete_post=True):
        self.cfg = cfg
        self.video_path = video_path
        self.index_pool = index_pool
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.sample_rate = sample_rate
        self.frame_queue = frame_queue
        self.delete_post = delete_post

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def __start__(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        src = self.load_next_src()
        logger.info('Loading next video stream from [{}]....'.format(src))
        self.cap = cv2.VideoCapture(src)
        logger.info('Loading done from: [{}]'.format(src))
        self.started = True
        self.thread.start()
        return self

    def load_next_src(self):
        logger.debug('Loading video stream from video index pool....')
        self.posix = self.video_path / self.index_pool.get()
        self.src = str(self.posix)
        if not os.path.exists(self.src):
            logger.debug('Video path not exist: [{}]'.format(self.src))
            return -1
        return self.src

    def update(self):
        cnt = 0
        while self.started:
            # with self.read_lock:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.update_capture(cnt)
                cnt = 0
                continue
            if cnt % self.sample_rate == 0:
                self.frame_queue.put(frame)
            cnt += 1
        logger.info('Video Capture [{}]: cancel..'.format(self.cfg.index))

    def update_capture(self, cnt):
        logger.debug('Read frame done from [{}].Has loaded [{}] frames'.format(self.src, cnt))
        logger.debug('Read next frame from video ....')
        self.cap.release()
        if self.posix.exists() and self.delete_post:
            self.posix.unlink()
        src = self.load_next_src()
        if src == -1:
            self.started = False
        self.cap = cv2.VideoCapture(src)

    def post_update(self):
        pass

    def read(self):
        if not self.started:
            self.__start__()
        return self.frame_queue.get()
        # with self.read_lock:
        # frame = self.frame.copy()
        # grabbed = self.grabbed
        # return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.cap.release()
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


class VideoOfflineCapture(VideoCaptureThreading):
    def __init__(self, video_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig, offline_path: Path,
                 sample_rate=5,
                 width=640, height=480, delete_post=False):
        super().__init__(video_path, index_pool, frame_queue, cfg, sample_rate, width, height, delete_post)
        self.offline_path = offline_path
        self.streams_list = list(self.offline_path.glob('*'))
        self.pos = 0

    def load_next_src(self):
        logger.debug('Loading next video stream ....')
        if self.pos >= len(self.streams_list):
            logger.info('Load completely for [{}]'.format(str(self.offline_path)))
            return -1
        self.posix = self.streams_list[self.pos]
        self.src = str(self.posix)
        self.pos += 1
        if not os.path.exists(self.src):
            logger.debug('Video path not exist: [{}]'.format(self.src))
            return -1
        return self.src

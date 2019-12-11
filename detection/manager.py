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

from multiprocessing import Manager, Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import stream
import interface as I
from .capture import *
from .detector import *
from utils import *
from typing import List
from utils import clean_dir, logger
from config import enable_options
from collections import deque

# from capture import *
import cv2
import imutils
import traceback
import time
import threading
import traceback


# Monitor will build multiple video stream receivers according the video configuration
class DetectionMonitor(object):

    def __init__(self, video_config_path: Path, stream_path: Path, sample_path: Path, frame_path: Path,
                 region_path: Path,
                 offline_path: Path = None, build_pool=True) -> None:
        super().__init__()
        # self.cfgs = I.load_video_config(video_config_path)[-1:]
        self.cfgs = I.load_video_config(video_config_path)
        # self.cfgs = [c for c in self.cfgs if c.enable]
        self.cfgs = [c for c in self.cfgs if enable_options[c.index]]
        self.quit = False
        # Communication Pipe between detector and stream receiver
        self.pipes = [Manager().Queue(c.max_streams_cache) for c in self.cfgs]
        self.stream_path = stream_path
        self.sample_path = sample_path
        self.frame_path = frame_path
        self.region_path = region_path
        self.offline_path = offline_path
        self.process_pool = None
        self.thread_pool = None
        if build_pool:
            self.process_pool = Pool(processes=cpu_count() - 1)
            self.thread_pool = ThreadPoolExecutor()
        self.clean()
        self.stream_receivers = [
            stream.StreamReceiver(self.stream_path / str(c.index), offline_path, c, self.pipes[idx]) for idx, c in
            enumerate(self.cfgs)]

    def monitor(self):
        self.call()
        self.wait()

    def call(self):
        for i, cfg in enumerate(self.cfgs):
            # clean all legacy streams and candidates files before initialization
            self.init_stream_receiver(i)
            self.init_detection(cfg, i)

    def wait(self):
        logger.info('Wait processes done.')
        if self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()

        logger.info('Closed Pool')

    def init_detection(self, cfg, i):
        self.process_pool.apply_async(detect,
                                      (self.stream_path / str(cfg.index), self.region_path / str(cfg.index),
                                       self.pipes[i], cfg,))

    # def init_stream_receiver(self, cfg, i):
    #     self.process_pool.apply_async(I.read_stream, (self.stream_path / str(cfg.index), cfg, self.pipes[i],))
    def init_stream_receiver(self, i):
        return self.process_pool.apply_async(self.stream_receivers[i].receive_online)

    def clean(self):
        clean_dir(self.sample_path)
        clean_dir(self.stream_path)
        clean_dir(self.region_path)


# Base class embedded controllers of detector
# Each video has a detector controller
# But a controller will manager [row*col] concurrency threads or processes
# row and col are definied in video configuration
class EmbeddingControlMonitor(DetectionMonitor):
    def __init__(self, video_config_path: Path, stream_path: Path, sample_path: Path, frame_path: Path, region_path,
                 offline_path: Path = None, build_pool=True) -> None:
        super().__init__(video_config_path, stream_path, sample_path, frame_path, region_path, offline_path, build_pool)
        self.caps_queue = [Manager().Queue(maxsize=500) for c in self.cfgs]
        self.caps = []
        self.controllers = []

    def init_caps(self):
        for idx, c in enumerate(self.cfgs):
            if c.online == "http":
                self.caps.append(
                    VideoOnlineSampleCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                             self.pipes[idx],
                                             self.caps_queue[idx],
                                             c, c.sample_rate))

            elif c.online == "rtsp":
                self.caps.append(
                    VideoRtspCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                     self.pipes[idx], self.caps_queue[idx], c, c.sample_rate)
                )
            else:
                self.caps.append(
                    VideoOfflineCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                        self.offline_path / str(c.index),
                                        self.pipes[idx],
                                        self.caps_queue[idx], c, c.sample_rate,
                                        delete_post=False))

    def call(self):

        self.init_caps()
        # Init stream receiver firstly, ensures video index that is arrived before detectors begin detection..
        for i, cfg in enumerate(self.cfgs):
            res = self.init_stream_receiver(i)
            # logger.debug(res.get())

        # Run video capture from stream
        for i in range(len(self.cfgs)):
            self.caps[i].read()

        # Init detector controller
        self.init_controllers()

    def init_controllers(self):
        pass

        # Concurrency based multi processes


class EmbeddingControlBasedProcessMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, sample_path, frame_path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, sample_path, frame_path, region_path, offline_path)

    def init_controllers(self):
        self.controllers = [
            ProcessBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                           self.frame_path / str(cfg.index),
                                           self.caps_queue[idx],
                                           self.pipes[idx]
                                           ) for
            idx, cfg in enumerate(self.cfgs)]
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            res, detect_proc = self.controllers[i].start(self.process_pool)
            logger.info('Done init detector controller [{}]....'.format(cfg.index))

    # Concurrency based multi threads


class EmbeddingControlBasedThreadMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, sample_path, frame_path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, sample_path, frame_path, region_path, offline_path)

    def init_controllers(self):
        self.controllers = [
            ProcessBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                           self.frame_path / str(cfg.index),
                                           self.caps_queue[idx],
                                           self.pipes[idx]
                                           ) for
            idx, cfg in enumerate(self.cfgs)]
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            self.thread_res = self.controllers[i].start(self.thread_pool)
            logger.info('Done init detector controller [{}]....'.format(cfg.index))

    def wait(self):
        super().wait()
        wait(self.thread_res, return_when=ALL_COMPLETED)

    # Concurrency based multiple threads and multiple processes


class EmbeddingControlBasedThreadAndProcessMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, sample_path: Path, frame_path: Path,
                 region_path: Path,
                 offline_path=None) -> None:
        super().__init__(video_config_path, stream_path, sample_path, frame_path, region_path, offline_path)

    def init_controllers(self):
        self.controllers = [
            ProcessAndThreadBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                                    self.frame_path / str(cfg.index),
                                                    self.caps_queue[idx],
                                                    self.pipes[idx]
                                                    ) for
            idx, cfg in enumerate(self.cfgs)]
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


class DispatchBlock(object):

    def __init__(self, frame, index, original_shape) -> None:
        super().__init__()
        self.frame = frame
        self.index = index
        self.shape = original_shape


class DetectorController(object):
    def __init__(self, cfg: VideoConfig, stream_path: Path, region_path: Path, frame_path: Path,
                 frame_queue: Queue,
                 index_pool: Queue) -> None:
        super().__init__()
        self.cfg = cfg
        self.stream_path = stream_path
        self.frame_path = frame_path
        self.region_path = region_path
        self.result_cnt = 0
        self.stream_cnt = 0
        # self.process_pool = process_pool
        self.col = cfg.routine['col']
        self.row = cfg.routine['row']

        self.send_pipes = [Manager().Queue() for i in range(self.col * self.row)]
        self.receive_pipes = [Manager().Queue() for i in range(self.col * self.row)]
        self.index_pool = index_pool
        self.frame_queue = frame_queue
        self.result_queue = Manager().Queue(self.cfg.max_streams_cache)
        self.quit = False
        self.frame_cnt = Manager().Value('i', 0)
        # self.frame_cnt =
        self.next_prepare_event = Manager().Event()
        self.next_prepare_event.clear()
        self.pre_detect_index = -self.cfg.future_frames
        self.history_write = False
        self.original_frame_cache = Manager().dict()
        self.render_frame_cache = Manager().dict()
        self.history_frame_deque = Manager().list()
        self.record_cnt = 48
        self.render_task_cnt = 0
        self.construct_cnt = 0

        # time scheduler to clear cache
        self.render_events = Manager().dict()
        self.last_detection = time.time()
        self.runtime = time.time()
        # self.clear_point = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')

        # def __getstate__(self):

    #     self_dict = self.__dict__.copy()
    #     del self_dict['process_pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    def clear_original_cache(self):
        len_cache = len(self.original_frame_cache)
        if len_cache > 1000:
            # original_head = self.original_frame_cache.keys()[0]
            self.clear_cache(self.original_frame_cache)
            logger.info(
                'Clear half original frame caches.')

    def clear_cache(self, cache, num=2):
        try:
            half_len = len(cache) // num
            cnt = 0
            keys = cache.keys()
            for k in keys:
                if k in self.original_frame_cache:
                    self.original_frame_cache.pop(k)
                    cnt += 1
                    if cnt == half_len:
                        break
            # logger.info(self.original_frame_cache.keys())
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        # self.render_frame_cache.clear()
        # self.original_frame_cache.clear()

    def clear_render_cache(self):
        last_detect_internal = time.time() - self.last_detection
        time_thresh = self.cfg.future_frames * 1.5 * 3
        if last_detect_internal > time_thresh:
            self.clear_cache(self.render_frame_cache)
            logger.info('Clear half render frame caches')

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        frame = self.frame_queue.get()
        frame, original_frame = self.preprocess(frame)
        self.col_step = int(frame.shape[0] / self.col)
        self.row_step = int(frame.shape[1] / self.row)

    def init_detectors(self):
        logger.info('Init total [{}] detectors....'.format(self.col * self.row))
        self.detectors = []
        for i in range(self.col):
            for j in range(self.row):
                region_detector_path = self.region_path / str(self.cfg.index) / (str(i) + '-' + str(j))
                index = self.col * i + j
                self.detectors.append(
                    Detector(self.col_step, self.row_step, i, j, self.cfg, self.send_pipes[index],
                             self.receive_pipes[index],
                             region_detector_path))
        self.result_path = self.region_path / str(self.cfg.index) / 'frames'
        self.detect_stream_path = self.region_path / str(self.cfg.index) / 'streams'
        self.test_path = self.region_path / str(self.cfg.index) / 'tests'
        self.detect_stream_path.mkdir(exist_ok=True, parents=True)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.test_path.mkdir(exist_ok=True, parents=True)
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
        # frame = cv2.Filter(frame, ksize=(3, 3), sigmaX=0)
        return frame, original_frame

    # def start(self):
    #     self.process_pool.apply_async(self.control, (self,))

    def start(self, pool):
        self.init_control_range()
        self.init_detectors()
        return None

    def write_frame_work(self):
        logger.info('Writing process.......')
        current_time = time.strftime('%m-%d-%H:%M-', time.localtime(time.time()))
        target = self.result_path / (current_time + str(self.result_cnt) + '.png')
        logger.info('Writing stream frame into: [{}]'.format(str(target)))
        while True:
            if self.quit:
                break
            try:
                r = self.get_result_from_queue()
                self.result_cnt += 1
                current_time = time.strftime('%m-%d-%H:%M-', time.localtime(time.time()))
                target = self.result_path / (current_time + str(self.result_cnt) + '.png')
                cv2.imwrite(str(target), r)
            except Exception as e:
                logger.error(e)
        return True

    def render_detection_frame(self, frame_idx, render_cache, frame_cache, task_id):
        self.stream_cnt += 1
        current_time = time.strftime('%m-%d-%H:%M:%S-', time.localtime(time.time()))
        target = self.detect_stream_path / (current_time + str(self.stream_cnt) + '.mp4')
        logger.info('Render task [{}]: Writing detection stream frame into: [{}]'.format(task_id, str(target)))
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_write = cv2.VideoWriter(str(target), self.fourcc, 24.0, (1920, 1080), True)
        next_cnt = 0
        # logger.info('Idx:[{}]'.format(frame_idx))
        # logger.info('Pre :[{}]'.format(render_cache.keys()))
        try:
            for idx in frame_idx:
                if idx in render_cache.keys():
                    video_write.write(render_cache[idx])
                    render_cache.pop(idx)
                    next_cnt = idx + 1
                elif idx in frame_cache.keys():
                    video_write.write(frame_cache[idx])
                    # frame_cache.pop(idx)
                    next_cnt = idx + 1
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        # the future frames count
        # next_frame_cnt = 48
        success = 0
        # wait the futures frames is accessable
        if not self.next_prepare_event.is_set():
            logger.info('Wait frames accessible....')
            # wait the future frames prepared,if ocurring time out, give up waits
            self.next_prepare_event.wait(30)
            logger.info('Frames accessible....')
        # logger.info('Render task Begin with frame [{}]'.format(next_cnt))
        start = time.time()
        # logger.info('After :[{}]'.format(render_cache.keys()))
        while True:
            # failed to wait
            end = time.time() - start
            try:
                if next_cnt in render_cache.keys():
                    frame = render_cache[next_cnt]
                    video_write.write(frame)
                    # logger.info('Writing frame [{}] into video stream from render_cache'.format(next_cnt))
                    # test_target = str(self.test_path / (str(next_cnt) + '.png'))
                    # cv2.imwrite(test_target, frame)
                    # render_cache.pop(next_cnt)
                    next_cnt += 1
                    success += 1
                # if render frame not found ,using the original frame as substitude
                elif next_cnt in frame_cache.keys():
                    frame = frame_cache[next_cnt]
                    video_write.write(frame)
                    # frame_cache.pop(next_cnt)
                    next_cnt += 1
                    success += 1
                else:
                    logger.info('Lost frame index: [{}]'.format(next_cnt))
                    logger.info('Original frame index: {}'.format(frame_cache.keys()))
                    logger.info('Renderframe index: {}'.format(render_cache.keys()))
                if end >= 30:
                    logger.info('Task time overflow, complete previous render task.')
                    break
                if success == self.cfg.future_frames:
                    logger.info('Render task [{}]: Full write attache [{}] frames'.format(task_id, success))
                    break
            except Exception as e:
                logger.error(e)
        video_write.release()
        logger.info('Render task [{}]: Done write detection stream frame into: [{}]'.format(task_id, str(target)))
        return True

    def get_result_from_queue(self):
        return self.result_queue.get()

    def collect_and_reconstruct(self):
        logger.info('Detection controller [{}] start collect and construct'.format(self.cfg.index))
        cnt = 0
        start = time.time()
        while True:
            if self.quit:
                break
            logger.debug('Collecting sub-frames into a original frame....')
            results = self.collect()
            frame, binary, thresh = self.construct(results)
            logger.debug('Done Construct sub-frames into a original frame....')
            cnt += 1
            if cnt % 100 == 0:
                end = time.time() - start
                logger.info(
                    'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
                        self.cfg.index, round(end, 2), round(end / 100, 2)))
                start = time.time()
                cnt = 0
            if self.cfg.draw_boundary:
                frame = self.draw_boundary(frame)
            # logger.info('Done constructing of sub-frames into a original frame....')
            if self.cfg.show_window:
                cv2.imshow('Reconstructed Frame', frame)
                cv2.waitKey(1)
        return True

    def draw_boundary(self, frame):
        shape = frame.shape
        for i in range(self.col - 1):
            start = (0, self.col_step * (i + 1))
            end = (shape[1] - 1, self.col_step * (i + 1))
            cv2.line(frame, start, end, (0, 0, 255), thickness=1)
        for j in range(self.row - 1):
            start = (self.row_step * (j + 1), 0)
            end = (self.row_step * (j + 1), shape[0] - 1)
            cv2.line(frame, start, end, (0, 0, 255), thickness=1)
        return frame

    def dispatch(self):
        # start = time.time()
        while True:
            if self.quit:
                break
            frame = self.frame_queue.get()
            self.frame_cnt.set(self.frame_cnt.get() + 1)
            self.original_frame_cache[self.frame_cnt.get()] = frame
            # self.render_frame_cache[self.frame_cnt.get()] = frame
            # logger.info(self.original_frame_cache.keys())

            frame, original_frame = self.preprocess(frame)
            if self.frame_cnt.get() % self.cfg.sample_rate == 0:
                for idx, sp in enumerate(self.send_pipes):
                    sp.put(DispatchBlock(crop_by_se(frame, self.detectors[idx].start, self.detectors[idx].end),
                                         self.frame_cnt.get(), original_frame.shape))

            self.clear_original_cache()
            logger.info('Dispatch frame to all detectors....')
            # internal = (time.time() - start) / 60
            # if int(internal) == self.cfg.sample_internal:
            #     cv2.imwrite(str(self.frame_path / ))
        return True

    def collect(self):
        res = []
        for rp in self.receive_pipes:
            res.append(rp.get())
        logger.info('Collect sub-frames from all detectors....')
        return res

    def construct(self, results: List[DetectionResult]):
        # sub_frames = [r.frame for r in results]
        # sub_binary = [r.binary for r in results]
        # sub_thresh = [r.thresh for r in results]
        # constructed_frame = self.construct_rgb(sub_frames)
        # constructed_binary = self.construct_gray(sub_binary)
        # constructed_thresh = self.construct_gray(sub_thresh)
        try:
            self.construct_cnt += 1
            current_index = results[0].frame_index
            if current_index not in self.original_frame_cache:
                logger.info('Current index: [{}] not in original frame cache.May cache was cleared by timer')
                return
            original_frame = self.original_frame_cache[current_index]
            detect_flag = False
            if len(self.history_frame_deque) >= self.record_cnt:
                self.history_frame_deque.pop(0)
            self.history_frame_deque.append(current_index)
            for r in results:
                if len(r.rects):
                    self.result_queue.put(original_frame)
                    detect_flag = True
                    self.last_detection = time.time()
                    if r.frame_index not in self.original_frame_cache:
                        logger.info('Unknown frame index: [{}] to fetch frame in cache.'.format(r.frame_index))
                        continue
                    for rect in r.rects:
                        color = np.random.randint(0, 255, size=(3,))
                        color = [int(c) for c in color]
                        p1 = (rect[0] - 80, rect[1] - 80)
                        p2 = (rect[0] + 100, rect[1] + 100)
                        # cv2.rectangle(original_frame, (rect[0] - 20, rect[1] - 20),
                        #               (rect[0] + rect[2] + 20, rect[1] + rect[3] + 20),
                        #               color, 2)
                        cv2.rectangle(original_frame, p1, p2, color, 2)
                    self.render_frame_cache[current_index] = original_frame
            next_detect_stream_occurred = current_index - self.pre_detect_index >= self.cfg.future_frames
            # logger.info('current index:[{}]'.format(current_index))
            if detect_flag and next_detect_stream_occurred:
                # logger.info('enter pre detect index:[{}]'.format(self.pre_detect_index))
                if self.next_prepare_event.is_set():
                    self.next_prepare_event.clear()
                # begin task asynchronously  in case blocking collector
                thread = threading.Thread(
                    target=self.render_detection_frame,
                    # pass a copy history frame index
                    args=(list(self.history_frame_deque), self.render_frame_cache, self.original_frame_cache,
                          self.render_task_cnt,))
                self.render_task_cnt += 1
                thread.start()
                # self.construct_cnt = current_index
                self.pre_detect_index = current_index
                self.history_write = True
            # self.result_queue.put(constructed_frame)
            # self.render_frame_cache[current_index] = self.original_frame_cache[current_index]
            if current_index - self.pre_detect_index >= self.cfg.future_frames and self.history_write:
                # notify render task that the future frames(2s default required) are done
                if not self.next_prepare_event.is_set():
                    self.next_prepare_event.set()
                    logger.info(
                        'Notify detection stream writer.Current frame index [{}],Previous detected frame index [{}]...'.format(
                            current_index, self.pre_detect_index))
                self.history_write = False
            self.clear_render_cache()
            # return constructed_frame, constructed_binary, constructed_thresh
            return original_frame, None, None
        except Exception as e:
            logger.info(e)

    def construct_rgb(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.col, self.row, self.col_step, self.row_step, 3))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3, 4))
        constructed_frame = np.reshape(sub_frames, (self.col * self.col_step, self.row * self.row_step, 3))
        return constructed_frame

    def construct_gray(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.col, self.row, self.col_step, self.row_step))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3))
        constructed_frame = np.reshape(sub_frames, (self.col * self.col_step, self.row * self.row_step))
        return constructed_frame


class ReconstructResult(object):

    def __init__(self, rcf, rcb, rct) -> None:
        super().__init__()
        self.reconstruct_frame = rcf
        self.reconstruct_binary = rcb
        self.reconstruct_rct = rct


class ProcessBasedDetectorController(DetectorController):

    def start(self, pool: Pool):
        super().start(pool)
        res = pool.apply_async(self.collect_and_reconstruct, ())
        pool.apply_async(self.dispatch, ())
        pool.apply_async(self.write_frame_work)
        logger.info('Running detectors.......')
        detect_proc_res = []
        for idx, d in enumerate(self.detectors):
            logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.row_index))
            detect_proc_res.append(pool.apply_async(d.detect, ()))
            # detect_proc_res.append(pool.submit(d.detect, ()))
            logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.row_index))
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
                logger.info(
                    'Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.row_index))
                thread_res.append(pool.submit(d.detect))
                # detect_proc_res.append(pool.submit(d.detect, ()))
                logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.row_index))
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
        thread_res.append(thread_pool.submit(self.write_frame_work))
        pool_res.append(pr2)
        logger.info('Running detectors.......')
        for idx, d in enumerate(self.detectors):
            logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.row_index))
            thread_res.append(thread_pool.submit(d.detect))
            # detect_proc_res.append(pool.submit(d.detect, ()))
            logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.row_index))
        return pool_res, thread_res
        # self.monitor.wait_pool()
        # self.loop_work()

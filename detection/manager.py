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
from detection.params import DispatchBlock, ConstructResult, BlockInfo, ConstructParams
from .capture import *
from .detector import *
from .detect_funcs import detect_based_task, collect_and_reconstruct
from detection import DetectorParams
from utils import *
from typing import List
from utils import clean_dir, logger
from config import enable_options

# from capture import *
import cv2
import imutils
import time
import threading
import traceback
import ray


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
                self.init_http_caps(c, idx)
            elif c.online == "rtsp":
                self.init_rtsp_caps(c, idx)
            else:
                self.init_offline_caps(c, idx)

    def init_offline_caps(self, c, idx):
        self.caps.append(
            VideoOfflineCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                self.offline_path / str(c.index),
                                self.pipes[idx],
                                self.caps_queue[idx], c, idx, c.sample_rate,
                                delete_post=False))

    def init_http_caps(self, c, idx):
        self.caps.append(
            VideoOnlineSampleCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                     self.pipes[idx],
                                     self.caps_queue[idx],
                                     c, idx, c.sample_rate))

    def init_rtsp_caps(self, c, idx):
        self.caps.append(
            VideoRtspCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                             self.pipes[idx], self.caps_queue[idx], c, idx, c.sample_rate)
        )

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


class EmbeddingControlBasedTaskMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, sample_path, frame_path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, sample_path, frame_path, region_path, offline_path)
        self.task_futures = []

    def init_controllers(self):
        self.controllers = [
            TaskBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path,
                                        self.frame_path / str(cfg.index),
                                        self.caps_queue[idx],
                                        self.pipes[idx]
                                        ) for
            idx, cfg in enumerate(self.cfgs)]
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            self.task_futures.append(self.controllers[i].start(self.process_pool))
            logger.info('Done init detector controller [{}]....'.format(cfg.index))

    def init_rtsp_caps(self, c, idx):
        self.caps.append(
            VideoRtspCallbackCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                     self.pipes[idx], self.caps_queue[idx], c, idx, self, c.sample_rate)
        )

    def call(self):

        self.init_caps()
        # Init stream receiver firstly, ensures video index that is arrived before detectors begin detection..
        for i, cfg in enumerate(self.cfgs):
            res = self.init_stream_receiver(i)
            # logger.debug(res.get())

        # Init detector controller
        self.init_controllers()

        # Run video capture from stream
        for i in range(len(self.cfgs)):
            self.caps[i].read()
            # r =self.process_pool.apply_async(self.caps[i].read, ())
            # r.get()

    def callback(self, index, frame):
        self.controllers[index].call_task(frame, self.process_pool)

    def wait(self):
        if self.process_pool is not None:
            logger.info('Waiting processes canceled.')
            results = [r.get() for r in self.task_futures]
            self.process_pool.close()
            self.process_pool.join()


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
        self.col_step = 0
        self.row_step = 0
        self.block_info = BlockInfo(self.row, self.col, self.row_step, self.col_step)

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
        self.render_rect_cache = Manager().dict()
        self.detect_index = Manager().dict()
        self.record_cnt = 48
        self.render_task_cnt = 0
        self.construct_cnt = 0
        self.dispatch_cnt = 0

        # time scheduler to clear cache
        self.render_events = Manager().dict()
        self.last_detection = time.time()
        self.runtime = time.time()
        # self.clear_point = 0
        # self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.stream_render = DetectionStreamRender(0, self.cfg.future_frames, self)

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
            thread = threading.Thread(
                target=clear_cache,
                args=(self.original_frame_cache,))
            # self.clear_cache(self.original_frame_cache)
            thread.start()
            logger.info(
                'Clear half original frame caches.')

    def clear_render_cache(self):
        last_detect_internal = time.time() - self.last_detection
        time_thresh = self.cfg.future_frames * 1.5 * 3
        if last_detect_internal > time_thresh and len(self.render_frame_cache) > 500:
            thread = threading.Thread(
                target=clear_cache,
                args=(self.render_frame_cache,))
            thread.start()
            logger.info('Clear half render frame caches')
        if len(self.render_rect_cache) > 500:
            thread = threading.Thread(
                target=clear_cache,
                args=(self.render_rect_cache,))
            thread.start()

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        frame = self.frame_queue.get()
        frame, original_frame = self.preprocess(frame)
        self.col_step = int(frame.shape[0] / self.col)
        self.row_step = int(frame.shape[1] / self.row)
        self.block_info = BlockInfo(self.row, self.col, self.row_step, self.col_step)

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
        current_time = time.strftime('%m-%d-%H-%M-', time.localtime(time.time()))
        target = self.result_path / (current_time + str(self.result_cnt) + '.png')
        logger.info('Writing stream frame into: [{}]'.format(str(target)))
        while True:
            if self.quit:
                break
            try:
                r = self.get_result_from_queue()
                self.result_cnt += 1
                current_time = time.strftime('%m-%d-%H-%M-', time.localtime(time.time()))
                target = self.result_path / (current_time + str(self.result_cnt) + '.png')
                cv2.imwrite(str(target), r)
            except Exception as e:
                logger.error(e)
        return True

    def get_result_from_queue(self):
        return self.result_queue.get()

    def collect_and_reconstruct(self, args, pool):
        logger.info('Detection controller [{}] start collect and construct'.format(self.cfg.index))
        cnt = 0
        start = time.time()
        while True:
            if self.quit:
                break
            # logger.debug('Collecting sub-frames into a original frame....')
            # start = time.time()
            results = self.collect(args)
            # logger.info('Collect consume [{}]'.format(time.time() - start ))
            construct_result: ConstructResult = self.construct(results)
            # logger.debug('Done Construct sub-frames into a original frame....')
            cnt += 1
            if cnt % 100 == 0:
                end = time.time() - start
                logger.info(
                    'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
                        self.cfg.index, round(end, 2), round(end / 100, 2)))
                start = time.time()
                cnt = 0
            if self.cfg.draw_boundary:
                frame = draw_boundary(construct_result.frame, self.block_info)
            # logger.info('Done constructing of sub-frames into a original frame....')
            if self.cfg.show_window:
                cv2.imshow('Reconstructed Frame', construct_result.frame)
                cv2.waitKey(1)
        return True

    # def draw_boundary(self, frame):
    #     shape = frame.shape
    #     for i in range(self.col - 1):
    #         start = (0, self.col_step * (i + 1))
    #         end = (shape[1] - 1, self.col_step * (i + 1))
    #         cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    #     for j in range(self.row - 1):
    #         start = (self.row_step * (j + 1), 0)
    #         end = (self.row_step * (j + 1), shape[0] - 1)
    #         cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    #     return frame

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
                logger.info('Dispatch frame to all detectors....')
                for idx, sp in enumerate(self.send_pipes):
                    sp.put(DispatchBlock(crop_by_se(frame, self.detectors[idx].start, self.detectors[idx].end),
                                         self.frame_cnt.get(), original_frame.shape))

            self.clear_original_cache()
            # internal = (time.time() - start) / 60
            # if int(internal) == self.cfg.sample_internal:
            #     cv2.imwrite(str(self.frame_path / ))
        return True

    def collect(self, args):
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
        logger.info('Controller [{}]: Construct frames into a original frame....'.format(self.cfg.index))
        try:
            self.construct_cnt += 1
            current_index = results[0].frame_index
            while current_index not in self.original_frame_cache:
                logger.info('Current index: [{}] not in original frame cache.May cache was cleared by timer'.format(
                    current_index))
                time.sleep(0.5)
                # logger.info(self.original_frame_cache.keys())
            original_frame = self.original_frame_cache[current_index]
            for r in results:
                if len(r.rects):
                    self.result_queue.put(original_frame)
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
                    self.render_rect_cache[current_index] = r.rects
                    self.stream_render.reset(current_index)
            self.stream_render.notify(current_index)
            self.clear_render_cache()
            # return constructed_frame, constructed_binary, constructed_thresh
            return ConstructResult(original_frame, None, None)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

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


class DetectionStreamRender(object):

    def __init__(self, detect_index, future_frames, controller: DetectorController) -> None:
        super().__init__()
        self.detect_index = detect_index
        self.detect_stream_path = controller.region_path / str(controller.cfg.index) / 'streams'
        self.detect_stream_path = controller.region_path / str(controller.cfg.index) / 'streams'
        self.stream_cnt = 0
        self.is_trigger_write = False
        self.write_done = False
        self.controller = controller
        self.future_frames = future_frames
        self.sample_rate = controller.cfg.sample_rate
        self.render_frame_cache = controller.render_frame_cache
        self.render_rect_cache = controller.render_rect_cache
        self.render_task_cnt = 0
        self.original_frame_cache = controller.original_frame_cache
        self.next_prepare_event = Manager().Event()
        self.next_prepare_event.set()
        # self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    def reset(self, detect_index):
        if detect_index - self.detect_index > self.future_frames:
            self.detect_index = detect_index
            self.is_trigger_write = False
            self.write_done = False
            self.next_prepare_event.set()
            logger.info('Reset stream render')

    def notify(self, current_index):
        # next_detect_stream_occurred = current_index - self.detect_index >= self.future_frames \
        #                               and not self.is_trigger_write
        if not self.is_trigger_write:
            if self.next_prepare_event.is_set():
                self.next_prepare_event.clear()
                # begin task asynchronously  in case blocking collector
                thread = threading.Thread(
                    target=self.render_task,
                    # pass a copy history frame index
                    args=(current_index, self.render_frame_cache, self.render_rect_cache,
                          self.original_frame_cache, self.render_task_cnt,))
                self.render_task_cnt += 1
                thread.start()
                self.is_trigger_write = True
        if current_index - self.detect_index >= self.future_frames and self.write_done:
            # notify render task that the future frames(2s default required) are done
            if not self.next_prepare_event.is_set():
                self.next_prepare_event.set()
                logger.info(
                    'Notify detection stream writer.Current frame index [{}],Previous detected frame index [{}]...'.format(
                        current_index, self.detect_index))

    def write_video_work(self, video_write, next_cnt, end_cnt, render_cache, rect_cache, frame_cache, task_id):
        if next_cnt < 1:
            next_cnt = 1
        start = time.time()
        while next_cnt < end_cnt:
            try:
                if next_cnt in render_cache:
                    forward_cnt = next_cnt + self.sample_rate
                    if forward_cnt > end_cnt:
                        forward_cnt = end_cnt
                    while forward_cnt > next_cnt:
                        if forward_cnt in render_cache:
                            break
                        forward_cnt -= 1
                    if forward_cnt - next_cnt <= 1:
                        video_write.write(render_cache[next_cnt])
                        next_cnt += 1
                    elif forward_cnt - next_cnt > 1:
                        step = forward_cnt - next_cnt
                        first_rects = rect_cache[next_cnt]
                        last_rects = rect_cache[forward_cnt]
                        for i in range(step):
                            draw_flag = True
                            for j in range(min(len(first_rects), len(last_rects))):
                                first_rect = first_rects[j]
                                last_rect = last_rects[j]
                                delta_x = (last_rect[0] - first_rect[0]) / step
                                delta_y = (last_rect[1] - first_rect[1]) / step
                                if abs(delta_x) > 100 / step or abs(delta_y) > 100 / step:
                                    draw_flag = False
                                    break
                                color = np.random.randint(0, 255, size=(3,))
                                color = [int(c) for c in color]
                                p1 = (first_rect[0] + int(delta_x * i) - 80, first_rect[1] + int(delta_y * i) - 80)
                                p2 = (first_rect[0] + int(delta_x * i) + 100, first_rect[1] + int(delta_y * i) + 100)
                                frame = frame_cache[next_cnt]
                                cv2.rectangle(frame, p1, p2, color, 2)
                            if not draw_flag:
                                frame = frame_cache[next_cnt]
                            video_write.write(frame)
                            next_cnt += 1
                elif next_cnt in frame_cache:
                    video_write.write(frame_cache[next_cnt])
                    next_cnt += 1
                else:
                    logger.info('Lost frame index: [{}]'.format(next_cnt))
                end = time.time()
                if end - start > 30:
                    logger.info('Task time overflow, complete previous render task.')
                    break
            except Exception as e:
                logger.error(e)
        return next_cnt

    def render_task(self, current_idx, render_cache, rect_cache, frame_cache, task_id):
        self.stream_cnt += 1
        current_time = time.strftime('%m-%d-%H-%M-%S-', time.localtime(time.time()))
        target = self.detect_stream_path / (current_time + str(self.stream_cnt) + '.mp4')
        logger.info('Render task [{}]: Writing detection stream frame into: [{}]'.format(task_id, str(target)))
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_write = cv2.VideoWriter(str(target), self.fourcc, 24.0, (1920, 1080), True)
        next_cnt = current_idx - self.future_frames
        next_cnt = self.write_video_work(video_write, next_cnt, current_idx, render_cache, rect_cache,
                                         frame_cache, task_id)
        # the future frames count
        # next_frame_cnt = 48
        # wait the futures frames is accessable
        if not self.next_prepare_event.is_set():
            logger.info('Wait frames accessible....')
            start = time.time()
            # wait the future frames prepared,if ocurring time out, give up waits
            self.next_prepare_event.wait(30)
            logger.info("Wait [{}] seconds".format(time.time() - start))
            logger.info('Frames accessible....')
        # logger.info('Render task Begin with frame [{}]'.format(next_cnt))
        # logger.info('After :[{}]'.format(render_cache.keys()))

        end_cnt = next_cnt + self.future_frames
        next_cnt = self.write_video_work(video_write, next_cnt, end_cnt, render_cache, rect_cache, frame_cache, task_id)
        video_write.release()
        logger.info('Render task [{}]: Done write detection stream frame into: [{}]'.format(task_id, str(target)))
        self.write_done = True
        return True


class ProcessBasedDetectorController(DetectorController):

    def start(self, pool: Pool):
        super().start(pool)
        res = pool.apply_async(self.collect_and_reconstruct, (None,))
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


class TaskBasedDetectorController(ProcessBasedDetectorController):

    def __init__(self, cfg: VideoConfig, stream_path: Path, region_path: Path, frame_path: Path, frame_queue: Queue,
                 index_pool: Queue) -> None:
        super().__init__(cfg, stream_path, region_path, frame_path, frame_queue, index_pool)
        # self.construct_params = ray.put(
        #     ConstructParams(self.result_queue, self.original_frame_cache, self.render_frame_cache,
        #                     self.render_rect_cache, self.stream_render, 500, self.cfg))
        self.construct_params =ConstructParams(self.result_queue, self.original_frame_cache, self.render_frame_cache,
                                               self.render_rect_cache, self.stream_render, 500, self.cfg)


    def init_detectors(self):
        logger.info('Init total [{}] detectors....'.format(self.col * self.row))
        self.detectors = []
        self.detect_params = []
        for i in range(self.col):
            for j in range(self.row):
                region_detector_path = self.region_path / str(self.cfg.index) / (str(i) + '-' + str(j))
                # index = self.col * i + j
                # self.detectors.append(
                #     TaskBasedDetector(self.col_step, self.row_step, i, j, self.cfg, self.send_pipes[index],
                #                       self.receive_pipes[index],
                #                       region_detector_path))
                self.detect_params.append(
                    DetectorParams(self.col_step, self.row_step, i, j, self.cfg, region_detector_path))
        self.result_path = self.region_path / str(self.cfg.index) / 'frames'
        self.detect_stream_path = self.region_path / str(self.cfg.index) / 'streams'
        self.test_path = self.region_path / str(self.cfg.index) / 'tests'
        self.detect_stream_path.mkdir(exist_ok=True, parents=True)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.test_path.mkdir(exist_ok=True, parents=True)
        logger.info('Detectors init done....')

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        empty = np.zeros(self.cfg.shape).astype(np.uint8)
        frame, original_frame = self.preprocess(empty)
        self.col_step = int(frame.shape[0] / self.col)
        self.row_step = int(frame.shape[1] / self.row)
        self.block_info = BlockInfo(self.row, self.col, self.row_step, self.col_step)

    def collect(self, args):
        return [f.get() for f in args]

    def collect_and_reconstruct(self, args):
        results = self.collect(args)
        construct_result: ConstructResult = self.construct(results)
        if construct_result is not None:
            frame = construct_result.frame
            if self.cfg.draw_boundary:
                frame = draw_boundary(frame, self.block_info)
                # logger.info('Done constructing of sub-frames into a original frame....')
            if self.cfg.show_window:
                frame = imutils.resize(frame, width=800)
                cv2.imshow('Reconstructed Frame', frame)
                cv2.waitKey(1)
        else:
            logger.error('Empty reconstruct result.')

        return True

    def dispatch_based_queue(self, pool):
        # start = time.time()
        while True:
            if self.quit:
                break
            frame = self.frame_queue.get()
            self.dispatch_frame(frame, pool)
        return True

    def dispatch_frame(self, frame, pool):
        start = time.time()
        self.frame_cnt.set(self.frame_cnt.get() + 1)
        self.original_frame_cache[self.frame_cnt.get()] = frame
        # self.render_frame_cache[self.frame_cnt.get()] = frame
        # logger.info(self.original_frame_cache.keys())
        frame, original_frame = self.preprocess(frame)
        if self.frame_cnt.get() % self.cfg.sample_rate == 0:
            logger.info('Controller [{}]: Dispatch frame to all detectors....'.format(self.cfg.index))
            async_futures = []
            for d in self.detect_params:
                block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                      self.frame_cnt.get(), original_frame.shape)
                # async_futures.append(pool.apply_async(d.detect_based_task, (block,)))
                # async_futures.append(pool.apply_async(detect_based_task, (block, d,)))
                async_futures.append(detect_based_task.remote(block, d))
            # self.collect_and_reconstruct(async_futures)
            r = pool.apply_async(collect_and_reconstruct,
                                 (async_futures, self.construct_params, self.block_info, self.cfg,))
            r.get()
            # collect_and_reconstruct.remote(async_futures, self.construct_params, self.block_info, self.cfg)
        self.dispatch_cnt += 1
        if self.dispatch_cnt % 100 == 0:
            end = time.time() - start
            logger.info(
                'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
                    self.cfg.index, round(end, 2), round(end / 100, 2)))
            self.dispatch_cnt = 0
        self.clear_original_cache()

    def call_task(self, frame, pool):
        self.dispatch_frame(frame, pool)

    def start(self, pool: Pool):
        self.init_control_range()
        self.init_detectors()
        # self.dispatch_based_queue(pool)
        res = pool.apply_async(self.write_frame_work)
        return res


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

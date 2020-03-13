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

import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from multiprocessing import cpu_count
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from multiprocessing import Pool
import stream
from detection.params import DispatchBlock, ConstructResult, BlockInfo, ConstructParams, DetectorParams
from detection.render import DetectionStreamRender
from stream.websocket import *
from multiprocessing.managers import SharedMemoryManager
from utils.cache import FrameCache
# from utils import NoDaemonPool as Pool
from .capture import *
from stream.rtsp import FFMPEG_VideoStreamer
from .component import stream_pipes
from .detect_funcs import detect_based_task
from .detector import *
from config import ModelType
import time
import imutils


# from pynput.keyboard import Key, Controller, Listener
# import keyboard
# from capture import *


class MonitorType(Enum):
    PROCESS_BASED = 1,
    THREAD_BASED = 2,
    PROCESS_THREAD_BASED = 3
    RAY_BASED = 4
    TASK_BASED = 5


# Monitor will build multiple video stream receivers according the video configuration
class DetectionMonitor(object):

    def __init__(self, cfgs: List[VideoConfig], scfg: ServerConfig, stream_path: Path, sample_path: Path,
                 frame_path: Path,
                 region_path: Path, offline_path: Path = None, build_pool=True) -> None:
        super().__init__()
        # self.cfgs = I.load_video_config(cfgs)[-1:]
        # self.cfgs = I.load_video_config(cfgs)
        # self.cfgs = [c for c in self.cfgs if c.enable]
        # self.cfgs = [c for c in cfgs if enable_options[c.index]]
        self.scfg = scfg
        self.cfgs = cfgs
        self.quit = False
        # Communication Pipe between detector and stream receiver
        self.pipes = [Manager().Queue(c.max_streams_cache) for c in self.cfgs]
        self.time_stamp = generate_time_stamp()
        self.stream_path = stream_path / self.time_stamp
        self.sample_path = sample_path / self.time_stamp
        self.frame_path = frame_path / self.time_stamp
        self.region_path = region_path / self.time_stamp
        self.offline_path = offline_path
        self.process_pool = None
        self.thread_pool = None
        self.shut_down_event = Manager().Event()
        self.shut_down_event.clear()
        self.scheduler = BackgroundScheduler()
        if build_pool:
            # pool_size = min(len(cfgs) * 2, cpu_count() - 1)
            # self.process_pool = Pool(processes=pool_size)
            self.process_pool = Pool(processes=len(self.cfgs) * 5)
            self.thread_pool = ThreadPoolExecutor()
        self.clean()
        self.stream_receivers = [
            stream.StreamReceiver(self.stream_path / str(c.index), offline_path, c, self.pipes[idx]) for idx, c in
            enumerate(self.cfgs)]

        self.scheduler.add_job(self.notify_shut_down, 'cron',
                               month=self.scfg.cron['end']['month'],
                               day=self.scfg.cron['end']['day'],
                               hour=self.scfg.cron['end']['hour'],
                               minute=self.scfg.cron['end']['minute'])
        self.scheduler.start()
        self.frame_cache_manager = SharedMemoryManager()
        self.frame_cache_manager.start()

    def monitor(self):
        self.call()
        self.wait()

    # def set_runtime(self, runtime):
    #     self.runtime = runtime

    def shut_down_from_keyboard(self):
        logger.info('Click Double Enter to shut down system.')
        while True and not self.shut_down_event.is_set():
            c = sys.stdin.read(1)
            logger.info(c)
            if c == '\n':
                self.notify_shut_down()
                break
        # if keycode == Key.enter:
        #     self.shut_down_event.set()

    def shut_down_after(self, runtime=-1):
        if runtime == -1:
            return
        logger.info('System will exit after [{}] seconds'.format(runtime))
        threading.Timer(runtime, self.notify_shut_down).start()

    def notify_shut_down(self):
        if not self.shut_down_event.is_set():
            self.shut_down_event.set()

    def listen(self):
        # Listener(on_press=self.shut_down_from_keyboard).start()
        threading.Thread(target=self.shut_down_from_keyboard, daemon=True).start()

        logger.info('*******************************Monitor: Listening exit event********************************')
        # if self.runtime != -1:
        #     time.sleep(self.runtime)
        # else:
        #     input('')
        self.shut_down_event.wait()
        logger.info('*******************************Monitor: preparing exit system********************************')
        self.cancel()

    def cancel(self):
        pass

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
        if self.process_pool is not None:
            self.process_pool.apply_async(detect,
                                          (self.stream_path / str(cfg.index), self.region_path / str(cfg.index),
                                           self.pipes[i], cfg,))

    # def init_stream_receiver(self, cfg, i):
    #     self.process_pool.apply_async(I.read_stream, (self.stream_path / str(cfg.index), cfg, self.pipes[i],))
    def init_stream_receiver(self, i):
        if self.process_pool is not None:
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
    def __init__(self, cfgs: List[VideoConfig], scfg, stream_path: Path, sample_path: Path, frame_path: Path,
                 region_path, offline_path: Path = None, build_pool=True) -> None:
        super().__init__(cfgs, scfg, stream_path, sample_path, frame_path, region_path, offline_path, build_pool)
        self.caps_queue = [Manager().Queue() for c in self.cfgs]
        self.msg_queue = [Manager().Queue() for c in self.cfgs]
        self.stream_stacks = [Manager().list() for c in self.cfgs]
        self.push_streamers = [PushStreamer(cfg, self.stream_stacks[idx]) for idx, cfg in enumerate(self.cfgs)]
        self.frame_caches = []
        for idx, cfg in enumerate(self.cfgs):
            if cfg.use_sm:
                frame_cache = FrameCache(self.frame_cache_manager, cfg.cache_size,
                                         np.zeros((cfg.shape[1], cfg.shape[0], 3), dtype=np.uint8).nbytes,
                                         shape=cfg.shape)
                self.frame_caches.append(frame_cache)
            else:
                list_cache = Manager().list([None] * cfg.cache_size)
                self.frame_caches.append(list_cache)
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

    def __init__(self, cfgs: Path, scfg, stream_path: Path, sample_path, frame_path, region_path: Path,
                 offline_path: Path = None) -> None:
        super().__init__(cfgs, scfg, stream_path, sample_path, frame_path, region_path, offline_path)

    def init_controllers(self):
        self.controllers = [
            ProcessBasedDetectorController(cfg, self.stream_path / str(cfg.index), self.region_path / str(cfg.index),
                                           self.frame_path / str(cfg.index),
                                           self.caps_queue[idx],
                                           self.pipes[idx]
                                           ) for
            idx, cfg in enumerate(self.cfgs)]
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            res, detect_proc = self.controllers[i].start(self.process_pool)
            res.get()
            logger.info('Done init detector controller [{}]....'.format(cfg.index))


class EmbeddingControlBasedTaskMonitor(EmbeddingControlMonitor):

    def __init__(self, cfgs: List[VideoConfig], scfg, stream_path, sample_path, frame_path, region_path: Path,
                 offline_path=None, build_pool=True) -> None:
        super().__init__(cfgs, scfg, stream_path, sample_path, frame_path, region_path, offline_path)
        # self.classify_model = classify_model
        # HandlerSSD.SSD_MODEL = ssd_model
        self.scfg = scfg
        self.task_futures = []
        for c in self.cfgs:
            c.date = self.time_stamp
            # self.process_pool = None
        # self.thread_pool = None
        self.stream_renders = None
        self.render_notify_queues = [Manager().Queue() for c in self.cfgs]
        self.stream_renders = []

    def init_stream_renders(self):
        self.stream_renders = [
            DetectionStreamRender(c, 0, c.future_frames, self.msg_queue[idx], self.controllers[idx].rect_stream_path,
                                  self.controllers[idx].original_stream_path, self.controllers[idx].render_frame_cache,
                                  self.controllers[idx].render_rect_cache, self.controllers[idx].original_frame_cache,
                                  self.render_notify_queues[idx]) for idx, c
            in
            enumerate(self.cfgs)]

    def init_controllers(self):
        self.controllers = [
            TaskBasedDetectorController(self.scfg, cfg, self.stream_path / str(cfg.index),
                                        self.region_path / str(cfg.index), self.frame_path / str(cfg.index),
                                        self.caps_queue[idx], self.pipes[idx], self.msg_queue[idx],
                                        self.stream_stacks[idx],
                                        self.render_notify_queues[idx], self.frame_caches[idx]) for
            idx, cfg in enumerate(self.cfgs)]
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            # self.task_futures.append(self.controllers[i].start(self.thread_pool))
            self.process_pool.apply_async(self.controllers[i].start, args=(None,))
            # self.controllers[i].start(self.process_pool)
            logger.info('Done init detector controller [{}]....'.format(cfg.index))

    def init_rtsp_caps(self, c, idx):
        self.caps.append(
            VideoRtspCallbackCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                     self.pipes[idx], self.caps_queue[idx], c, idx, self.controllers[idx],
                                     c.sample_rate)
        )

    def init_offline_caps(self, c, idx):
        self.caps.append(
            VideoOfflineCallbackCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                        self.offline_path / str(c.index),
                                        self.pipes[idx],
                                        self.caps_queue[idx], c, idx, self.controllers[idx], self.shut_down_event,
                                        c.sample_rate,
                                        delete_post=False))

    def cancel(self):
        for idx, cap in enumerate(self.caps):
            cap.quit.set()
            self.controllers[idx].quit.set()
            self.push_streamers[idx].quit.set()
            self.stream_renders[idx].quit.set()

    def init_websocket_clients(self):
        for idx, cfg in enumerate(self.cfgs):
            threading.Thread(target=websocket_client, daemon=True,
                             args=(self.msg_queue[idx], cfg, self.scfg)).start()
            logger.info(f'Controller [{cfg.index}]: Websocket client [{cfg.index}] is initializing...')
            # asyncio.get_running_loop().run_until_complete(websocket_client_async(self.msg_queue[idx]))

    def call(self):
        self.init_websocket_clients()
        # Init detector controller
        self.init_controllers()
        self.init_caps()
        self.init_stream_renders()
        # Init stream receiver firstly, ensures video index that is arrived before detectors begin detection..
        for i, cfg in enumerate(self.cfgs):
            res = self.init_stream_receiver(i)
            # logger.debug(res.get())
        # Run video capture from stream
        for i in range(len(self.cfgs)):
            if self.process_pool is not None:
                self.task_futures.append(
                    self.process_pool.apply_async(self.caps[i].read, (self.scfg,)))
                self.task_futures.append(
                    self.process_pool.apply_async(self.push_streamers[i].push_stream, ()))
                self.task_futures.append(self.process_pool.apply_async(self.stream_renders[i].loop_render_msg, ()))
                # self.task_futures[-1].get()

    def wait(self):
        if self.process_pool is not None:
            try:
                # self.listen()
                # logger.info('Waiting processes canceled.')
                self.listen()
                for idx, r in enumerate(self.task_futures):
                    if r is not None:
                        logger.info(
                            '*******************************Controller [{}]: waiting process canceled********************************'.format(
                                self.cfgs[idx].index))
                        r.get()
                        logger.info(
                            '*******************************Controller [{}]: exit********************************'.format(
                                self.cfgs[idx].index))

                # results = [r.get() for r in self.task_futures if r is not None]
                self.frame_cache_manager.shutdown()
                self.process_pool.close()
                self.process_pool.join()
            except:
                self.process_pool.terminate()


class DetectorController(object):
    def __init__(self, cfg: VideoConfig, stream_path: Path, candidate_path: Path, frame_path: Path,
                 frame_queue: Queue,
                 index_pool: Queue,
                 msg_queue: Queue, frame_cache: FrameCache) -> None:
        super().__init__()
        self.cfg = cfg
        self.dol_id = 10000
        self.dol_gone = True
        self.stream_path = stream_path
        self.frame_path = frame_path
        self.candidate_path = candidate_path
        self.block_path = candidate_path / 'blocks'
        self.result_path = self.candidate_path / 'frames'
        self.crop_result_path = self.candidate_path / 'crops'
        self.rect_stream_path = self.candidate_path / 'render-streams'
        self.original_stream_path = self.candidate_path / 'original-streams'
        self.test_path = self.candidate_path / 'tests'
        self.create_workspace()

        self.result_cnt = 0
        self.stream_cnt = 0
        # self.process_pool = process_pool
        self.x_num = cfg.routine['col']
        self.y_num = cfg.routine['row']
        self.x_step = 0
        self.y_step = 0
        self.block_info = BlockInfo(self.y_num, self.x_num, self.y_step, self.x_step)

        # self.send_pipes = [Manager().Queue() for i in range(self.x_num * self.y_num)]
        # self.receive_pipes = [Manager().Queue() for i in range(self.x_num * self.y_num)]
        # self.pipe = stream_pipes[self.cfg.index]
        self.index_pool = index_pool
        self.frame_queue = frame_queue
        self.msg_queue = msg_queue
        self.result_queue = Manager().Queue(self.cfg.max_streams_cache)
        self.init_push = False
        self.quit = Manager().Event()
        self.quit.clear()
        self.status = Manager().Value('i', SystemStatus.SHUT_DOWN)
        self.frame_cnt = Manager().Value('i', 0)
        self.pre_cnt = 0
        # self.frame_cnt =
        self.next_prepare_event = Manager().Event()
        self.next_prepare_event.clear()
        self.pre_detect_index = -self.cfg.future_frames
        self.history_write = False
        # self.original_frame_cache = Manager().dict()
        self.original_frame_cache = Manager().list()
        self.cache_size = self.cfg.cache_size
        # self.original_frame_cache[:] = [None] * self.cache_size
        self.original_frame_cache = frame_cache

        # self.original_hash_cache = Manager().list()
        # self.render_frame_cache = Manager().dict()
        self.render_frame_cache = Manager().list()
        self.render_frame_cache[:] = [None] * self.cache_size

        self.history_frame_deque = Manager().list()

        self.render_rect_cache = Manager().list()
        self.render_rect_cache[:] = [None] * self.cache_size
        # self.render_rect_cache = Manager().dict()
        # self.detect_index = Manager().dict()
        self.record_cnt = 48
        self.render_task_cnt = 0
        self.construct_cnt = 0
        self.dispatch_cnt = 0

        # time scheduler to clear cache
        self.render_events = Manager().dict()
        self.last_detection = -1
        self.continuous_filter_flag = False
        self.runtime = time.time()
        # self.clear_point = 0
        # self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # self.stream_render = DetectionStreamRender(self.cfg, 0, self.cfg.future_frames, self.msg_queue, self)
        # self.video_streamer = FFMPEG_VideoStreamer(self.cfg.push_to, [self.cfg.shape[1], self.cfg.shape[0]], 24)
        self.LOG_PREFIX = f'Controller [{self.cfg.index}]: '
        self.save_cache = {}

        # def __getstate__(self):

    #     self_dict = self.__dict__.copy()
    #     del self_dict['process_pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    def cancel(self):
        pass

    # def clear_original_cache(self):
    #     len_cache = len(self.original_frame_cache)
    #     if len_cache > 1000:
    #         # original_head = self.original_frame_cache.keys()[0]
    #         thread = threading.Thread(
    #             target=clear_cache,
    #             args=(self.original_frame_cache,), daemon=True)
    #         # self.clear_cache(self.original_frame_cache)
    #         thread.start()
    #         logger.info(
    #             'Clear half original frame caches.')
    #
    # def clear_render_cache(self):
    #     # last_detect_internal = time.time() - self.last_detection
    #     # time_thresh = self.cfg.future_frames * 1.5 * 3
    #     # if last_detect_internal > time_thresh and len(self.render_frame_cache) > 500:
    #     if len(self.render_frame_cache) > 500:
    #         thread = threading.Thread(
    #             target=clear_cache,
    #             args=(self.render_frame_cache,), daemon=True)
    #         thread.start()
    #         logger.info('Clear half render frame caches')
    #     if len(self.render_rect_cache) > 500:
    #         thread = threading.Thread(
    #             target=clear_cache,
    #             args=(self.render_rect_cache,), daemon=True)
    #         thread.start()

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        frame = self.frame_queue.get()
        frame, original_frame = preprocess(frame, self.cfg)
        self.x_step = int(frame.shape[1] / self.x_num)
        self.y_step = int(frame.shape[0] / self.y_num)
        self.block_info = BlockInfo(self.y_num, self.x_num, self.y_step, self.x_step)

    def init_detectors(self):
        logger.info('Init total [{}] detectors....'.format(self.x_num * self.y_num))
        self.detectors = []
        for i in range(self.x_num):
            for j in range(self.y_num):
                region_detector_path = self.block_path / (str(i) + '-' + str(j))
                index = self.x_num * i + j
                self.detectors.append(
                    Detector(self.x_step, self.y_step, i, j, self.cfg, self.send_pipes[index],
                             self.receive_pipes[index],
                             region_detector_path))
        logger.info('Detectors init done....')

    def create_workspace(self):

        self.rect_stream_path.mkdir(exist_ok=True, parents=True)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.crop_result_path.mkdir(exist_ok=True, parents=True)
        self.original_stream_path.mkdir(exist_ok=True, parents=True)
        self.block_path.mkdir(exist_ok=True, parents=True)
        self.test_path.mkdir(exist_ok=True, parents=True)

    # def start(self):
    #     self.process_pool.apply_async(self.control, (self,))

    def start(self, pool):
        self.status.set(SystemStatus.RUNNING)
        self.init_control_range()
        self.init_detectors()
        return None

    def write_frame_work(self):
        logger.info(
            '*******************************Controler [{}]: Init detection frame frame routine********************************'.format(
                self.cfg.index))
        if not self.cfg.save_box:
            logger.info(
                '*******************************Controler [{}]: Frame writing routine disabled********************************'.format(
                    self.cfg.index))
            return

        while True:
            if self.status.get() == SystemStatus.SHUT_DOWN and self.result_queue.empty():
                logger.info(
                    '*******************************Controller [{}]: Frame write routine exit********************************'.format(
                        self.cfg.index))
                break
            try:
                # r = self.get_result_from_queue()
                if not self.result_queue.empty():
                    result_queue = self.result_queue.get(timeout=1)
                    r, rects = result_queue[0], result_queue[2]
                    self.result_cnt += 1
                    current_time = generate_time_stamp() + '_'
                    img_name = current_time + str(self.result_cnt) + '.png'
                    target = self.result_path / img_name
                    cv2.imwrite(str(target), r)
                    self.label_crop(r, img_name, rects)
                    self.save_bbox(img_name, rects)
            except Exception as e:
                logger.error(e)
        return True

    def get_result_from_queue(self):
        return self.result_queue.get(timeout=2)

    #
    # def collect_and_reconstruct(self, args, pool):
    #     logger.info('Controller [{}] start collect and construct'.format(self.cfg.index))
    #     cnt = 0
    #     start = time.time()
    #     while True:
    #         if self.status.get() == SystemStatus.SHUT_DOWN:
    #             break
    #         # logger.debug('Collecting sub-frames into a original frame....')
    #         # start = time.time()
    #         results = self.collect(args)
    #         # logger.info('Collect consume [{}]'.format(time.time() - start ))
    #         construct_result: ConstructResult = self.construct(results)
    #         # logger.debug('Done Construct sub-frames into a original frame....')
    #         cnt += 1
    #         if (cnt * self.cfg.sample_rate) % 100 == 0:
    #             end = time.time() - start
    #             logger.info(
    #                 'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
    #                     self.cfg.index, round(end, 2), round(end / 100, 2)))
    #             start = time.time()
    #             cnt = 0
    #         frame = construct_result.frame
    #         if self.cfg.draw_boundary:
    #             frame, _ = preprocess(construct_result.frame, self.cfg)
    #             frame = draw_boundary(frame, self.block_info)
    #         # logger.info('Done constructing of sub-frames into a original frame....')
    #         if self.cfg.show_window:
    #             cv2.imshow('Reconstructed Frame', frame)
    #             cv2.waitKey(1)
    #     return True

    def dispatch(self):
        # start = time.time()
        while True:
            if self.status.get() == SystemStatus.SHUT_DOWN:
                break
            frame = self.frame_queue.get()
            self.frame_cnt.set(self.frame_cnt.get() + 1)
            self.original_frame_cache[self.frame_cnt.get() % self.cache_size] = frame
            # self.render_frame_cache[self.frame_cnt.get()] = frame
            # logger.info(self.original_frame_cache.keys())
            frame, original_frame = preprocess(frame, self.cfg)
            if self.frame_cnt.get() % self.cfg.sample_rate == 0:
                logger.info('Dispatch frame to all detectors....')
                for idx, sp in enumerate(self.send_pipes):
                    sp.put(DispatchBlock(crop_by_se(frame, self.detectors[idx].start, self.detectors[idx].end),
                                         self.frame_cnt.get(), original_frame.shape))

            # self.clear_original_cache()
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

    def construct_rgb(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.x_num, self.y_num, self.x_step, self.y_step, 3))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3, 4))
        constructed_frame = np.reshape(sub_frames, (self.x_num * self.x_step, self.y_num * self.y_step, 3))
        return constructed_frame

    def construct_gray(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.y_num, self.x_num, self.y_step, self.x_step))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3))
        constructed_frame = np.reshape(sub_frames, (self.y_num * self.y_step, self.x_num * self.x_step))
        return constructed_frame

    def save_bbox(self, frame_name, boundary_rect):
        bbox_path = str(self.candidate_path / 'bbox.json')
        self.save_cache[frame_name] = boundary_rect

        if not osp.exists(bbox_path):
            fw = open(bbox_path, 'w')
            fw.write(json.dumps(self.save_cache, indent=4))
            fw.close()

        if len(self.save_cache) == 2:
            fr = open(bbox_path, 'r')
            save_file = json.load(fr)
            fr.close()

            for key in self.save_cache:
                save_file[key] = self.save_cache[key]

            fw = open(bbox_path, 'w')
            fw.write(json.dumps(save_file, indent=4))
            fw.close()

            self.save_cache = {}

    def construct(self, *args) -> ConstructResult:
        pass

    def label_crop(self, frame, label_name, rects):
        # label_w, label_h = 224, 224
        # # p1 = (rect[0] - 80, rect[1] - 80)
        # # p2 = (rect[0] + 100, rect[1] + 100)
        # center_x, center_y = round(rects[0][0] + rects[0][2] / 2), round(rects[0][1] + rects[0][3] / 2)
        # start_x, start_y = round(center_x - label_w / 2), round(center_y - label_h / 2)
        # end_x = start_x + label_w
        # end_y = start_y + label_h
        # # start_x, start_y = rects[0][0] - 77, rects[0][1] - 77
        # # end_x = rects[0][0] + 97
        # # end_y = rects[0][1] + 97
        # if start_x < 0:
        #     start_x = 0
        # if start_y < 0:
        #     start_y = 0
        # if end_x > shape[1]:
        #     end_x = shape[1]
        # if end_y > shape[0]:
        #     end_y = shape[0]
        for idx, rect in enumerate(rects):
            crop_path = self.crop_result_path / (str(idx) + '_' + label_name)
            cropped = crop_by_rect(self.cfg, rect, frame)
            cv2.imwrite(str(crop_path), cropped)


class ProcessBasedDetectorController(DetectorController):

    def start(self, pool: Pool):
        super().start(pool)
        res = pool.apply_async(self.collect_and_reconstruct, (None, None))
        pool.apply_async(self.dispatch, ())
        pool.apply_async(self.write_frame_work)
        logger.info('Running detectors.......')
        detect_proc_res = []
        for idx, d in enumerate(self.detectors):
            logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.x_index, d.y_index))
            detect_proc_res.append(pool.apply_async(d.detect, ()))
            # detect_proc_res.append(pool.submit(d.detect, ()))
            logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.x_index, d.y_index))
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
                    'Submit detector [{},{},{}] task..'.format(self.cfg.index, d.x_index, d.y_index))
                thread_res.append(pool.submit(d.detect))
                # detect_proc_res.append(pool.submit(d.detect, ()))
                logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.x_index, d.y_index))
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        return thread_res
        # self.monitor.wait_pool()
        # self.loop_work()


class TaskBasedDetectorController(ThreadBasedDetectorController):

    def __init__(self, server_cfg: ServerConfig, cfg: VideoConfig, stream_path: Path, candidate_path: Path,
                 frame_path: Path, frame_queue: Queue, index_pool: Queue, msg_queue: Queue, streaming_queue: List,
                 render_notify_queue, frame_cache: FrameCache) -> None:
        super().__init__(cfg, stream_path, candidate_path, frame_path, frame_queue, index_pool, msg_queue, frame_cache)
        # self.construct_params = ray.put(
        #     ConstructParams(self.result_queue, self.original_frame_cache, self.render_frame_cache,
        #                     self.render_rect_cache, self.stream_render, 500, self.cfg))
        self.server_cfg = server_cfg
        self.construct_params = ConstructParams(self.result_queue, self.original_frame_cache, self.render_frame_cache,
                                                self.render_rect_cache, None, 500, self.cfg)
        # self.pool = ThreadPoolExecutor()
        # self.threads = []
        self.push_stream_queue = streaming_queue
        self.display_pipe = Manager().Queue(1000)
        self.detect_params = []
        self.detectors = []
        self.args = Manager().list()
        self.frame_stack = Manager().list()
        # self.stream_render = stream_render
        self.render_notify_queue = render_notify_queue

    def listen(self):
        if self.quit.wait():
            self.status.set(SystemStatus.SHUT_DOWN)
            # self.stream_render.quit.set()

    def init_detectors(self):
        logger.info(
            '*******************************Controller [{}]: Init total [{}] detectors********************************'.format(
                self.cfg.index,
                self.x_num * self.y_num))
        for i in range(self.x_num):
            for j in range(self.y_num):
                region_detector_path = self.block_path / (str(i) + '-' + str(j))
                # index = self.col * i + j
                # self.detectors.append(
                #     TaskBasedDetector(self.col_step, self.row_step, i, j, self.cfg, self.send_pipes[index],
                #                       self.receive_pipes[index],
                #                       region_detector_path))
                self.detect_params.append(
                    DetectorParams(self.x_step, self.y_step, i, j, self.cfg, region_detector_path))
        logger.info(
            '*******************************Controller [{}]: detectors init done ********************************'.format(
                self.cfg.index))

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        empty = np.zeros(self.cfg.shape).astype(np.uint8)
        frame, _ = preprocess(empty, self.cfg)
        self.x_step = int(frame.shape[1] / self.x_num)
        self.y_step = int(frame.shape[0] / self.y_num)
        self.block_info = BlockInfo(self.y_num, self.x_num, self.y_step, self.x_step)

    def collect(self, args):
        return [f.result() for f in args]

    def collect_and_reconstruct(self, *args):
        collect_start = time.time()
        # results = self.collect(args)
        # results = args[0]
        logger.debug(
            'Controller [{}]: Collect consume [{}] seconds'.format(self.cfg.index, time.time() - collect_start))
        construct_result: ConstructResult = self.construct(*args)
        if construct_result is not None:
            frame = construct_result.frame
            if self.cfg.draw_boundary:
                frame, _ = preprocess(frame, self.cfg)
                frame = draw_boundary(frame, self.block_info)
                # logger.info('Done constructing of sub-frames into a original frame....')
            if self.cfg.show_window:
                pass
            # if self.cfg.show_window:
            #     frame = imutils.resize(frame, width=800)
            #     self.display_pipe.put(frame)
            # cv2.imshow('Reconstructed Frame', frame)
            # cv2.waitKey(1)
        else:
            logger.error('Empty reconstruct result.')
        return construct_result

    def dispatch_based_queue(self):
        # start = time.time()
        while True:
            if self.quit:
                break
            frame = self.frame_queue.get()
            self.dispatch_frame(frame)
        return True

    def send(self, frame):
        self.pipe[0].send(frame)

    def post_detect(self, frame, idx) -> List[DetectionResult]:
        sub_results = []
        for d in self.detect_params:
            block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                  idx, frame.shape)
            sub_results.append(detect_based_task(block, d))
        return sub_results

    def construct(self, *args) -> ConstructResult:
        # sub_frames = [r.frame for r in results]
        results = args[0]
        _model = args[1]
        original_frame = args[-1]
        sub_binary = [r.binary for r in results]
        # sub_thresh = [r.thresh for r in results]
        # constructed_frame = self.construct_rgb(sub_frames)
        constructed_binary = self.construct_gray(sub_binary)
        # constructed_thresh = self.construct_gray(sub_thresh)
        logger.debug(f'Controller [{self.cfg.index}]: Construct frames into a original frame....')
        try:
            self.construct_cnt += 1
            current_index = results[0].frame_index
            try_time = 0
            # while current_index not in self.original_frame_cache:
            #     logger.info(
            #         f'Current index: [{current_index}] not in original frame cache.May cache was cleared by timer')
            #     time.sleep(0.1)
            #     try_time += 1
            #     if try_time > 24:
            #         return ConstructResult(None, None, None)
            # logger.info(self.original_frame_cache.keys())
            # original_frame = self.original_frame_cache[current_index]
            render_frame = original_frame.copy()
            push_flag = False
            for r in results:
                if len(r.rects):
                    # if r.frame_index not in self.original_frame_cache:
                    #     logger.info('Unknown frame index: [{}] to fetch frame in cache.'.format(r.frame_index))
                    #     continue
                    # if self.original_frame_cache[r.frame_index % ]
                    #     logger.info('Unknown frame index: [{}] to fetch frame in cache.'.format(r.frame_index))
                    #     continue
                    rects = []
                    r.rects = cvt_rect(r.rects)
                    for rect in r.rects:
                        if len(r.rects) >= 3:
                            logger.info(f'To many rect candidates: [{len(r.rects)}].Abandoned..... ')
                            return ConstructResult(original_frame, None, None, frame_index=current_index)
                        candidate = crop_by_rect(self.cfg, rect, render_frame)

                        start = time.time()
                        # obj_class, output = _model.predict(candidate)
                        model_result = True
                        if not self.cfg.cv_only:
                            obj_class, output = _model.predict(candidate)
                            model_result = (obj_class == 0)
                            logger.debug(
                                self.LOG_PREFIX + f'Model Operation Speed Rate: [{round(1 / (time.time() - start), 2)}]/FPS')
                        if model_result == 0:
                            # logger.info(f'Predict: [{output}]')
                            # print(output.shape)
                            # if True:
                            # is_filtered = self.filter_continuous_detect(current_index, original_frame, len(r.rects),
                            #                                             results)
                            # if is_filtered:
                            #     self.last_detection = current_index
                            #     return ConstructResult(original_frame, constructed_binary, None)
                            logger.info(
                                f'============================Controller [{self.cfg.index}]: Dolphin Detected============================')
                            self.dol_gone = False
                            push_flag = True
                            rects.append(rect)
                            p1, p2 = bbox_points(self.cfg, rect, render_frame.shape)
                            logger.info(f'Dolphin position: TL:[{p1}],BR:[{p2}]')
                            if self.cfg.render:
                                color = np.random.randint(0, 255, size=(3,))
                                color = [int(c) for c in color]
                                # p1, p2 = bbox_points(self.cfg, rect, render_frame.shape)
                                # logger.info(f'Dolphin position: TL:[{p1}],BR:[{p2}]')
                                cv2.putText(render_frame, 'Asaeorientalis', p1,
                                            cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                                cv2.rectangle(render_frame, p1, p2, color, 2)

                    r.rects = rects

                    if push_flag:
                        json_msg = creat_detect_msg_json(video_stream=self.cfg.rtsp, channel=self.cfg.index,
                                                         timestamp=current_index, rects=r.rects, dol_id=self.dol_id)
                        logger.info(f'put detect message in msg_queue...')
                        self.msg_queue.put(json_msg)
                        self.result_queue.put((original_frame, r.frame_index, r.rects))
                        self.render_frame_cache[current_index % self.cache_size] = render_frame
                        self.render_rect_cache[current_index % self.cache_size] = r.rects
                        if self.cfg.render:
                            # threading.Thread(target=self.stream_render.reset, args=(current_index,),
                            #                  daemon=True).start()
                            self.render_notify_queue.put((current_index, 'reset'))
                        # self.last_detection = self.stream_render.detect_index
                        # logger.info(self.LOG_PREFIX + f'Last detection frame index [{self.last_detection}]')
                    else:
                        if not self.dol_gone:
                            empty_msg = creat_detect_empty_msg_json(video_stream=self.cfg.rtsp, channel=self.cfg.index,
                                                                    timestamp=current_index, dol_id=self.dol_id)
                            self.msg_queue.put(empty_msg)
                            logger.info(self.LOG_PREFIX + f'Send empty msg: {empty_msg}')
                            self.dol_id += 1
                            self.dol_gone = True

            if self.cfg.render:
                self.render_notify_queue.put((current_index, 'notify'))
                # threading.Thread(target=self.stream_render.notify, args=(current_index,), daemon=True).start()
            # if not push_flag:
            #     video_streamer.write_frame(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
            # self.clear_render_cache()
            # logger.info(f'Construct detect flag: [{push_flag}]')
            return ConstructResult(render_frame, constructed_binary, None, detect_flag=push_flag, results=results,
                                   frame_index=current_index)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        # finally:
        #     return ConstructResult(render_frame, constructed_binary, None, detect_flag=push_flag, results=results)

    def filter_continuous_detect(self, current_index, original_frame, len_rect, results):
        diff_frame = current_index - self.last_detection
        # if self.continuous_filter_flag:
        #     logger.info(f'Controller [{self.cfg.index}]: Frame [{current_index}] is still in filter window range')
        #     return True
        # if diff_frame > self.cfg.detect_internal:
        #     self.continuous_filter_flag = False
        logger.info(f'Controller [{self.cfg.index}]: Diff frame [{diff_frame}]')
        if (self.last_detection != -1) and (0 <= diff_frame <= self.cfg.detect_internal):
            logger.info(
                f'****************************Controller [{self.cfg.index}]: '
                f'Enter continuous exception handle process at Frame '
                f'[{current_index}]***********************************')

            hit_precision = 0
            similarity_seq = []
            hit_cnt = 0
            start = time.time()
            for idx in range(current_index + 1, current_index + self.cfg.search_window_size):
                # if idx in self.original_frame_cache:
                history_frame = self.original_frame_cache[idx % self.cache_size]
                sub_results = self.post_detect(history_frame, idx)
                for sr_idx, sr in enumerate(sub_results):
                    rl = min(len_rect, len(sr.rects))
                    for rl_idx in range(rl):
                        sr_rgb_patch = crop_by_rect(self.cfg, sr.rects[rl_idx], history_frame)
                        r_rgb_patch = crop_by_rect(self.cfg, results[sr_idx].rects[rl_idx],
                                                   original_frame)
                        similarity = cal_rgb_similarity(sr_rgb_patch, r_rgb_patch, 'ssim')
                        logger.debug(
                            f'Controller [{self.cfg.index}]: Cosine Distance {round(similarity, 2)}')
                        logger.debug(
                            f'Controller [{self.cfg.index}]: Frame [{idx}]: cosine similarity '
                            f'{round(similarity, 2)}')
                        hit_precision += similarity
                        similarity_seq.append(similarity)
                        # cv2.imwrite('data/test/0208/' + str(current_index) + '_r_' + str(
                        #     hit_cnt) + '.png', r_patch)
                        # cv2.imwrite('data/test/0208/' + str(current_index) + '_sr_' + str(
                        #     hit_cnt) + '.png', sr_patch)
                        hit_cnt += 1
            if hit_cnt:
                hit_precision /= hit_cnt
                logger.info(
                    f'Controller [{self.cfg.index}]: Hit count {hit_cnt}')

                logger.info(
                    f'Controller [{self.cfg.index}]: Average hit precision {round(hit_precision, 2)}')

            seq_std = cal_std_similarity(similarity_seq)
            logger.info(
                f'Controller [{self.cfg.index}]: Frame sequence std: {round(seq_std, 4)}')
            logger.info(
                self.LOG_PREFIX + f'continuous exception handle process consumes [{round(time.time() - start, 2)}]s')

            if hit_cnt and seq_std <= self.cfg.similarity_thresh:
                logger.info(
                    f'Controller [{self.cfg.index}]: Continuous detection report at ' +
                    f'Frame [{current_index}].Skipped by continuous exception filter rule.............')
                # self.continuous_filter_flag = True
                return True
            return False

    def loop_stack(self):
        classifier = None
        ssd_detector = None
        if self.server_cfg.detect_mode == ModelType.SSD:
            ssd_detector = SSDDetector(model_path=self.server_cfg.detect_model_path, device_id=self.server_cfg.cd_id)
            ssd_detector.run()
            logger.info(
                f'*******************************Capture [{self.cfg.index}]: Running SSD Model********************************')
        elif self.server_cfg.detect_mode == ModelType.CLASSIFY:
            classifier = DolphinClassifier(model_path=self.server_cfg.classify_model_path,
                                           device_id=self.server_cfg.dt_id)
            classifier.run()
            logger.info(
                f'*******************************Capture [{self.cfg.index}]: Running Classifier Model********************************')
        logger.info(
            '*******************************Controller [{}]: Init Loop Stack********************************'.format(
                self.cfg.index))
        pre_index = 0
        while self.status.get() == SystemStatus.RUNNING:
            try:
                s = time.time()
                if not len(self.frame_stack):
                    continue
                _, current_index = self.frame_stack.pop()
                frame = self.original_frame_cache[current_index % self.cache_size]
                if current_index < pre_index:
                    continue
                pre_index = current_index
                e = 1 / (time.time() - s)
                logger.info(self.LOG_PREFIX + f'Stack Pop Speed: [{round(e, 2)}]/FPS')
                self.dispatch_frame(frame, None, ssd_detector, classifier, current_index)
            except Exception as e:
                logger.error(e)
                # pass
                # logger.info(self.LOG_PREFIX + 'Empty Frame Stack')
        logger.info(
            '*******************************Controller [{}]: Loop Stack Exit********************************'.format(
                self.cfg.index))

    def dispatch_to_stack(self, *args):
        # if not len(self.args):
        #     self.args.append(args)
        if len(self.frame_stack) > 2000:
            self.frame_stack[:] = []
        frame = args[0]
        self.frame_cnt.set(self.frame_cnt.get() + 1)
        s = time.time()
        # if frame.shape[1] > 1920:
        #     frame = imutils.resize(frame, width=1920)
        # ray.put(frame)
        self.original_frame_cache[self.frame_cnt.get() % self.cache_size] = frame
        # self.original_hash_cache.append(frame)
        e = 1 / (time.time() - s)
        logger.info(self.LOG_PREFIX + f'Dict Put Speed: [{round(e, 2)}]/FPS')
        s = time.time()
        # self.frame_stack.append((frame, self.frame_cnt.get()))
        self.frame_stack.append((None, self.frame_cnt.get()))
        e = 1 / (time.time() - s)
        logger.info(self.LOG_PREFIX + f'Stack Put Speed: [{round(e, 2)}]/FPS')
        # logger.debug(self.LOG_PREFIX + f'Current Stack Size: [{len(self.frame_stack)}]')

    def dispatch_frame(self, *args):
        start = time.time()
        original_frame = args[0]
        self.pre_cnt = args[-1]
        # self.frame_cnt.set(self.frame_cnt.get() + 1)
        # self.original_frame_cache[self.frame_cnt.get()] = frame
        # if self.frame_cnt.get() <= self.cfg.pre_cache:
        #     return

        # if not self.init_push and self.frame_cnt.get() > 240:
        #     threading.Thread(target=self.push_stream, daemon=True).start()
        #     self.init_push = True

        # original_frame = self.original_frame_cache[self.pre_cnt]

        if self.server_cfg.detect_mode == ModelType.CLASSIFY:
            self.classify_based(args, original_frame.copy())
        elif self.server_cfg.detect_mode == ModelType.SSD:
            self.ssd_based(args, original_frame.copy())
        elif self.server_cfg.detect_mode == ModelType.FORWARD:
            self.forward(args, original_frame)
        # self.clear_original_cache()

    def forward(self, args, original_frame):
        if self.cfg.push_stream:
            self.push_stream_queue.append((original_frame, None, self.pre_cnt))

    def ssd_based(self, args, original_frame):
        ssd_model = args[2]
        try:
            if self.pre_cnt % self.cfg.sample_rate == 0:
                start = time.time()
                frames_results = ssd_model([original_frame])
                logger.debug(
                    self.LOG_PREFIX + f'Model Operation Speed Rate: [{round(1 / (time.time() - start), 2)}]/FPS')
                render_frame = original_frame.copy()
                detect_results = []
                detect_flag = False
                current_index = self.pre_cnt
                if len(frames_results):
                    for frame_result in frames_results:
                        if len(frame_result):
                            rects = [r for r in frame_result if r[4] > 0.7]
                            if len(rects):
                                if len(rects) >= 3:
                                    logger.info(f'To many rect candidates: [{len(rects)}].Abandoned..... ')
                                    return ConstructResult(original_frame, None, None, frame_index=self.pre_cnt)
                                detect_results.append(DetectionResult(rects=rects))
                                detect_flag = True
                                self.dol_gone = False
                                logger.info(
                                    f'============================Controller [{self.cfg.index}]: Dolphin Detected============================')
                                logger.info(f'put detect message in msg_queue...')
                                for rect in rects:
                                    p1, p2 = bbox_points(self.cfg, rect, render_frame.shape)
                                    logger.info(f'Dolphin position: TL:[{p1}],BR:[{p2}]')
                                    if self.cfg.render:
                                        color = np.random.randint(0, 255, size=(3,))
                                        color = [int(c) for c in color]
                                        cv2.putText(render_frame, 'Asaeorientalis', p1,
                                                    cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                                        cv2.rectangle(render_frame, p1, p2, color, 2)
                        if detect_flag:
                            json_msg = creat_detect_msg_json(video_stream=self.cfg.rtsp, channel=self.cfg.index,
                                                             timestamp=current_index, rects=rects,
                                                             dol_id=self.dol_id)
                            self.msg_queue.put(json_msg)
                            self.result_queue.put((render_frame, current_index, rects))
                            self.render_frame_cache[current_index % self.cache_size] = render_frame
                            self.render_rect_cache[current_index % self.cache_size] = rects
                            if self.cfg.render:
                                threading.Thread(target=self.stream_render.reset, args=(current_index,),
                                                 daemon=True).start()
                            self.last_detection = self.stream_render.detect_index
                            logger.info(self.LOG_PREFIX + f'Last detection frame index [{self.last_detection}]')
                        else:
                            if not self.dol_gone:
                                empty_msg = creat_detect_empty_msg_json(video_stream=self.cfg.rtsp,
                                                                        channel=self.cfg.index,
                                                                        timestamp=current_index, dol_id=self.dol_id)
                                self.dol_id += 1
                                self.msg_queue.put(empty_msg)
                                self.dol_gone = True
                if self.cfg.render:
                    threading.Thread(target=self.stream_render.notify, args=(current_index,), daemon=True).start()
                construct_result = ConstructResult(None, None, None, None, detect_flag, detect_results,
                                                   frame_index=self.pre_cnt)
                if self.cfg.push_stream:
                    self.push_stream_queue.append((original_frame, construct_result, self.pre_cnt))
            else:
                if self.cfg.push_stream:
                    self.push_stream_queue.append((original_frame, None, self.pre_cnt))
        except Exception as e:
            traceback.print_stack()
            logger.info(e)

    def classify_based(self, args, original_frame):
        if self.pre_cnt % self.cfg.sample_rate == 0:
            logger.debug('Controller [{}]: Dispatch frame to all detectors....'.format(self.cfg.index))
            async_futures = []
            try:
                frame, original_frame = preprocess(original_frame, self.cfg)
                for d in self.detect_params:
                    block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                          self.pre_cnt, original_frame.shape)
                    # async_futures.append(pool.apply_async(d.detect_based_task, (block,)))
                    async_futures.append(detect_based_task(block, d))
                    # detect_td = threading.Thread(
                    #     target=detect_based_task,
                    #     args=(detect_based_task, block, d))
                    # async_futures.append(detect_td.start())
                    # async_futures.append(self.pool.submit(detect_based_task, block, d))
                    # async_futures.append(detect_based_task.remote(block, d))
                proc_res: ConstructResult = self.collect_and_reconstruct(async_futures, args[3], original_frame)
                if self.cfg.push_stream:
                    self.push_stream_queue.append((proc_res.frame, proc_res, proc_res.frame_index))
            except Exception as e:
                traceback.print_stack()
                logger.error(e)
        else:
            try:
                if self.cfg.push_stream:
                    self.push_stream_queue.append((original_frame, None, self.pre_cnt))
            except Exception as e:
                logger.error(e)

    def display(self):
        logger.info(
            '*******************************Controller [{}]: Init video player********************************'.format(
                self.cfg.index))
        while True:
            if self.status.get() == SystemStatus.SHUT_DOWN:
                logger.info(
                    '*******************************Controller [{}]: Video player exit********************************'.format(
                        self.cfg.index))
                break
            try:
                if not self.display_pipe.empty():
                    frame = self.display_pipe.get(timeout=1)
                    cv2.imshow('Controller {}'.format(self.cfg.index), frame)
                    cv2.waitKey(1)
            except Exception as e:
                logger.error(e)
        return True

    def start(self, pool: Pool):
        self.status.set(SystemStatus.RUNNING)
        self.init_control_range()
        self.init_detectors()
        # self.dispatch_based_queue(pool)
        # res = self.pool.submit(self.write_frame_work, ())
        threading.Thread(target=self.listen, daemon=True).start()
        threading.Thread(target=self.write_frame_work, daemon=True).start()
        # threading.Thread(target=self.display, daemon=True).start()
        # threading.Thread(target=self.loop_stack, daemon=True).start()
        self.loop_stack()
        return True


class PushStreamer(object):
    def __init__(self, cfg: VideoConfig, stream_stack: List) -> None:

        super().__init__()
        self.cfg = cfg
        self.stream_stack = stream_stack
        self.LOG_PREFIX = f'Push Streamer [{self.cfg.index}]: '
        self.quit = Manager().Event()
        self.quit.clear()
        self.status = Manager().Value('i', SystemStatus.RUNNING)

    def listen(self):
        if self.quit.wait():
            self.status.set(SystemStatus.SHUT_DOWN)
            # self.stream_render.quit.set()

    def push_stream(self):
        logger.info(
            f'*******************************Controller [{self.cfg.index}]: Init push stream service********************************')
        draw_cnt = 0
        tmp_results = []
        video_streamer = FFMPEG_VideoStreamer(self.cfg.push_to, size=(self.cfg.shape[1], self.cfg.shape[0]), fps=25,
                                              codec='h264', )
        video_streamer.write_frame(np.zeros((self.cfg.shape[1], self.cfg.shape[0], 3), dtype=np.uint8))
        # time.sleep(6)
        pre_index = 0
        threading.Thread(target=self.listen, daemon=True).start()
        while self.status.get() == SystemStatus.RUNNING:
            try:
                ps = time.time()
                # if self.status.get() == SystemStatus.SHUT_DOWN:
                #     video_streamer.close()
                #     break
                # se = 1 / (time.time() - ps)
                # logger.debug(self.LOG_PREFIX + f'Get Signal Speed Rate: [{round(se, 2)}]/FPS')
                # gs = time.time()
                frame, proc_res, frame_index = self.stream_stack.pop()
                logger.debug(f'Push Streamer [{self.cfg.index}]: Cache queue size: [{len(self.stream_stack)}]')

                if len(self.stream_stack) > 1000:
                    self.stream_stack[:] = []
                    logger.info(self.LOG_PREFIX + 'Too much frames blocked in stream queue.Cleared')
                    continue

                if pre_index < frame_index:
                    pre_index = frame_index
                else:
                    continue

                # end = 1 / (time.time() - gs)
                # logger.debug(self.LOG_PREFIX + f'Get Frame Speed Rate: [{round(end, 2)}]/FPS')
                detect_flag = (proc_res is not None and proc_res.detect_flag)
                # logger.info(f'Draw cnt: [{draw_cnt}]')
                # if proc_res is not None:
                #     logger.info(f'Detect flag: [{proc_res.detect_flag}]')
                # ds = time.time()
                if detect_flag:
                    # logger.info('Detect flag~~~~~~~~~~')
                    draw_cnt = 0
                    tmp_results = proc_res.results
                is_draw_over = draw_cnt <= 36
                if is_draw_over:
                    # logger.info('Draw next frames~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    for r in tmp_results:
                        for rect in r.rects:
                            color = np.random.randint(0, 255, size=(3,))
                            color = [int(c) for c in color]
                            p1, p2 = bbox_points(self.cfg, rect, frame.shape)
                            # p1 = (int(rect[0]), int(rect[1]))
                            # p2 = (int(rect[2]), int(rect[3]))

                            cv2.putText(frame, 'Asaeorientalis', p1,
                                        cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                            cv2.rectangle(frame, p1, p2, color, 2)
                            # if self.server_cfg.detect_mode == ModelType.SSD:
                            #     cv2.putText(frame, str(round(r[4], 2)), (p2[0], p2[1]),
                            #                 cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                    draw_cnt += 1
                if self.cfg.write_timestamp:
                    time_stamp = generate_time_stamp("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, time_stamp, (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # de = 1 / (time.time() - ds)
                # logger.debug(self.LOG_PREFIX + f'Draw Speed Rate: [{round(de, 2)}]/FPS')
                # logger.info(f'Frame index [{frame_index}]')
                # if frame_index % self.cfg.sample_rate == 0:
                #     for _ in range(2):
                #         video_streamer.write_frame(frame)
                # else:
                #     video_streamer.write_frame(frame)
                # end = 1 / (time.time() - ps)
                # ws = time.time()
                video_streamer.write_frame(frame)
                # w_end = 1 / (time.time() - ws)
                end = 1 / (time.time() - ps)
                # logger.debug(self.LOG_PREFIX + f'Writing Speed Rate: [{round(w_end, 2)}]/FPS')
                logger.info(f'Streamer [{self.cfg.index}]: Streaming Speed Rate: [{round(end, 2)}]/FPS')
            except Exception as e:
                pass
                # logger.warning(e)
        logger.info(
            '*******************************Controller [{}]:  Push stream service exit********************************'.format(
                self.cfg.index))

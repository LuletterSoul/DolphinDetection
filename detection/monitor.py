#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: monitor.py
@time: 3/18/20 5:04 PM
@version 1.0
@desc:
"""
import sys
import threading
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
from multiprocessing import Manager, Pool
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import List
import numpy as np
import os
import traceback

from apscheduler.schedulers.background import BackgroundScheduler

import stream
from config import VideoConfig, ServerConfig
from .capture import VideoOfflineCapture, VideoOnlineSampleCapture, VideoRtspCapture, VideoRtspVlcCapture, \
    VideoOfflineVlcCapture
from pysot.tracker.service import TrackingService
from stream.rtsp import PushStreamer
from stream.websocket import websocket_client
from utils import generate_time_stamp, logger, clean_dir
from utils.cache import SharedMemoryFrameCache, ListCache
from .capture import VideoRtspCallbackCapture, \
    VideoOfflineCallbackCapture
from .controller import TaskBasedDetectorController, detect
from .render import DetectionSignalHandler, DetectionStreamRender


class MonitorType(Enum):
    PROCESS_BASED = 1,
    THREAD_BASED = 2,
    PROCESS_THREAD_BASED = 3
    RAY_BASED = 4
    TASK_BASED = 5


class DetectionMonitor(object):
    """
    A global manager that creates camera workdirs based on the camera's configuration file,
    initializes many services, and defines basic state control functions.
    """

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
        self.time_stamp = generate_time_stamp('%m%d')
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
            # build service
            # pool_size = min(len(cfgs) * 2, cpu_count() - 1)
            # self.process_pool = Pool(processes=pool_size)
            self.process_pool = Pool(processes=len(self.cfgs) * 5)
            self.thread_pool = ThreadPoolExecutor()
        # self.clean()
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

    def init_stream_receiver(self, i):
        if self.process_pool is not None:
            return self.process_pool.apply_async(self.stream_receivers[i].receive_online)

    def clean(self):
        clean_dir(self.sample_path)
        clean_dir(self.stream_path)
        clean_dir(self.region_path)


class DetectionRecorder(object):
    def __init__(self, save_dir, timestamp):
        self.save_dir = save_dir
        self.timestamp = timestamp
        self.detect_time = Manager().Value('i', 0)

    def record(self):
        self.detect_time.set(self.detect_time.get() + 1)
        with open(f'{self.save_dir}/{self.timestamp}.txt', 'w') as f:
            f.write(str(self.detect_time.get()))


class EmbeddingControlMonitor(DetectionMonitor):
    """
    add controller embedding
    """

    def __init__(self, cfgs: List[VideoConfig], scfg, stream_path: Path, sample_path: Path, frame_path: Path,
                 region_path, offline_path: Path = None, build_pool=True) -> None:
        super().__init__(cfgs, scfg, stream_path, sample_path, frame_path, region_path, offline_path, build_pool)
        self.caps_queue = [Manager().Queue() for c in self.cfgs]
        self.msg_queue = [Manager().Queue() for c in self.cfgs]
        self.stream_stacks = []
        self.frame_caches = []
        self.init_caches()
        self.push_streamers = [PushStreamer(cfg, self.stream_stacks[idx]) for idx, cfg in enumerate(self.cfgs)]
        # self.stream_stacks = [Manager().list() for c in self.cfgs]
        self.caps = []
        self.controllers = []
        self.recorder = DetectionRecorder(self.region_path, self.time_stamp)

    def init_caches(self):
        for idx, cfg in enumerate(self.cfgs):
            template = np.zeros((cfg.shape[1], cfg.shape[0], cfg.shape[2]), dtype=np.uint8)
            if cfg.use_sm:
                frame_cache = SharedMemoryFrameCache(self.frame_cache_manager, cfg.cache_size,
                                                     template.nbytes,
                                                     shape=cfg.shape)
                self.frame_caches.append(frame_cache)
                self.stream_stacks.append(
                    [frame_cache,
                     Manager().list()])
            else:
                self.frame_caches.append(ListCache(Manager(), cfg.cache_size, template))
                self.stream_stacks.append(Manager().list())

    def init_caps(self):
        """
        load different video captures according mode
        :return:
        """
        for idx, c in enumerate(self.cfgs):
            if c.online == "http":
                self.init_http_caps(c, idx)
            elif c.online == "rtsp":
                self.init_rtsp_caps(c, idx)
            elif c.online == 'vlc_rtsp':
                self.init_vlc_rtsp_caps(c, idx)
            elif c.online == 'vlc_offline':
                self.init_vlc_offline_caps(c, idx)
            elif c.online == 'offline':
                self.init_offline_caps(c, idx)

    def init_offline_caps(self, c, idx):
        """
        use offline video capture
        :param c:
        :param idx:
        :return:
        """
        self.caps.append(
            VideoOfflineCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                self.offline_path / str(c.index),
                                self.pipes[idx],
                                self.caps_queue[idx], c, idx, c.sample_rate,
                                delete_post=False))

    def init_http_caps(self, c, idx):
        """

        :param c:
        :param idx:
        :return:
        """
        self.caps.append(
            VideoOnlineSampleCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                     self.pipes[idx],
                                     self.caps_queue[idx],
                                     c, idx, c.sample_rate))

    def init_vlc_rtsp_caps(self, c, idx):
        pass

    def init_vlc_offline_caps(self, c, idx):
        pass

    def init_rtsp_caps(self, c, idx):
        """
        use rtsp video capture
        :param c:
        :param idx:
        :return:
        """
        self.caps.append(
            VideoRtspCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                             self.pipes[idx], self.caps_queue[idx], c, idx, c.sample_rate)
        )

    def call(self):
        """
        run some components
        :return:
        """
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
        """
        define a abstract method, overrided by subclass
        :return:
        """
        pass

        # Concurrency based multi processes


class EmbeddingControlBasedTaskMonitor(EmbeddingControlMonitor):
    """
    creates many heavy services and run them in a process pool
    """

    def __init__(self, cfgs: List[VideoConfig], scfg, stream_path, sample_path, frame_path, region_path: Path,
                 offline_path=None, build_pool=True) -> None:
        super().__init__(cfgs, scfg, stream_path, sample_path, frame_path, region_path, offline_path)
        # self.classify_model = classify_model
        # HandlerSSD.SSD_MODEL = ssd_model
        self.scfg = scfg
        self.task_futures = []
        self.pipe_manager = Manager()
        for c in self.cfgs:
            c.date = self.time_stamp
            # self.process_pool = None
        # self.thread_pool = None
        self.render_notify_queues = [self.pipe_manager.Queue() for c in self.cfgs]
        self.stream_renders = []
        self.track_service = None
        self.track_requester = None
        self.detect_handlers = []
        self.track_input_pipe = self.pipe_manager.Queue()

    def init_track_poster(self):
        """
        init tracking service
        :param cfgs:
        :return:
        """
        self.track_service = TrackingService(self.scfg.track_cfg_path, self.cfgs, self.scfg.track_model_path,
                                             self.frame_caches)
        self.track_service.run()
        self.track_requester = self.track_service.get_request_instance()

    def init_stream_renders(self):
        """
        init rendering service
        :return:
        """
        self.stream_renders = [
            DetectionStreamRender(c, self.scfg, 0, c.future_frames, self.msg_queue[idx],
                                  self.controllers[idx].rect_stream_path,
                                  self.controllers[idx].original_stream_path,
                                  self.controllers[idx].render_rect_cache, self.controllers[idx].original_frame_cache,
                                  self.render_notify_queues[idx], self.region_path / str(c.index),
                                  self.controllers[idx].preview_path,
                                  self.controllers[idx].detect_params) for idx, c
            in
            enumerate(self.cfgs)]

    def init_detection_signal_handler(self):
        """
        init detection signal forward process service
        :return:
        """
        self.detect_handlers = [
            DetectionSignalHandler(c, self.scfg, 0, c.search_window_size, self.msg_queue[idx],
                                   self.controllers[idx].rect_stream_path,
                                   self.controllers[idx].original_stream_path,
                                   self.controllers[idx].render_rect_cache, self.controllers[idx].original_frame_cache,
                                   self.render_notify_queues[idx], self.region_path / str(c.index),
                                   self.controllers[idx].preview_path,
                                   self.track_requester, self.render_notify_queues[idx],
                                   self.controllers[idx].detect_params) for idx, c
            in
            enumerate(self.cfgs)]
        for idx, h in enumerate(self.detect_handlers):
            self.controllers[idx].init_detect_handler(h)

    def init_controllers(self):
        """
        init detection service
        :return:
        """
        self.controllers = [
            TaskBasedDetectorController(self.scfg, cfg, self.stream_path / str(cfg.index),
                                        self.region_path / str(cfg.index), self.frame_path / str(cfg.index),
                                        self.caps_queue[idx], self.pipes[idx], self.msg_queue[idx],
                                        self.stream_stacks[idx],
                                        self.render_notify_queues[idx], self.frame_caches[idx],
                                        self.recorder)
            for
            idx, cfg in enumerate(self.cfgs)]

    def init_rtsp_caps(self, c, idx):
        """
        init online rtsp video stream reciever
        :param c:
        :param idx:
        :return:
        """
        self.caps.append(
            VideoRtspCallbackCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                     self.pipes[idx], self.caps_queue[idx], c, idx, self.controllers[idx],
                                     c.sample_rate)
        )

    def init_vlc_rtsp_caps(self, c, idx):
        """
        init online rtsp video stream reciever
        :param c:
        :param idx:
        :return:
        """
        self.caps.append(
            VideoRtspVlcCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                self.pipes[idx], self.caps_queue[idx], c, idx, self.controllers[idx],
                                c.sample_rate)
        )

    def init_vlc_offline_caps(self, c, idx):
        """

        init offline vlc video stream reciever
        Args:
            c:
            idx:
        Returns:
        """
        self.caps.append(
            VideoOfflineVlcCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                   self.offline_path / str(c.index),
                                   self.pipes[idx],
                                   self.caps_queue[idx], c, idx, self.controllers[idx], self.shut_down_event,
                                   c.sample_rate,
                                   delete_post=False))

    def init_offline_caps(self, c, idx):
        """
        init offline rtsp video stream reciever
        :param c:
        :param idx:
        :return:
        """
        self.caps.append(
            VideoOfflineCallbackCapture(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                        self.offline_path / str(c.index),
                                        self.pipes[idx],
                                        self.caps_queue[idx], c, idx, self.controllers[idx], self.shut_down_event,
                                        c.sample_rate,
                                        delete_post=False))

    def cancel(self):
        total_detection_cnt = 0
        try:
            for idx, cap in enumerate(self.caps):
                total_detection_cnt += self.controllers[idx].global_index.get()
                print(total_detection_cnt)
                cap.quit.set()
                self.controllers[idx].quit.set()
                self.push_streamers[idx].quit.set()
                self.stream_renders[idx].quit.set()
                if self.cfgs[idx].use_sm:
                    self.frame_caches[idx].close()
                    # self.stream_stacks[idx][0].close()
            self.track_service.cancel()
        except Exception as e:
            traceback.print_exc()
        # persist the number of  total detection frames
        print(f'Write total detection cnt: {total_detection_cnt}')
        with open(f'{self.region_path}/{self.time_stamp}_total_cnt.txt', 'w') as f:
            f.write(str(total_detection_cnt))

    def init_websocket_clients(self):
        """
        run websocket service
        :return:
        """
        for idx, cfg in enumerate(self.cfgs):
            threading.Thread(target=websocket_client, daemon=True,
                             args=(self.msg_queue[idx], cfg, self.scfg)).start()
            logger.info(f'Controller [{cfg.index}]: Websocket client [{cfg.index}] is initializing...')
            # asyncio.get_running_loop().run_until_complete(websocket_client_async(self.msg_queue[idx]))

    def call(self):
        # service initialization order is important,some services depend on other services
        self.init_websocket_clients()
        self.init_track_poster()
        # Init detector controller
        self.init_controllers()
        self.init_caps()
        self.init_stream_renders()
        self.init_detection_signal_handler()
        self.run_processes()

    def run_processes(self):
        """
        Run each core system service in sub-process.The system uses a process pool to maintain all subprocess
        But sub-processes will be terminated sometimes due to an underlying exception which cannot be captured
        by daemon process,we should call each future return value of sub-process explicitly, to
        obtain the exception detail.
        :return:
        """
        for i, cfg in enumerate(self.cfgs):
            res = self.init_stream_receiver(i)
            # logger.debug(res.get())
        # Run video capture from stream
        for i, cfg in enumerate(self.cfgs):
            logger.info('Init detector controller [{}]....'.format(cfg.index))
            task_future = self.process_pool.apply_async(self.controllers[i].start, args=(None,))
            # don't call future object get() method before
            # all sub process is init,otherwise will be blocked probably due to some services are looping
            # task_future.get()
            logger.info('Done init detector controller [{}]....'.format(cfg.index))
        for i in range(len(self.cfgs)):
            if self.process_pool is not None:
                self.task_futures.append(
                    self.process_pool.apply_async(self.caps[i].read, (self.scfg,)))
                # self.task_futures[-1].get()
                # self.task_futures.append(
                #    self.process_pool.apply_async(self.push_streamers[i].push_stream, ()))
                self.task_futures.append(self.process_pool.apply_async(self.stream_renders[i].
                                                                       loop, ()))
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
                        # r.get()
                        logger.info(
                            '*******************************Controller [{}]: exit********************************'.format(
                                self.cfgs[idx].index))

                # results = [r.get() for r in self.task_futures if r is not None]
                self.frame_cache_manager.shutdown()
                self.process_pool.close()
                self.process_pool.join()
            except:
                self.process_pool.terminate()

#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: capture.py
@time: 2019/11/24 11:25
@version 1.0
@desc:
"""
import os
import threading
from multiprocessing.queues import Queue
from pathlib import Path

import cv2

from config import VideoConfig
from utils import logger
import time
import shutil


class VideoCaptureThreading:
    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 sample_rate=5, width=640, height=480, delete_post=True):
        self.cfg = cfg
        self.video_path = video_path
        self.sample_path = sample_path
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
        self.runtime = 0
        self.posix = None

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
            logger.info('Video path not exist: [{}]'.format(self.src))
            return -1
        return self.src

    def update(self):
        cnt = 0
        start = time.time()
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
            self.runtime = time.time() - start
        logger.info('Video Capture [{}]: cancel..'.format(self.cfg.index))

    def update_capture(self, cnt):
        logger.debug('Read frame done from [{}].Has loaded [{}] frames'.format(self.src, cnt))
        logger.debug('Read next frame from video ....')
        self.cap.release()
        self.handle_history()
        src = self.load_next_src()
        if src == -1:
            self.started = False
        self.cap = cv2.VideoCapture(src)

    def handle_history(self):
        if self.posix.exists() and self.delete_post:
            self.posix.unlink()

    def post_update(self):
        pass

    def post_frame_process(self, frame):
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
    def __init__(self, video_path: Path, sample_path: Path, offline_path: Path, index_pool: Queue, frame_queue: Queue,
                 cfg: VideoConfig, sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, sample_rate, width, height, delete_post)
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


# Sample video stream at intervals
class VideoOnlineSampleCapture(VideoCaptureThreading):
    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, sample_rate, width, height, delete_post)
        self.sample_cnt = 0
        self.sample_path.mkdir(exist_ok=True, parents=True)

    def handle_history(self):
        if int(self.runtime / 60 + 1) % self.cfg.sample_internal == 0:
            current_time = time.strftime('%m-%d-%H:%M-', time.localtime(time.time()))
            filename = os.path.basename(str(self.posix))
            target = self.sample_path / (current_time + filename)
            logger.info('Sample video stream into: [{}]'.format(target))
            shutil.copy(self.posix, target)
        super().handle_history()


# Read stream from rtsp
class VideoRtspCapture(VideoOnlineSampleCapture):
    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig, 
                 sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, sample_rate, width, height, delete_post)
        self.sample_path.mkdir(exist_ok=True, parents=True)
        self.saved_time = ""

    def load_next_src(self):
        logger.debug("Loading next video rtsp stream ....")
        return self.cfg.rtsp
        
    def handle_history(self):
        pass

    def update(self):
        cnt = 0
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.update_capture(cnt)
                cnt = 0
                continue
            self.post_frame_process(frame)
            cnt += 1
        logger.info("Video Capture [{}]: process..".format(self.cfg.index))

    def post_frame_process(self, frame):
        self.sample_cnt += 1
        if self.sample_cnt % self.cfg.rtsp_saved_per_frame == 0:
            current_time = time.strftime('%m-%d-%H-%M-', time.localtime(time.time()))
            if current_time != self.saved_time:
                self.sample_cnt = 0
            self.saved_time = current_time
            target = self.sample_path / (current_time + str(self.sample_cnt) + '.png')
            logger.info("Sample rtsp video stream into: [{}]".format(target))
            cv2.imwrite(str(target), frame)
            
            
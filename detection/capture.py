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
            logger.info('Video path not exist: [{}]'.format(self.src))
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

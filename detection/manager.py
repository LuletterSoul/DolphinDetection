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

from multiprocessing import Manager, Pool, Queue
import interface as I
from pathlib import Path
from config import VideoConfig
from utils import *
from typing import List
from utils import clean_dir, logger
from detection import Detector
import threading
import cv2
import imutils


class DetectionMonitor(object):

    def __init__(self, video_config_path: Path, stream_path: Path, candidate_path: Path) -> None:
        super().__init__()
        self.cfgs = I.load_video_config(video_config_path)[:1]
        # Communication Pipe between detector and stream receiver
        self.pipes = [Manager().Queue() for c in self.cfgs]
        self.stream_path = stream_path
        self.candidate_path = candidate_path
        self.pool = Pool()

    def monitor(self):
        for i, cfg in enumerate(self.cfgs):
            # clean all legacy streams and candidates files before initialization
            clean_dir(self.stream_path)
            clean_dir(self.candidate_path)
            self.pool.apply_async(I.read_stream, (self.stream_path / str(cfg.index), cfg, self.pipes[i],))
            self.pool.apply_async(I.detect,
                                  (self.stream_path / str(cfg.index), self.candidate_path / str(cfg.index),
                                   self.pipes[i], cfg,))
        self.pool.close()
        self.pool.join()


class DetectorController(object):
    def __init__(self, cfg: VideoConfig, stream_path: Path, region_path: Path, process_pool: Pool,
                 index_pool: Queue) -> None:
        super().__init__()
        self.cfg = cfg
        self.stream_path = stream_path
        self.region_path = region_path
        self.process_pool = process_pool
        self.cap = VideoCaptureThreading(stream_path, index_pool)
        self.cap.start()
        self.row = cfg.routine['row']
        self.col = cfg.routine['col']

        self.send_pipes = [Manager().Queue() for i in range(self.row * self.col)]
        self.receive_pipes = [Manager().Queue() for i in range(self.row * self.col)]

        frame = self.cap.read()
        frame = self.preprocess(frame, cfg)
        self.row_step = int(frame.shape[0] / self.row)
        self.col_step = int(frame.shape[1] / self.col)

        logger.info('Init total [{}] detectors....'.format(self.row * self.col))
        self.detectors = []
        for i in range(self.row):
            for j in range(self.col):
                self.detectors.append(Detector(self.row_step, self.col_step, i, j, cfg, self.send_pipes[i * j],
                                               self.receive_pipes[i * j],
                                               self.region_path / self.cfg.index / str(i) + '-' + str(j)))
        logger.debug('Detectors init done....')

    def preprocess(self, frame):
        if self.cfg.resize['scale'] != -1:
            frame = cv2.resize(frame, (0, 0), fx=self.cfg.resize['scale'], fy=self.cfg.resize['scale'])
        elif self.cfg.resize['width'] != -1:
            frame = imutils.resize(frame, self.cfg.resize['width'])
        elif self.cfg.resize['height '] != -1:
            frame = imutils.resize(frame, self.cfg.resize['height'])
        frame = crop_by_roi(frame, self.cfg.roi)
        # frame = imutils.resize(frame, width=1000)
        # frame = frame[340:, :, :]
        # frame = frame[170:, :, :]
        original_frame = frame.copy()
        frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        return frame, original_frame

    def control(self):
        logger.debug('Running detectors.......')
        for d in self.detectors:
            self.process_pool.apply_async(d.detect, ())
        while True:
            ret, frame = self.cap.read()
            frame = self.preprocess(frame)
            self.dispatch(frame)
            logger.info('Collecting sub-frames from detectors....')
            sub_frames = self.collect()
            logger.info('Done collected from detectors.....')
            logger.info('Constructing sub-frames into a original frame....')
            frame = self.construct(sub_frames)
            logger.info('Done constructing of sub-frames into a original frame....')
            cv2.imshow('Construected Frame', frame)
            cv2.waitKey(0)

    def dispatch(self, frame):
        for sp in self.send_pipes:
            sp.put(frame)

    def collect(self):
        sub_frames = []
        for rp in self.receive_pipes:
            sub_frames.append(rp.get())
        return sub_frames

    def construct(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.row, self.col, self.row_step, self.col_step, 3))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3, 4))
        sub_frames = np.reshape(sub_frames, (self.row * self.row_step, self.col * self.col_step, 3))
        return sub_frames


class VideoCaptureThreading:
    def __init__(self, video_path: Path, index_pool: Queue, width=640, height=480):
        logger.debug('Loading video stream from video index pool....')
        self.posix = self.video_path / index_pool.get()
        self.src = str(self.posix)
        if not os.path.exists(self.src):
            logger.debug('Video path not exist: [{}]'.format(self.src))
        self.cap = cv2.VideoCapture(self.src)
        logger.debug('Loading done ....')
        self.video_path = video_path
        self.index_pool = index_pool
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                logger.info('Read frame failed from [{}].'.format(self.src))
                logger.debug('Read next frame from video index pool..........')
                self.cap.release()
                self.posix.unlink()
                self.src = str(self.video_path / self.index_pool.get())
                if not os.path.exists(self.src):
                    logger.debug('Video path not exist: [{}]'.format(self.src))
                    continue
                self.cap = cv2.VideoCapture(self.src)
                logger.debug('Loading done from: [{}]'.format(self.src))
                continue
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            # frame = self.frame.copy()
            # grabbed = self.grabbed
            return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.cap.release()
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

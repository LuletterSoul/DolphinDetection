#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test.py
@time: 2019/11/15 14:49
@version 1.0
@desc:
"""

import time
from multiprocessing import Manager, Pool

import interface as I
from config import *
from detection.manager import DetectionMonitor
from utils import *


def test_video_config():
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    cfg = VideoConfig.from_json(cfg[0])
    print(cfg.resize)

def test_load_label_json():
    cfg = I.load_label_config(LABEL_SAVE_PATH / 'samples.json')
    print(cfg[0].center)


def test_load_video_json():
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    print(cfg[0].url)


def test_adaptive_thresh():
    frames = test_read_frame()
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    for f in frames:
        binary = I.thresh(f, cfg[0])
        cv2.imshow('binary', binary)
        cv2.waitKey(0)


def test_read_steam():
    # clean all exist streams
    q = Manager().Queue()
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    I.read_stream(STREAM_SAVE_DIR / str(cfg.index), cfg[0], q)
    p = Pool()
    p.apply_async(I.read_stream, (STREAM_SAVE_DIR / str(cfg.index), cfg, q,))
    # I.read_stream(STREAM_SAVE_DIR, cfg['videos'][0], q)


def test_detect():
    clean_dir(STREAM_SAVE_DIR)
    cfgs = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    # for i, cfg in enumerate(cfgs[:1]):
    p = Pool()
    qs = [Manager().Queue(), Manager().Queue()]
    for i, cfg in enumerate(cfgs):
        time.sleep(1)
        # q = Queue()
        p.apply_async(I.read_stream, (STREAM_SAVE_DIR / str(cfg.index), cfg, qs[i],))
        p.apply_async(I.detect, (STREAM_SAVE_DIR / str(cfg.index), CANDIDATE_SAVE_DIR, qs[i], cfg,))
        # p.apply_async(init_detect, (STREAM_SAVE_DIR / str(cfg.index), cfg,))
    p.close()
    p.join()
    print('Init Done')


def test_detect_monitor():
    monitor = DetectionMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR, CANDIDATE_SAVE_DIR)
    monitor.monitor()


def test_read_frame():
    data = STREAM_SAVE_DIR / str(0) / '139.ts'
    samples = I.read_frame(data, FRAME_SAVE_DIR / str(0))
    return samples


if __name__ == '__main__':
    # test_load_video_json()
    test_load_label_json()
    # test_video_config()
    # test_read_steam()
    # test_read_frame()
    # test_detect()
    # test_detect_monitor()
    # test_adaptive_thresh()

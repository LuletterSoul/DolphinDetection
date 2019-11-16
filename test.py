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

import json
from config import *
import interface as I
from utils import *
from multiprocessing import Manager, Pool, Queue
from concurrent.futures import ThreadPoolExecutor

def test_video_config():
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    cfg = VideoConfig.from_json(cfg[0])
    print(cfg.resize)


def test_load_video_json():
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    print(cfg[0].url)


def test_read_steam():
    # clean all exist streams
    q = Manager().Queue()
    cfg = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    I.read_stream(STREAM_SAVE_DIR / str(cfg.index), cfg[0], q)

    # I.read_stream(STREAM_SAVE_DIR, cfg['videos'][0], q)


def test_detect():
    clean_dir(STREAM_SAVE_DIR)
    cfgs = I.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
    # for i, cfg in enumerate(cfgs[:1]):
    p = Pool()
    for i, cfg in enumerate(cfgs):
        q = Manager().Queue()
        # q = Queue()
        p.apply_async(I.read_stream, (STREAM_SAVE_DIR / str(cfg.index), cfg, q,))
        p.apply_async(I.detect, (STREAM_SAVE_DIR / str(cfg.index), CANDIDATE_SAVE_DIR, q, cfg,))
        # p.apply_async(init_detect, (STREAM_SAVE_DIR / str(cfg.index), cfg,))
    p.close()
    p.join()
    print('Init Done')


def init_detect(stream_path: Path, cfg: VideoConfig):
    # q = Queue()
    print('Init')
    try:
        with ThreadPoolExecutor() as p:
            q = Manager().Queue()
            p.apply_async(I.read_stream, (stream_path, cfg, q,))
            p.apply_async(I.detect, (stream_path, cfg, q, cfg))
            p.close()
            p.join()
    except Exception as e:
        print(e)


def test_read_img():
    data = str(STREAM_SAVE_DIR / '020.ts')
    I.read_frame(data, str(FRAME_SAVE_DIR))


if __name__ == '__main__':
    # test_load_video_json()
    # test_video_config()
    # test_read_steam()
    # test_read_img()
    test_detect()

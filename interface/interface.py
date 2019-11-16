#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: main.py
@time: 2019/11/15 14:53
@version 1.0
@desc:
"""

import traceback

import detection
import stream
from config import *
from utils.log import logger


def detect(video_path, candidate_save_path, mq, cfg):
    try:
        detection.detect(video_path, candidate_save_path, mq, cfg)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)


def thresh(frame_path):
    detection.adaptive_thresh(frame_path)


def read_stream(sream_save_path, vcfg, mq):
    stream.read(sream_save_path, vcfg, mq)


def read_frame(input_path, output_path):
    return stream.process_video(input_path, output_path)


def load_video_config(cfg_path):
    cfg_objs = json.load(open(cfg_path))['videos']
    cfgs = [VideoConfig.from_json(c) for c in cfg_objs]
    return cfgs

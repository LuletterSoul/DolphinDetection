#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: config.py
@time: 2019/11/15 10:43
@version 1.0
@desc:
"""
import json
import logging
import os
from pathlib import Path

LOG_LEVER = logging.INFO

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_SAVE_DIR = PROJECT_DIR / 'data/candidates'
CANDIDATE_SAVE_DIR.mkdir(exist_ok=True, parents=True)

LOG_DIR = PROJECT_DIR / 'log'
LOG_DIR.mkdir(exist_ok=True, parents=True)

STREAM_SAVE_DIR = PROJECT_DIR / 'data/videos'
STREAM_SAVE_DIR \
    .mkdir(exist_ok=True, parents=True)

SAMPLE_SAVE_DIR = PROJECT_DIR / 'data/samples'
SAMPLE_SAVE_DIR \
    .mkdir(exist_ok=True, parents=True)

OFFLINE_STREAM_SAVE_DIR = PROJECT_DIR / 'data/offline'
OFFLINE_STREAM_SAVE_DIR \
    .mkdir(exist_ok=True, parents=True)

FRAME_SAVE_DIR = PROJECT_DIR / 'data/frames'
FRAME_SAVE_DIR \
    .mkdir(exist_ok=True, parents=True)

VIDEO_CONFIG_DIR = PROJECT_DIR / 'vcfg'
VIDEO_CONFIG_DIR.mkdir(exist_ok=True, parents=True)

LABEL_IMAGE_PATH = PROJECT_DIR / 'data/labels/image'
LABEL_TARGET_PATH = PROJECT_DIR / 'data/labels/target'
LABEL_IMAGE_PATH.mkdir(exist_ok=True, parents=True)
LABEL_TARGET_PATH.mkdir(exist_ok=True, parents=True)
LABEL_SAVE_PATH = PROJECT_DIR / 'data/labels'

from enum import Enum


class MonitorType(Enum):
    PROCESS_BASED = 1,
    THREAD_BASED = 2,
    PROCESS_THREAD_BASED = 3


# select monitor type, process-based means the system will create a process for each component,such as detector,
# stream receiver, frame dispatcher and frame collector...
# Required much resources because system divide resources into process units,
# which are limited by CPU cores
# MONITOR = MonitorType.PROCESS_BASED


MONITOR = MonitorType.PROCESS_BASED


class VideoConfig:
    def __init__(self, index, name, ip, port, suffix, headers, m3u8_url, url, roi, resize, show_window,
                 window_position, routine, sample_rate, draw_boundary, enable, filtered_ratio, max_streams_cache,
                 online, sample_internal, save_box):
        self.index = index
        self.name = name
        self.ip = ip
        self.port = port
        self.suffix = suffix
        self.headers = headers
        self.m3u8_url = m3u8_url
        self.url = url
        self.roi = roi
        self.resize = resize
        self.show_window = show_window
        self.window_position = window_position
        self.routine = routine
        self.sample_rate = sample_rate
        self.draw_boundary = draw_boundary
        self.enable = enable
        self.filtered_ratio = filtered_ratio
        self.max_streams_cache = max_streams_cache
        self.online = online
        self.sample_internal = sample_internal
        self.save_box = save_box

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_dict):
        # json_dict = json.loads(json_str)
        return cls(**json_dict)

# example usage
# User("tbrown", "Tom Brown").to_json()
# User.from_json(User("tbrown", "Tom Brown").to_json()).to_json()

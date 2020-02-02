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
import psutil
from pathlib import Path
import time

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
BINARY_SAVE_PATH = PROJECT_DIR / 'data/labels/binarys'

INFORM_SAVE_PATH = PROJECT_DIR / 'vcfg'

from enum import Enum


class MonitorType(Enum):
    PROCESS_BASED = 1,
    THREAD_BASED = 2,
    PROCESS_THREAD_BASED = 3
    RAY_BASED = 4
    TASK_BASED = 5


class Env(Enum):
    DEV = 1,
    TEST = 2,
    DEPLOYMENT = 3,


# select monitor type, process-based means the system will create a process for each component,such as detector,
# stream receiver, frame dispatcher and frame collector...
# Required much resources because system divide resources into process units,
# which are limited by CPU cores
# MONITOR = MonitorType.PROCESS_BASED

ENV = Env.DEV

MONITOR = MonitorType.TASK_BASED

# enable_options = {
#     0: False,
#     1: False,
#     2: False,
#     3: False,
#     4: False,
#     5: True,
#     6: True,
#     7: True,
#     8: True,
#     9: False,
#     10: False,
#     11: False,
#     12: False,
#     13: False,
#     14: False,
#     15: False,
#     16: False,
#     17: False,
#     # 17: True,
# }

enable_options = {
    0: True,
    1: True,
    2: True,
    3: True,
    4: True,
    5: False,
    6: False,
    7: False,
    8: False,
}


class VideoConfig:
    def __init__(self, index, name, shape, ip, port, suffix, headers, m3u8_url, url, roi, resize, show_window,
                 window_position, routine, sample_rate, draw_boundary, enable, filtered_ratio, max_streams_cache,
                 online, sample_internal, save_box, show_box, rtsp, rtsp_saved_per_frame, future_frames,
                 alg):
        self.index = index
        self.name = name
        self.shape = shape
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
        self.show_box = show_box
        self.rtsp = rtsp
        self.rtsp_saved_per_frame = rtsp_saved_per_frame
        self.future_frames = future_frames
        self.alg = alg

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_dict):
        # json_dict = json.loads(json_str)
        return cls(**json_dict)


class LabelConfig:
    """
    Image label class
    """

    def __init__(self, start, end, center):
        self.start = start
        self.end = end
        self.center = center

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_dict):
        return cls(**json_dict)


# example usage
# User("tbrown", "Tom Brown").to_json()
# User.from_json(User("tbrown", "Tom Brown").to_json()).to_json()
class SystemStatus(Enum):
    RUNNING = 1,
    SHUT_DOWN = 2,
    RESUME = 3


_timer = getattr(time, 'monotonic', time.time)
num_cpus = psutil.cpu_count() or 1


def timer():
    return _timer() * num_cpus


pid_cpuinfo = {}


def cpu_usage():
    id = os.getpid()
    p = psutil.Process(id)
    # while True:
    #     print(p.cpu_percent())

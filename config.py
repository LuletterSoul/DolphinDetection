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
import os
from pathlib import Path
import logging

LOG_LEVER = logging.INFO

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_SAVE_DIR = PROJECT_DIR / 'data/candidates'
CANDIDATE_SAVE_DIR.mkdir(exist_ok=True, parents=True)

LOG_DIR = PROJECT_DIR / 'log'
LOG_DIR.mkdir(exist_ok=True, parents=True)

STREAM_SAVE_DIR = PROJECT_DIR / 'data/videos'
STREAM_SAVE_DIR \
    .mkdir(exist_ok=True, parents=True)

FRAME_SAVE_DIR = PROJECT_DIR / 'data/frames'
FRAME_SAVE_DIR \
    .mkdir(exist_ok=True, parents=True)

VIDEO_CONFIG_DIR = PROJECT_DIR / 'vcfg'
VIDEO_CONFIG_DIR.mkdir(exist_ok=True, parents=True)

import json


class VideoConfig:
    def __init__(self, index, name, ip, port, suffix, headers, m3u8_url, url, roi, resize, show_window=True):
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

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_dict):
        # json_dict = json.loads(json_str)
        return cls(**json_dict)

# example usage
# User("tbrown", "Tom Brown").to_json()
# User.from_json(User("tbrown", "Tom Brown").to_json()).to_json()

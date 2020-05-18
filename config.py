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
from enum import Enum
from pathlib import Path


class Environment(object):
    PROD = 'prod'
    TEST = 'test'
    DEV = 'dev'


class LogLevel(object):
    INFO = 'INFO'
    DEBUG = 'DEBUG',
    ERROR = 'ERROR'


class ModelType(object):
    SSD = 'ssd'
    CLASSIFY = 'classify'
    CASCADE = 'cascade'
    FORWARD = 'fwd'


LOG_LEVER = logging.DEBUG

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = PROJECT_DIR / 'log'
LOG_DIR.mkdir(exist_ok=True, parents=True)


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


ENV = Env.DEV


class Config(object):
    """
    Base configuration class, built-in json to object method
    """

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_dict):
        # json_dict = json.loads(json_str)
        return cls(**json_dict)

    @classmethod
    def from_yaml(cls, yaml_dict):
        return cls(**yaml_dict)


class ServerConfig(Config):
    """
    Server configuration definitions
    """

    def __init__(self, env, log_level, http_ip, http_port, wc_ip, wc_port, send_msg, run_direct, cron, root,
                 classify_model_path,
                 detect_model_path,
                 cascade_model_path,
                 cascade_model_cfg,
                 track_cfg_path,
                 track_model_path,
                 detect_mode,
                 stream_save_path,
                 sample_save_dir,
                 frame_save_dir,
                 candidate_save_dir, offline_stream_save_dir) -> None:
        self.env = env
        self.log_level = log_level
        self.http_ip = http_ip
        self.http_port = http_port
        self.wc_ip = wc_ip
        self.send_msg = send_msg
        self.run_direct = run_direct
        self.cron = cron
        self.wc_port = wc_port
        self.detect_mode = detect_mode
        self.track_model_path = track_model_path
        self.track_cfg_path = track_cfg_path
        self.classify_model_path = classify_model_path
        self.detect_model_path = Path(os.path.join(PROJECT_DIR, detect_model_path))
        self.cascade_model_path = cascade_model_path
        self.cascade_model_cfg = cascade_model_cfg
        self.track_model_path = track_model_path
        self.stream_save_path = stream_save_path
        self.sample_save_dir = sample_save_dir
        self.frame_save_dir = frame_save_dir
        self.candidate_save_dir = candidate_save_dir
        self.offline_stream_save_dir = offline_stream_save_dir
        self.root = root
        self.convert_to_poxis()

    def set_root(self, root):
        self.root = root
        self.convert_to_poxis()

    def set_candidate_save_dir(self, cdp):
        self.candidate_save_dir = cdp
        self.convert_to_poxis()

    def convert_to_poxis(self, ):
        if self.root == '':
            self.stream_save_path = Path(os.path.join(PROJECT_DIR, self.stream_save_path))
            self.sample_save_dir = Path(os.path.join(PROJECT_DIR, self.sample_save_dir))
            self.frame_save_dir = Path(os.path.join(PROJECT_DIR, self.frame_save_dir))
            self.candidate_save_dir = Path(os.path.join(PROJECT_DIR, self.candidate_save_dir))
            self.classify_model_path = Path(os.path.join(PROJECT_DIR, self.classify_model_path))
            self.offline_stream_save_dir = Path(os.path.join(PROJECT_DIR, self.offline_stream_save_dir))
        else:
            self.stream_save_path = Path(os.path.join(self.root, self.stream_save_path))
            self.sample_save_dir = Path(os.path.join(self.root, self.sample_save_dir))
            self.frame_save_dir = Path(os.path.join(self.root, self.frame_save_dir))
            self.candidate_save_dir = Path(os.path.join(self.root, self.candidate_save_dir))
            self.classify_model_path = Path(os.path.join(self.root, self.classify_model_path))
            self.offline_stream_save_dir = Path(os.path.join(self.root, self.offline_stream_save_dir))


class VideoConfig(Config):
    """
    Video configuration object
    """

    def __init__(self, index, name, camera_id, channel, shape, ip, dip, port, suffix, headers, m3u8_url, url, roi,
                 resize, show_window,
                 push_stream,
                 window_position, routine, sample_rate, draw_boundary, enable, filtered_ratio, max_streams_cache,
                 online, cap_loop, sample_internal, detect_internal, search_window_size, use_sm, cache_size,
                 similarity_thresh,
                 pre_cache,
                 render,
                 post_filter,
                 forward_filter,
                 limit_freq,
                 freq_thresh,
                 save_box, show_box, cv_only, ssd_divide_four,
                 rtsp, push_to, write_timestamp,
                 enable_sample_frame,
                 rtsp_saved_per_frame,
                 future_frames, bbox,
                 alg):
        self.index = index
        self.camera_id = camera_id
        self.channel = channel
        self.name = name
        self.shape = shape
        self.ip = ip
        self.dip = dip
        self.port = port
        self.suffix = suffix
        self.headers = headers
        self.m3u8_url = m3u8_url
        self.url = url
        self.roi = roi
        self.resize = resize
        self.show_window = show_window
        self.push_stream = push_stream
        self.use_sm = use_sm
        self.cache_size = cache_size
        self.window_position = window_position
        self.routine = routine
        self.sample_rate = sample_rate
        self.draw_boundary = draw_boundary
        self.enable = enable
        self.filtered_ratio = filtered_ratio
        self.max_streams_cache = max_streams_cache
        self.online = online
        self.cap_loop = cap_loop
        self.sample_internal = sample_internal
        self.save_box = save_box
        self.show_box = show_box
        self.cv_only = cv_only
        self.ssd_divide_four = ssd_divide_four
        self.rtsp = rtsp
        self.push_to = push_to
        self.write_timestamp = write_timestamp
        self.enable_sample_frame = enable_sample_frame
        self.rtsp_saved_per_frame = rtsp_saved_per_frame
        self.future_frames = future_frames
        self.detect_internal = detect_internal
        self.search_window_size = search_window_size
        self.similarity_thresh = similarity_thresh
        self.pre_cache = pre_cache
        self.render = render
        self.post_filter = post_filter
        self.forward_filter = forward_filter
        self.limit_freq = limit_freq
        self.freq_thresh = freq_thresh
        self.bbox = bbox
        self.alg = alg


class LabelConfig:
    """
    Image label class
    """

    def __init__(self, start, end, center):
        self.start = start
        self.end = end
        self.center = center


class SystemStatus(Enum):
    RUNNING = 1,
    SHUT_DOWN = 2,
    RESUME = 3


class ObjectClass(Enum):
    DOLPHIN = 0,
    OTHER = 1,

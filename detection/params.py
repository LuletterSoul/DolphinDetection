#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: params.py
@time: 2019/12/13 16:45
@version 1.0
@desc:
"""
from pathlib import Path

from config import VideoConfig
from utils import logger


# class CacheClearConfiguration(object):
#
#     def __init__(self, future_frames, len_thresh) -> None:
#         super().__init__()
#         self.future_frames = future_frames
#         self.len_thresh = len_thresh


class ConstructParams(object):

    def __init__(self, result_queue, original_frame_cache, render_frame_cache, render_rect_cache, stream_render,
                 len_thresh, cfg: VideoConfig) -> None:
        super().__init__()
        self.result_queue = result_queue
        self.original_frame_cache = original_frame_cache
        self.render_frame_cache = render_frame_cache
        self.render_rect_cache = render_rect_cache
        self.stream_render = stream_render
        self.len_thresh = len_thresh
        self.cfg = cfg


class DispatchBlock(object):

    def __init__(self, frame, index, original_shape) -> None:
        super().__init__()
        self.frame = frame
        self.index = index
        self.shape = original_shape


class ConstructResult(object):

    def __init__(self, frame, binary, thresh, last_detection_time=None) -> None:
        super().__init__()
        self.frame = frame
        self.binary = binary
        self.thresh = thresh
        self.last_detection_time = last_detection_time


class ConstructParams(object):

    def __init__(self, result_queue, original_frame_cache, render_frame_cache, render_rect_cache, stream_render,
                 len_thresh, cfg: VideoConfig) -> None:
        super().__init__()
        self.result_queue = result_queue
        self.original_frame_cache = original_frame_cache
        self.render_frame_cache = render_frame_cache
        self.render_rect_cache = render_rect_cache
        self.stream_render = stream_render
        # self.cache_clear_cfg = cache_clear_cfg
        self.len_thresh = len_thresh
        self.cfg = cfg


class BlockInfo(object):

    def __init__(self, row, col, row_step, col_step) -> None:
        super().__init__()
        self.row = row
        self.col = col
        self.row_step = row_step
        self.col_step = col_step


class ReconstructResult(object):

    def __init__(self, rcf, rcb, rct) -> None:
        super().__init__()
        self.reconstruct_frame = rcf
        self.reconstruct_binary = rcb
        self.reconstruct_rct = rct


class DetectorParams(object):

    def __init__(self, col_step, row_step, row_index, col_index, cfg: VideoConfig,
                 region_save_path: Path) -> None:
        super().__init__()
        # self.video_path = video_path
        # self.region_save_path = region_save_path
        self.cfg = cfg
        self.row_step = col_step
        self.col_step = row_step
        self.col_index = row_index
        self.row_index = col_index
        self.start = [self.row_index * row_step, self.col_index * col_step]
        self.end = [(self.row_index + 1) * row_step, (self.col_index + 1) * col_step]
        self.region_save_path = region_save_path
        self.region_save_path.mkdir(exist_ok=True, parents=True)
        logger.debug(
            'Detector [{},{}]: region save to: [{}]'.format(self.col_index, self.col_index, str(self.region_save_path)))

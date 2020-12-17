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

    def __init__(self, frame, binary, thresh, last_detection_time=None, detect_flag=False, results=None,
                 frame_index=1) -> None:
        super().__init__()
        self.frame = frame
        self.binary = binary
        self.thresh = thresh
        self.last_detection_time = last_detection_time
        self.detect_flag = detect_flag
        self.results = results
        self.frame_index = frame_index


class ConstructParams(object):

    def __init__(self, result_queue, original_frame_cache, render_rect_cache, stream_render,
                 len_thresh, cfg: VideoConfig) -> None:
        super().__init__()
        self.result_queue = result_queue
        self.original_frame_cache = original_frame_cache
        self.render_rect_cache = render_rect_cache
        self.stream_render = stream_render
        # self.cache_clear_cfg = cache_clear_cfg
        self.len_thresh = len_thresh
        self.cfg = cfg


class BlockInfo(object):

    def __init__(self, y_num, x_num, y_step, x_step) -> None:
        super().__init__()
        self.y_num = y_num
        self.x_num = x_num
        self.y_step = y_step
        self.x_step = x_step


class ReconstructResult(object):

    def __init__(self, rcf, rcb, rct) -> None:
        super().__init__()
        self.reconstruct_frame = rcf
        self.reconstruct_binary = rcb
        self.reconstruct_rct = rct


class FrameStorage(object):
    def __init__(self, frame_index, rects, save_frame=True, save_square_crop=True, save_original_crop=False):
        self.frame_index = frame_index
        self.rects = rects
        self.save_frame = save_frame
        self.save_square_crop = save_square_crop
        self.save_original_crop = save_original_crop


class DetectorParams(object):

    def __init__(self, x_step, y_step, x_index, y_index, cfg: VideoConfig, region_save_path: Path) -> None:
        super().__init__()
        # self.video_path = video_path
        # self.region_save_path = region_save_path
        self.cfg = cfg
        self.y_step = y_step
        self.x_step = x_step
        self.y_index = y_index
        self.x_index = x_index
        self.start = [self.x_index * x_step, self.y_index * y_step]
        self.end = [(self.x_index + 1) * x_step, (self.y_index + 1) * y_step]
        self.region_save_path = region_save_path
        self.region_save_path.mkdir(exist_ok=True, parents=True)
        logger.debug(
            'Detector [{},{}]: region save to: [{}]'.format(self.y_index, self.y_index, str(self.region_save_path)))

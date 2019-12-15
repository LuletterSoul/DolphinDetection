#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: common.py
@time: 2019/12/13 16:05
@version 1.0
@desc:
"""
import threading
import traceback
from config import VideoConfig

import cv2
import imutils
import time

from .crop import crop_by_roi


def preprocess(frame, cfg: VideoConfig):
    original_frame = frame.copy()
    frame = crop_by_roi(frame, cfg.roi)
    if cfg.resize['scale'] != -1:
        frame = cv2.resize(frame, (0, 0), fx=cfg.resize['scale'], fy=cfg.resize['scale'])
    elif cfg.resize['width'] != -1:
        frame = imutils.resize(frame, cfg.resize['width'])
    elif cfg.resize['height '] != -1:
        frame = imutils.resize(frame, cfg.resize['height'])
    # frame = imutils.resize(frame, width=1000)
    # frame = frame[340:, :, :]
    # frame = frame[170:, :, :]
    frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
    return frame, original_frame


def back(rects, start, shape, original_shape, cfg: VideoConfig):
    if not len(rects):
        return rects
    b_rects = []
    ratio = 1
    if cfg.resize['scale'] != -1:
        ratio = cfg.resize['scale']
    if cfg.resize['width'] != -1:
        ratio = original_shape[1] / cfg.routine['col'] / shape[1]
    if cfg.resize['height'] != -1:
        ratio = original_shape[0] / cfg.routine['row'] / shape[0]
    for r in rects:
        x = int((r[0] + start[0] + cfg.roi['x']) * ratio)
        y = int((r[1] + start[1] + cfg.roi['y']) * ratio)
        w = int(r[2] * ratio)
        h = int(r[3] * ratio)
        b_rects.append((x, y, w, h))
    return b_rects


def draw_boundary(frame, info):
    shape = frame.shape
    for i in range(info.x_num - 1):
        start = (info.x_step * (i + 1), 0)
        end = (info.x_step * (i + 1), shape[0] - 1)
        cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    for j in range(info.y_num - 1):
        start = (0, info.y_step * (j + 1))
        end = (shape[1] - 1, info.y_step * (j + 1))
        cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    return frame


def clear_cache(cache, num=2):
    half_len = len(cache) // num
    cnt = 0
    keys = cache.keys()
    for k in keys:
        if k in cache:
            try:
                cache.pop(k)
                cnt += 1
                if cnt == half_len:
                    break
            except Exception as e:
                pass
                # traceback.print_exc()


def clear_cache_by_len(cache, len_cache):
    if len(cache) > len_cache:
        thread = threading.Thread(
            target=clear_cache,
            args=(cache,))
        thread.start()


def generate_time_stamp(fmt='%m-%d-%H-%M'):
    return time.strftime(fmt, time.localtime(time.time()))

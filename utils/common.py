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

import cv2
import imutils

from .crop import crop_by_roi


def preprocess(frame, cfg):
    if cfg.resize['scale'] != -1:
        frame = cv2.resize(frame, (0, 0), fx=cfg.resize['scale'], fy=cfg.resize['scale'])
    elif cfg.resize['width'] != -1:
        frame = imutils.resize(frame, cfg.resize['width'])
    elif cfg.resize['height '] != -1:
        frame = imutils.resize(frame, cfg.resize['height'])
    frame = crop_by_roi(frame, cfg.roi)
    # frame = imutils.resize(frame, width=1000)
    # frame = frame[340:, :, :]
    # frame = frame[170:, :, :]
    original_frame = frame.copy()
    frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
    return frame, original_frame


def draw_boundary(frame, info):
    shape = frame.shape
    for i in range(info.col - 1):
        start = (0, info.col_step * (i + 1))
        end = (shape[1] - 1, info.col_step * (i + 1))
        cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    for j in range(info.row - 1):
        start = (info.row_step * (j + 1), 0)
        end = (info.row_step * (j + 1), shape[0] - 1)
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
                traceback.print_exc()


def clear_cache_by_len(cache, len_cache):
    if len(cache) > len_cache:
        thread = threading.Thread(
            target=clear_cache,
            args=(cache,))
        thread.start()

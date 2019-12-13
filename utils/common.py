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
import traceback
import time
import threading
from .log import logger
import cv2

# from detection import BlockInfo


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

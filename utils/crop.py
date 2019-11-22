#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: utils.py
@time: 2019/11/8 11:31
@version 1.0
@desc:
"""
import shutil
import cv2
import json
from collections import namedtuple


def clean_dir(path):
    shutil.rmtree(str(path))
    path \
        .mkdir(exist_ok=True, parents=True)


def in_range_equal(val, up_bound, low_bound=0):
    return up_bound >= val >= low_bound


def in_range(val, up_bound, low_bound=0):
    return up_bound > val >= low_bound


def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())


def json2obj(data): return json.loads(data, object_hook=_json_object_hook)


def crop_by_roi(img, roi):
    shape = img.shape
    x = roi['x']
    y = roi['y']
    width = roi['width']
    height = roi['height']
    if width != -1 and height != -1:
        is_in_range = in_range(y, shape[0]) and in_range(x, shape[1]) \
                      and in_range(y + height, shape[0]) \
                      and in_range(x + width, shape[1])
        if is_in_range:
            return img[y:y + height, x:x + width]
    elif width != -1:
        is_in_range = in_range(y, shape[0]) and in_range(x, shape[1]) and in_range(x + width, shape[1])
        if is_in_range:
            return img[y:, x:x + width]
    elif height != -1:
        is_in_range = in_range(y, shape[0]) and in_range(x, shape[1]) and in_range(y + height, shape[0])
        if is_in_range:
            return img[y:y + height, x:]
    else:
        is_in_range = in_range(y, shape[0]) and in_range(x, shape[1])
        if is_in_range:
            return img[y:, x:]
    if not is_in_range:
        raise Exception('Crop range out of bound.')


def crop_by_se(img, start, end):
    if not (start[0] <= end[0] and start[1] <= end[1]):
        raise Exception('Start and end is invalid.')
    shape = img.shape
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]
    is_in_range = in_range(y1, shape[0]) and in_range(x1, shape[1]) \
                  and in_range_equal(y2, shape[0]) \
                  and in_range_equal(x2, shape[1])

    if is_in_range:
        return img[y1:y2, x1:x2]
    else:
        raise Exception('Crop range out of bound.')

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
import json
import shutil
from collections import namedtuple
from pathlib import Path
from config import VideoConfig
import cv2


def clean_dir(path: Path):
    if not path.exists():
        return
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


def bbox_points(cfg: VideoConfig, rect, shape, delta_x=0, delta_y=0):
    center_x, center_y = round((rect[0] + rect[2]) / 2), round((rect[1] + rect[3]) / 2)
    start_x, start_y = round(center_x - cfg.bbox['w'] / 2 - round(delta_x)), round(
        center_y - cfg.bbox['h'] / 2 - round(delta_y))
    end_x = center_x + cfg.bbox['w'] / 2 + round(delta_x)
    end_y = center_y + cfg.bbox['h'] / 2 + round(delta_y)
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if end_x >= shape[1]:
        end_x = shape[1] - 1
    if end_y >= shape[0]:
        end_y = shape[0] - 1
    p1 = (int(start_x), int(start_y))
    p2 = (int(end_x), int(end_y))
    return p1, p2


def bbox_points_float(cfg: VideoConfig, rect, shape, delta_x=0, delta_y=0):
    center_x, center_y = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    start_x, start_y = center_x - cfg.bbox['w'] / 2 - delta_x, center_y - cfg.bbox['h'] / 2 - delta_y
    end_x = start_x + cfg.bbox['w'] + delta_x
    end_y = start_y + cfg.bbox['h'] + delta_y
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if end_x > shape[1]:
        end_x = shape[1]
    if end_y > shape[0]:
        end_y = shape[0]
    p1 = (int(start_x), int(start_y))
    p2 = (int(end_x), int(end_y))
    return p1, p2


def _bbox_points(w, h, rect, shape, delta_x=0, delta_y=0):
    center_x, center_y = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    # center_x, center_y = round(rect[0] + rect[2] / 2), round(rect[1] + rect[3] / 2)
    start_x, start_y = round(center_x - w / 2 - delta_x), round(
        center_y - h / 2 - delta_y)
    end_x = start_x + w + delta_x
    end_y = start_y + h + delta_y
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if end_x > shape[1]:
        end_x = shape[1]
    if end_y > shape[0]:
        end_y = shape[0]
    p1 = (int(start_x), int(start_y))
    p2 = (int(end_x), int(end_y))
    # p1 = (start_x, start_y)
    # p2 = (end_x, end_y)
    return p1, p2


def crop_by_rect(cfg: VideoConfig, rect, frame):
    shape = frame.shape
    p1, p2 = bbox_points(cfg, rect, shape)
    start_x, start_y = p1
    end_x, end_y = p2
    return cv2.resize(frame[start_y:end_y, start_x:end_x], (cfg.bbox['w'], cfg.bbox['h']))


def crop_by_rect_wh(w, h, rect, frame):
    shape = frame.shape
    p1, p2 = _bbox_points(w, h, rect, shape)
    start_x, start_y = p1
    end_x, end_y = p2
    return cv2.resize(frame[start_y:end_y, start_x:end_x], (w, h))

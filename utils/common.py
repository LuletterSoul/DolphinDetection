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
# import imutils
import time
from .crop import crop_by_roi
from skimage.measure import compare_ssim
import numpy as np


def preprocess(frame, cfg: VideoConfig):
    original_frame = frame.copy()
    frame = crop_by_roi(frame, cfg.roi)
    if cfg.resize['scale'] != -1:
        frame = cv2.resize(frame, (0, 0), fx=cfg.resize['scale'], fy=cfg.resize['scale'])
    elif cfg.resize['width'] != -1:
        frame = resize(frame, cfg.resize['width'])
    elif cfg.resize['height '] != -1:
        frame = resize(frame, cfg.resize['height'])
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


def cvt_rect(rects):
    new_rects = []
    for rect in rects:
        new_rects.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
    return new_rects


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


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def clear_cache_by_len(cache, len_cache):
    if len(cache) > len_cache:
        thread = threading.Thread(
            target=clear_cache,
            args=(cache,))
        thread.start()


def generate_time_stamp(fmt='%m%d%H%M'):
    return time.strftime(fmt, time.localtime(time.time()))


def sec2time(sec, n_msec=1):
    ''' Convert seconds to 'D days, HH:MM:SS.FFF' '''
    if hasattr(sec, '__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)


def createHistFeature(grid, small_grid=3):
    """
    # 生成颜色直方图特征
    :param grid: 数据
    :param small_grid: 细分的网格数
    :return: 数据的特征
    """
    hist_mask = np.array([])
    colnum = int(grid.shape[1] / small_grid)
    rownum = int(grid.shape[0] / small_grid)
    for i in range(small_grid):
        for j in range(small_grid):
            image = grid[i * colnum:(i + 1) * colnum, j * rownum:(j + 1) * rownum, :]
            hist_mask0 = cv2.calcHist([image], [0], None, [16], [0, 255])
            hist_mask1 = cv2.calcHist([image], [1], None, [16], [0, 255])
            hist_mask2 = cv2.calcHist([image], [2], None, [16], [0, 255])
            hist_mask_small = np.concatenate((hist_mask0, hist_mask1, hist_mask2), axis=0)
            if (len(hist_mask) == 0):
                hist_mask = hist_mask_small
            else:
                hist_mask = np.concatenate((hist_mask, hist_mask_small), axis=0)
    return hist_mask


def normalization(data):
    return data / np.linalg.norm(data)


def cal_rgb_similarity(img1, img2, alg='hist_cosine'):
    if alg == 'hist_cosine':
        return cal_hist_cosine_similarity(img1, img2)
    elif alg == 'ssim':
        return compare_ssim(img1, img2, multichannel=True)


def cal_std_similarity(seq):
    if not len(seq) or seq is None:
        return 0
    return normalization(seq).std()


def cal_hist_cosine_similarity(img1, img2):
    # img1_hist_feat = np.squeeze(createHistFeature(img1))
    # img2_hist_feat = np.squeeze(createHistFeature(img2))
    return cal_hist_cosine_distance(img1, img2)


def cal_hist_cosine_distance(img1, img2):
    img1_hist_feat = np.squeeze(normalization(createHistFeature(img1)))
    img2_hist_feat = np.squeeze(normalization(createHistFeature(img2)))
    # num = float(img1_hist_feat * img2_hist_feat.T)
    # denom = np.linalg.(img1_hist_feat) * np.linalg.norm(img2_hist_feat)
    # cos = num / denom
    # sim = 0.5 + 0.5 * cos
    return np.dot(img1_hist_feat, img2_hist_feat)


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

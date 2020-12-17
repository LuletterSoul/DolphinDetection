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

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from config import VideoConfig

import cv2
# import imutils
import time
from .crop import crop_by_roi
from skimage.measure import compare_ssim
import numpy as np
import imutils


def preprocess(frame, cfg: VideoConfig):
    """
    some preprocess operation such as denoising, image enhancement and crop by ROI
    :param frame:
    :param cfg: well-define frame detection range by ROI
    :return: processed frame, original frame
    """
    original_frame = frame.copy()
    frame = crop_by_roi(frame, cfg.roi)
    if cfg.resize['scale'] != -1:
        frame = cv2.resize(frame, (0, 0), fx=cfg.resize['scale'], fy=cfg.resize['scale'])
    elif cfg.resize['width'] != -1:
        frame = imutils.resize(frame, width=cfg.resize['width'])
    elif cfg.resize['height'] != -1:
        frame = imutils.resize(frame, height=cfg.resize['height'])
    # frame = imutils.resize(frame, width=1000)
    # frame = frame[340:, :, :]
    # frame = frame[170:, :, :]
    # frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
    return frame, original_frame


def back(rects, start, shape, original_shape, cfg: VideoConfig):
    """
    recover original bounding box size after detection-based down sample and divide blocks frame
    :param rects: detected bbox based cropped or resized frames
    :param start: start position of frame block
    :param shape: current shape
    :param original_shape: original shape
    :param cfg: video configuration, well-define bbox size should be
    :return:
    """
    if not len(rects):
        return rects
    b_rects = []
    ratio = 1
    if cfg.resize['scale'] != -1:
        ratio = cfg.resize['scale']
    elif cfg.resize['width'] != -1:
        ratio = original_shape[1] / cfg.routine['col'] / shape[1]
    elif cfg.resize['height'] != -1:
        ratio = original_shape[0] / cfg.routine['row'] / shape[0]
    for r in rects:
        x = int((r[0] + start[0]) * ratio + cfg.roi['x'])
        y = int((r[1] + start[1]) * ratio + cfg.roi['y'])
        w = int(r[2] * ratio)
        h = int(r[3] * ratio)
        b_rects.append((x, y, x + w, y + h))
    return b_rects


def cvt_rect(rects):
    """
    [x1,y1,w,h] --> [x1,y1,x2,y2] ,x2 = x1 + w; y2= y1+ h
    :param rects:
    :return:
    """
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


def clear_cache_by_len(cache, len_cache):
    if len(cache) > len_cache:
        thread = threading.Thread(
            target=clear_cache,
            args=(cache,))
        thread.start()


def generate_time_stamp(fmt='%m%d%H%M'):
    return time.strftime(fmt, time.localtime(time.time()))


def get_local_time(diff=0):
    return time.time() - diff


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


def paint_chinese_opencv(im, text, pos, color=None):
    if color is None:
        color = np.random.randint(0, 255, size=(3,))
        color = [int(c) for c in color]
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 50)
    fillColor = (color[0], color[1], color[2])  # (255,0,0)
    position = pos  # (100,100)
    if not isinstance(text, np.unicode):
        text = text.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text((position[0], position[1] - 60), text, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def split_img_to_four(image):
    """
    :param image:
    :return:
    """
    a1 = np.vsplit(image, 2)[0]
    a2 = np.vsplit(image, 2)[1]
    b1 = np.hsplit(a1, 2)[0]
    b2 = np.hsplit(a1, 2)[1]
    b3 = np.hsplit(a2, 2)[0]
    b4 = np.hsplit(a2, 2)[1]
    return b1, b2, b3, b4


def decode(frame, p, patch_idx):
    """
    recover original coordinates form four-dividen blocks
    :param frame:
    :param p:
    :param patch_idx:
    :return:
    """
    h, w, _ = frame.shape
    diff_h = h / 2
    diff_w = w / 2

    if not len(p[:, :]):
        return

    if patch_idx == 0:
        pass
    elif patch_idx == 1:
        p[:, :, 0] = p[:, :, 0] + diff_w
        p[:, :, 2] = p[:, :, 2] + diff_w
    elif patch_idx == 2:
        p[:, :, 1] = p[:, :, 1] + diff_h
        p[:, :, 3] = p[:, :, 3] + diff_h
    elif patch_idx == 3:
        p[:, :, 0] = p[:, :, 0] + diff_w
        p[:, :, 2] = p[:, :, 2] + diff_w
        p[:, :, 1] = p[:, :, 1] + diff_h
        p[:, :, 3] = p[:, :, 3] + diff_h


def to_bboxs_wh(rects):
    """
    [x1,y1,x2,y2] to [x1,y1,w,h]
    :param rects:
    :return:
    """
    return [to_bbox_wh(r) for r in rects]


def to_bbox_wh(r):
    return [r[0], r[1], r[2] - r[0], r[3] - r[1]]


def bgr_img_mean(img):
    b_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    r_mean = np.mean(img[:, :, 2])
    return np.array([b_mean, g_mean, r_mean])

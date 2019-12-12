#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: detect_funcs.py
@time: 2019/12/12 21:45
@version 1.0
@desc:
"""
import time
from multiprocessing import Queue
import traceback

# from interface import thresh as Thresh
# import interface

import imutils

from config import VideoConfig
from utils import *
import time
from .detector import DetectionResult


def back(self, rects, original_shape):
    b_rects = []
    ratio = original_shape[0] / self.shape[0]
    for r in rects:
        x = int((r[0] + self.start[0]) * ratio)
        y = int((r[1] + self.start[1]) * ratio)
        w = int(r[2] * ratio)
        h = int(r[3] * ratio)
        b_rects.append((x, y, w, h))
    return b_rects


def ratio(area, total):
    return (area / total) * 100


def is_in_ratio(area, total, cfg: VideoConfig):
    # logger.info('Area ration: [{}]'.format((area / total) * 100))
    return ratio(area, total) <= cfg.filtered_ratio


def less_ratio(area, shape, cfg: VideoConfig):
    total = shape[0] * shape[1]
    # logger.info('Area ration: [{}]'.format(self.ratio(area, total)))
    return ratio(area, total) >= cfg.alg['area_ratio']


def back(rects, start, shape, original_shape):
    b_rects = []
    ratio = original_shape[0] / shape[0]
    for r in rects:
        x = int((r[0] + start[0]) * ratio)
        y = int((r[1] + start[1]) * ratio)
        w = int(r[2] * ratio)
        h = int(r[3] * ratio)
        b_rects.append((x, y, w, h))
    return b_rects


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


def detect_based_task(block, params: DetectorParams):
    frame = block.frame
    # if args.cfg.alg['type'] == 'saliency':
    #     res = detect_saliency()
    if params.cfg.alg['type'] == 'thresh':
        res = detect_thresh_task(frame, block, params)
    # if args.cfg.alg['type'] == 'thresh_mask':
    #     frame = self.get_frame()
    #     self.shape = frame.shape
    #     mask = np.zeros((self.shape[0], self.shape[1])).astype(np.uint8)
    #     mask[60:420, :] = 255
    #     res = self.detect_mask_task(frame, mask, block)
    return res


def detect_thresh_task(frame, block, params: DetectorParams):
    start = time.time()
    if frame is None:
        logger.info('Detector: [{},{}] empty frame')
        return
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    adaptive_thresh = adaptive_thresh_size(frame, (5, 5), block_size=21, C=params.cfg.alg['mean'])
    # adaptive_thresh = cv2.bitwise_and(adaptive_thresh, mask)
    dilated = cv2.dilate(adaptive_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=1)
    img_con, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rects = []
    regions = []
    status = None
    coordinates = []
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # self.is_in_ratio(area, self.shape[0] * self.shape[1])
        if less_ratio(area, frame.shape, params.cfg) and rect[2] / rect[3] < 10:
            rects.append(rect)
    cv2.drawContours(img_con, contours, -1, 255, -1)
    # if self.cfg.show_window:
    #     cv2.imshow("Contours", img_con)
    #     cv2.waitKey(1)
    # self.detect_cnt += 1
    # logger.info(
    #     '~~~~ Detector: [{},{}] detect done [{}] frames..'.format(params.col_index, params.row_index,
    #                                                               params))
    res = DetectionResult(None, None, status, regions, dilated, dilated, coordinates, params.row_index,
                          params.col_index, block.index, back(rects, params.start, frame.shape, block.shape))
    end = time.time() - start
    logger.info('Detector: [{},{}]: using [{}] seconds'.format(params.col_index, params.row_index, end))
    # cv2.destroyAllWindows()
    return res

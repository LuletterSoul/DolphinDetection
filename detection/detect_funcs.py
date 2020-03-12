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
from typing import List

import imutils

from .component import mog2_dict
from detection.params import ConstructResult, ConstructParams, BlockInfo, DetectorParams
from utils import *
from .detector import DetectionResult


# import ray


# from interface import thresh as Thresh
# import interface


def ratio(area, total):
    return (area / total) * 100


def is_in_ratio(area, total, cfg: VideoConfig):
    # logger.info('Area ration: [{}]'.format((area / total) * 100))
    return ratio(area, total) <= cfg.filtered_ratio


def less_ratio(area, shape, cfg: VideoConfig):
    if area == -1:
        return True
    total = shape[0] * shape[1]
    # logger.info('Area ration: [{}]'.format(ratio(area, total)))
    return ratio(area, total) >= cfg.alg['area_ratio']


def greater_ratio(area, shape, cfg: VideoConfig):
    if cfg.alg['area_ratio'] == -1:
        return True
    total = shape[0] * shape[1]
    return ratio(area, total) < (cfg.alg['area_ratio'] * 3.5)


# @ray.remote
def detect_based_task(block, params: DetectorParams) -> DetectionResult:
    frame = block.frame
    # if args.cfg.alg['type'] == 'saliency':
    #     res = detect_saliency()

    if params.cfg.alg['type'] == 'mog2':
        res = detect_based_mog2(frame, block, params)

    elif params.cfg.alg['type'] == 'thresh':
        shape = frame.shape
        # logger.info(shape)
        # logger.info(params.start)
        # logger.info(params.end)
        res = detect_thresh_task(frame, block, params)
    elif params.cfg.alg['type'] == 'thresh_mask':
        shape = frame.shape
        mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
        mask[60:420, :] = 255
        res = detect_mask_task(frame, mask, block, params)
    return res


def detect_based_mog2(frame, block, params: DetectorParams):
    cfg = params.cfg
    mog2 = mog2_dict[cfg.index]
    start = time.time()
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    if frame is None:
        logger.info('Detector: [{},{}] empty frame')
        return
    binary = mog2.apply(frame)
    # erode = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    binary = cv2.erode(binary, erode_kernel)
    binary = cv2.dilate(binary, dilate_kernel)
    img_con, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rects = []
    regions = []
    status = None
    coordinates = []
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # self.is_in_ratio(area, self.shape[0] * self.shape[1])
        if less_ratio(area, frame.shape, params.cfg) and rect[2] / rect[3] < 10:
            logger.info("~~~~~~~~~~~~~~~~~~~~~~~Area:[{}]".format(area))
            rects.append(rect)
    cv2.drawContours(img_con, contours, -1, 255, -1)
    # if self.cfg.show_window:
    #     cv2.imshow("Contours", img_con)
    #     cv2.waitKey(1)
    # self.detect_cnt += 1
    # logger.info(
    #     '~~~~ Detector: [{},{}] detect done [{}] frames..'.format(params.col_index, params.row_index,
    #                                                               params))
    res = DetectionResult(None, None, status, regions, binary, binary, coordinates, params.x_index,
                          params.y_index, block.index, back(rects, params.start, frame.shape, block.shape, params.cfg))
    end = time.time() - start
    logger.info('Detector: [{},{}]: using [{}] seconds'.format(params.y_index, params.x_index, end))
    # cv2.destroyAllWindows()
    return res


def detect_thresh_task(frame, block, params: DetectorParams):
    start = time.time()
    if frame is None:
        logger.info('Detector: [{},{}] empty frame')
        return
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    adaptive_thresh = adaptive_thresh_size(frame, (5, 5), block_size=21, C=params.cfg.alg['mean'])
    adaptive_thresh = cv2.erode(adaptive_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                iterations=2)
    # adaptive_thresh = cv2.bitwise_and(adaptive_thresh, mask)
    dilated = cv2.dilate(adaptive_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=2)
    img_con, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    binary = np.zeros(dilated.shape, dtype=np.uint8)
    rects = []
    regions = []
    status = None
    coordinates = []
    filtered_contours = []
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # self.is_in_ratio(area, self.shape[0] * self.shape[1])
        if less_ratio(area, frame.shape, params.cfg) and greater_ratio(area, frame.shape, params.cfg) and rect[2] / \
                rect[3] < 10:
            rects.append(rect)
            filtered_contours.append(c)
    cv2.drawContours(binary, filtered_contours, -1, 255, -1)
    # if self.cfg.show_window:
    #     cv2.imshow("Contours", img_con)
    #     cv2.waitKey(1)
    # self.detect_cnt += 1
    # logger.info(
    #     '~~~~ Detector: [{},{}] detect done [{}] frames..'.format(params.col_index, params.row_index,
    #                                                               params))
    original_rects = back(rects, params.start, frame.shape, block.shape, params.cfg)
    res = DetectionResult(None, None, status, regions, binary, dilated, coordinates, params.x_index,
                          params.y_index, block.index, original_rects, rects)
    end = time.time() - start
    logger.debug('Detector: [{},{}]: using [{}] seconds'.format(params.y_index, params.x_index, end))
    # cv2.destroyAllWindows()
    return res


def detect_mask_task(frame, mask, block, params: DetectorParams):
    start = time.time()
    if frame is None:
        logger.info('Detector: [{},{}] empty frame')
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    adaptive_thresh = adaptive_thresh_size(frame, kernel_size=(5, 5), block_size=51, C=params.cfg.alg['mean'])
    dilated = cv2.dilate(adaptive_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=1)
    dilated = cv2.bitwise_and(dilated, mask)
    img_con, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rects = []
    regions = []
    status = None
    coordinates = []
    for c in contours:
        rect = cv2.boundingRect(c)
        rects.append(rect)
    cv2.drawContours(img_con, contours, -1, 255, -1)
    # if self.cfg.show_window:
    #     cv2.imshow("Contours", img_con)
    #     cv2.waitKey(1)
    # self.detect_cnt += 1
    res = DetectionResult(None, None, status, regions, dilated, dilated, coordinates, params.x_index,
                          params.y_index, block.index, back(rects, params.start, frame.shape, block.shape, params.cfg))
    end = time.time() - start
    logger.info('Detector: [{},{}]: using [{}] seconds'.format(params.y_index, params.x_index, end))
    return res


def detect(frame):
    if frame is None:
        logger.info('Detector: [{},{}] empty frame')
        return None
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    adaptive_thresh = adaptive_thresh_size(frame, kernel_size=(5, 5), block_size=21, C=40)
    cv2.imshow('CV', adaptive_thresh)
    cv2.waitKey(0)
    dilated = cv2.dilate(adaptive_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=1)
    img_con, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rects = []
    for c in contours:
        rect = cv2.boundingRect(c)
        rects.append(rect)
    cv2.drawContours(img_con, contours, -1, 255, -1)
    # if self.cfg.show_window:
    #     cv2.imshow("Contours", img_con)
    #     cv2.waitKey(1)
    # self.detect_cnt += 1
    return rects


def collect(args):
    return [f.get() for f in args]


def draw_boundary(frame, info: BlockInfo):
    shape = frame.shape
    for i in range(info.x_num - 1):
        start = (0, info.x_step * (i + 1))
        end = (shape[1] - 1, info.x_step * (i + 1))
        cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    for j in range(info.y_num - 1):
        start = (info.y_step * (j + 1), 0)
        end = (info.y_step * (j + 1), shape[0] - 1)
        cv2.line(frame, start, end, (0, 0, 255), thickness=1)
    return frame


# @ray.remote
def collect_and_reconstruct(args, params: ConstructParams, block_info: BlockInfo, cfg: VideoConfig):
    results = collect(args)
    construct_result: ConstructResult = construct(results, params)
    if construct_result is not None:
        frame = construct_result.frame
        if cfg.draw_boundary:
            frame = draw_boundary(frame, block_info)
            # logger.info('Done constructing of sub-frames into a original frame....')
        if cfg.show_window:
            frame = imutils.resize(frame, width=800)
            cv2.imshow('Reconstructed Frame', frame)
            cv2.waitKey(1)
    else:
        logger.error('Empty reconstruct result.')
    return True


# @ray.remote
def construct(results: List[DetectionResult], params: ConstructParams):
    # sub_frames = [r.frame for r in results]
    # sub_binary = [r.binary for r in results]
    # sub_thresh = [r.thresh for r in results]
    # constructed_frame = self.construct_rgb(sub_frames)
    # constructed_binary = self.construct_gray(sub_binary)
    # constructed_thresh = self.construct_gray(sub_thresh)
    logger.info('Controller [{}]: Construct frames into a original frame....'.format(params.cfg.index))
    try:
        # self.construct_cnt += 1
        current_index = results[0].frame_index
        while current_index not in params.original_frame_cache:
            logger.info('Current index: [{}] not in original frame cache.May cache was cleared by timer'.format(
                current_index))
            time.sleep(0.5)
            # logger.info(self.original_frame_cache.keys())
        original_frame = params.original_frame_cache[current_index]
        last_detection_time = None
        for r in results:
            if len(r.rects):
                params.result_queue.put(original_frame)
                last_detection = time.time()
                if r.frame_index not in params.original_frame_cache:
                    logger.info('Unknown frame index: [{}] to fetch frame in cache.'.format(r.frame_index))
                    continue
                for rect in r.rects:
                    color = np.random.randint(0, 255, size=(3,))
                    color = [int(c) for c in color]
                    p1 = (rect[0] - 80, rect[1] - 80)
                    p2 = (rect[0] + 100, rect[1] + 100)
                    # cv2.rectangle(original_frame, (rect[0] - 20, rect[1] - 20),
                    #               (rect[0] + rect[2] + 20, rect[1] + rect[3] + 20),
                    #               color, 2)
                    cv2.rectangle(original_frame, p1, p2, color, 2)
                params.render_frame_cache[current_index] = original_frame
                params.render_rect_cache[current_index] = r.rects
                params.stream_render.reset(current_index)
        params.stream_render.notify(current_index)
        clear_cache_by_len(params.render_frame_cache, params.len_thresh)
        clear_cache_by_len(params.render_rect_cache, params.len_thresh)
        # return constructed_frame, constructed_binary, constructed_thresh
        return ConstructResult(original_frame, None, None, last_detection_time)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)


import cv2

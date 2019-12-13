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
from typing import List
from utils import *
from .detector import DetectionResult
from detection.params import ConstructResult, ConstructParams, BlockInfo, DetectorParams
import imutils
# import ray


# from interface import thresh as Thresh
# import interface


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


# @ray.remote
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


def collect(args):
    return [f.get() for f in args]


def draw_boundary(frame, info: BlockInfo):
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


# @ray.remote
def collect_and_reconstruct(args, params: ConstructParams, block_info: BlockInfo, cfg: VideoConfig):
    results = ray.get(args)
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

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


def is_greater_ratio(area, shape, cfg: VideoConfig):
    if cfg.alg['area_ratio'] == -1:
        return True
    total = shape[0] * shape[1]
    # logger.info('Area ration: [{}]'.format(ratio(area, total)))
    return ratio(area, total) >= cfg.alg['area_ratio']


def is_less_ratio(area, shape, cfg: VideoConfig):
    if cfg.alg['area_ratio'] == -1:
        return True
    total = shape[0] * shape[1]
    # logger.info(f'Area [{ratio(area,total)}],area [{area}],total [{total}]')
    return ratio(area, total) < (cfg.alg['area_ratio'] * 3)


# @ray.remote
def detect_based_task(block, params: DetectorParams) -> DetectionResult:
    """
    coarser detection by traditional patter recognition method
    :param block:
    :param params:
    :return:
    """
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
        res = adaptive_thresh_with_rules(frame, block, params)
    elif params.cfg.alg['type'] == 'thresh_mask':
        shape = frame.shape
        mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
        mask[60:420, :] = 255
        res = adaptive_thresh_mask_no_rules(frame, mask, block, params)
    return res


def detect_based_mog2(frame, block, params: DetectorParams):
    cfg = params.cfg
    mog2 = mog2_dict[cfg.index]
    start = time.time()
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    if frame is None:
        logger.info('Detector: [{},{}] empty frame')
        return
    binary = mog2.apply(frame)
    # erode = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # binary = cv2.erode(binary, erode_kernel)
    binary = cv2.dilate(binary, kernel)
    # img_con, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(binary)
    rects = []
    regions = []
    status = None
    coordinates = []
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # self.is_in_ratio(area, self.shape[0] * self.shape[1])
        if is_greater_ratio(area, frame.shape, params.cfg) and rect[2] / rect[3] < 10:
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


def adaptive_thresh_with_rules(frame, block, params: DetectorParams):
    """
    perform adaptive binary thresh with filter rules
    :param frame: preprocessed frame by preprocessed module, may be smaller and blurred than the original.
    :param block:
    :param params: contain some data structure such as video configuration...
    :return:
    """
    start = time.time()

    # construct kernel element
    dk_size = params.cfg.alg['dk_size']
    ok_size = params.cfg.alg['ok_size']

    frame = cv2.pyrMeanShiftFiltering(frame, params.cfg.alg['sp'], params.cfg.alg['sr'])

    cv2.namedWindow(str(params.cfg.index) + '-' + 'Smooth', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(str(params.cfg.index) + '-' + 'Smooth', frame)
    cv2.waitKey(1)
    # adaptive thresh by size
    thresh_binary = adaptive_thresh_size(frame, block_size=params.cfg.alg['block_size'],
                                         C=params.cfg.alg['mean'])
    # TODO using multiple scales thresh to filter small object or noises
    # remove small objects
    if ok_size != -1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok_size, ok_size))
        binary = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, open_kernel)
    else:
        binary = thresh_binary

    # enlarge candidates a little
    if dk_size != -1:
        dilated_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk_size, dk_size))
        binary = cv2.dilate(binary, dilated_kernel)

    color_scale = params.cfg.alg['color_scale']
    color_range = params.cfg.alg['color_range']
    global_mean = cv2.mean(frame)
    if color_scale != -1:
        color_range = [global_mean[0] / color_scale, global_mean[1] / color_scale, global_mean[2] / color_scale]

    # global_mean = cv2.mean(frame)

    # compute linked components
    num_components, label_map, rects, centroids = cv2.connectedComponentsWithStats(binary)

    binary_map = np.zeros(binary.shape, dtype=np.uint8)
    global_binary_map = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, num_components):
        global_binary_map[label_map == i] = 255
    if params.cfg.show_window:
        cv2.namedWindow(str(params.cfg.index) + '-' + 'Global Binary', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(str(params.cfg.index) + '-' + 'Global Binary', global_binary_map)
        cv2.waitKey(1)
    filtered_rects = []
    block_bgr_means = []  # candidate block color mean
    # 0 index is background,skipped it
    for i in range(1, num_components):
        # only process if it has more than defualt(50pix) area
        # block color range is black enough
        if rects[i][cv2.CC_STAT_AREA] > params.cfg.alg['area'] and is_block_black(frame, i, label_map, color_range):
            # if rects[i][cv2.CC_STAT_AREA] > params.cfg.alg['area']:
            # if rects[i][cv2.CC_STAT_AREA] > params.cfg.alg['area']:
            # draw white pixels if current is a components
            # logger.info(f'Area: {rects[i][cv2.CC_STAT_AREA]}')
            # logger.info(f'Height: {rects[i][cv2.CC_STAT_HEIGHT]}')
            # logger.info(f'Width: {rects[i][cv2.CC_STAT_WIDTH]}')
            binary_map[label_map == i] = 255
            # merge all white blocks into a single binary map
            filtered_rects.append(rects[i])
    # rect coordinates in original frame
    original_rects = back(filtered_rects, params.start, frame.shape, block.shape, params.cfg)

    # load rect width and height thresh value from configuration
    # rect_width_thresh = params.cfg.alg['rwt']
    # rect_height_thresh = params.cfg.alg['rht']

    # dolphin size internal is usually at x~[200,300] pixels,y~[50,100] pixels in 12 focus, 1080P monitor
    # should be adjusted according the different focuses
    # original_rects = [rect for rect in original_rects if
    #                   abs(rect[2] - rect[0]) > rect_width_thresh and abs(rect[3] - rect[1]) > rect_height_thresh]
    if params.cfg.alg['filtered_by_wh']:
        original_rects = filtered_shot_block(original_rects, params.cfg)
    res = DetectionResult(None, None, None, None, binary_map, binary, [], params.x_index,
                          params.y_index, block.index, original_rects, rects)
    end = time.time() - start
    logger.debug('Detector: [{},{}]: using [{}] seconds'.format(params.y_index, params.x_index, end))
    return res


def cal_block_bgr_mean(frame, label, label_map):
    mask = (label_map == label).astype(np.uint8)
    block_pixels = mask.sum()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    block = cv2.bitwise_and(frame, mask)
    b_mean = np.sum(block[:, :, 0]) / block_pixels
    g_mean = np.sum(block[:, :, 1]) / block_pixels
    r_mean = np.sum(block[:, :, 2]) / block_pixels
    mean = [b_mean, g_mean, r_mean]
    return mean


def filtered_shot_block(rects, cfg: VideoConfig):
    rect_width_thresh = cfg.alg['rwt']
    rect_height_thresh = cfg.alg['rht']
    min_len = min(rect_width_thresh, rect_height_thresh)
    max_len = max(rect_width_thresh, rect_height_thresh)
    filtered_rects = []
    for r in rects:
        w = r[2] - r[0]
        h = r[3] - r[1]
        width = max(w, h)
        height = min(w, h)
        # dolphin may appear with a head only,width equal to heightz
        if (width >= max_len and height >= min_len) or (width >= max_len / 2 and height >= max_len / 2):
            filtered_rects.append(r)
    return filtered_rects


def is_block_black(frame, i, label_map, color_range):
    mean = cal_block_bgr_mean(frame, i, label_map)
    logger.info(f'Block mean: {mean}')
    logger.info(f'Color range {color_range}')
    return mean[0] < color_range[0] and mean[1] < color_range[1] and mean[2] < \
           color_range[2]


def adaptive_thresh_mask_no_rules(frame, mask, block, params: DetectorParams):
    """
    perform adaptive binary thresh without filtering rule, use a mask to exclude timestamp region when thresh
    :param frame:
    :param mask:
    :param block:
    :param params:
    :return:
    """
    start = time.time()
    if frame is None:
        logger.debug('Detector: [{},{}] empty frame')
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    adaptive_thresh = adaptive_thresh_size(frame, kernel_size=(5, 5), block_size=51, C=params.cfg.alg['mean'])
    dilated = cv2.dilate(adaptive_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=1)
    # exclude region mask
    dilated = cv2.bitwise_and(dilated, mask)
    # return value number of findContours will be different opencv version will be different
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # imcon,contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) three return values
    rects = []
    regions = []
    status = None
    coordinates = []
    for c in contours:
        rect = cv2.boundingRect(c)
        rects.append(rect)
    # cv2.drawContours(img_con, contours, -1, 255, -1)
    # if self.cfg.show_window:
    #     cv2.imshow("Contours", img_con)
    #     cv2.waitKey(1)
    # self.detect_cnt += 1
    res = DetectionResult(None, None, status, regions, dilated, dilated, coordinates, params.x_index,
                          params.y_index, block.index, back(rects, params.start, frame.shape, block.shape, params.cfg))
    end = time.time() - start
    logger.debug('Detector: [{},{}]: using [{}] seconds'.format(params.y_index, params.x_index, end))
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
                params.stream_render.reset(None)
        params.stream_render.notify(None)
        clear_cache_by_len(params.render_frame_cache, params.len_thresh)
        clear_cache_by_len(params.render_rect_cache, params.len_thresh)
        # return constructed_frame, constructed_binary, constructed_thresh
        return ConstructResult(original_frame, None, None, last_detection_time)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)

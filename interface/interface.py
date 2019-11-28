#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: main.py
@time: 2019/11/15 14:53
@version 1.0
@desc:
"""

import traceback

import detection
import stream
from config import *
from utils.log import logger
import utils.thresh as thresh
from typing import List
from multiprocessing import Queue


def detect(video_path, candidate_save_path, mq: Queue, cfg):
    """
    run object detection algorithm
    :param video_path: video stream save path
    :param candidate_save_path: the region candidates
    :param mq: process communication pipe in which alg will read the newest stream index
    :param cfg: video configuration
    :return:
    """
    try:
        detection.detect(video_path, candidate_save_path, mq, cfg)
        return True
    except Exception as e:
        traceback.print_exc()
        logger.error(e)


def thresh(frame, cfg=None):
    """
    do frame binarization based adaptive thresh
    :param frame:
    :return:
    """
    return thresh.adaptive_thresh(frame, cfg)


def read_stream(stream_save_path, vcfg, mq: Queue):
    """
    read real-time video stream from provided configuration
    :param stream_save_path: streams of each video will be save here
    :param vcfg: video configurations
    :param mq: process communication pipe in which stream receiver will write
    the newest stream index,coorperated with # object detector
    :return:
    """
    stream.read(stream_save_path, vcfg, mq)
    return True


def read_frame(input_path: Path, output_path: Path):
    """
    read frames from a stream
    :param input_path:
    :param output_path:
    :return:
    """
    return stream.process_video(input_path, output_path)


def load_video_config(cfg_path: Path) -> List[VideoConfig]:
    """
    load video configuration into a dict object from json file
    :param cfg_path:
    :return:
    """
    with open(cfg_path) as f:
        cfg_objs = json.load(f)['videos']
        cfgs = [VideoConfig.from_json(c) for c in cfg_objs]
        return cfgs

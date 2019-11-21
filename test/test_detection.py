#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test_detection_monitor.py
@time: 2019/11/21 14:20
@version 1.0
@desc:
"""
from unittest import TestCase
import detection
from config import *


class TestDetection(object):

    def test_detect_embadding_monitor(self):
        # monitor = detection.DetectionMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR, CANDIDATE_SAVE_DIR)
        monitor = detection.EmbeddingControlBasedProcessMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
                                                                CANDIDATE_SAVE_DIR)
        monitor.monitor()

    def test_control(self):
        # controller = detection.DetectorController()
        # self.fail()
        pass


if __name__ == '__main__':
    test_detection = TestDetection()
    test_detection.test_detect_embadding_monitor()
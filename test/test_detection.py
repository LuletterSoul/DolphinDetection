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
import traceback


class TestDetection(object):

    def test_detect_embedding_based_process_monitor(self):
        # monitor = detection.DetectionMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR, CANDIDATE_SAVE_DIR)
        monitor = detection.EmbeddingControlBasedProcessMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR, None,
                                                                CANDIDATE_SAVE_DIR)
        monitor.monitor()

    def test_detect_embedding_based_thread_monitor(self):
        try:
            monitor = detection.EmbeddingControlBasedThreadMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
                                                                   None, CANDIDATE_SAVE_DIR)
            monitor.monitor()
        except Exception as e:
            traceback.print_exc()

    def test_detect_embedding_based_thread_and_process_monitor(self):
        monitor = detection.EmbeddingControlBasedThreadAndProcessMonitor(VIDEO_CONFIG_DIR / 'video.json',
                                                                         STREAM_SAVE_DIR, None, None,
                                                                         CANDIDATE_SAVE_DIR)
        monitor.monitor()

    def test_control(self):
        # controller = detection.DetectorController()
        # self.fail()
        pass


if __name__ == '__main__':
    test_detection = TestDetection()
    # test_detection.test_detect_embedding_based_process_monitor()
    # test_detection.test_detect_embedding_based_thread_monitor()
    test_detection.test_detect_embedding_based_thread_and_process_monitor()

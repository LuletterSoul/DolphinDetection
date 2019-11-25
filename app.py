#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: app.py
@time: 11/22/19 11:11 AM
@version 1.0
@desc:
"""

from multiprocessing import Process

import detection
import interface
from config import *

if __name__ == '__main__':
    if MONITOR == MonitorType.PROCESS_THREAD_BASED:
        monitor = detection.EmbeddingControlBasedThreadAndProcessMonitor(VIDEO_CONFIG_DIR / 'video.json',
                                                                         STREAM_SAVE_DIR,
                                                                         CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)

    elif MONITOR == MonitorType.PROCESS_BASED:
        monitor = detection.EmbeddingControlBasedProcessMonitor(VIDEO_CONFIG_DIR / 'video.json',
                                                                STREAM_SAVE_DIR,
                                                                CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
    else:
        monitor = detection.EmbeddingControlBasedThreadMonitor(VIDEO_CONFIG_DIR / 'video.json',
                                                               STREAM_SAVE_DIR,
                                                               CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)

    # process = Process(target=monitor.monitor)
    # process.start()
    # process.join()
    monitor.monitor()

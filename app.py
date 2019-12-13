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

import detection
from config import *
import ray

# ray.init()

if __name__ == '__main__':
    # if MONITOR == MonitorType.RAY_BASED:
    #     # ray.init(object_store_memory=8 * 1024 * 1024)

    #     ray.init()
    #     try:
    #         monitor = detection.EmbeddingControlBasedRayMonitor.remote(VIDEO_CONFIG_DIR / 'video.json',
    #                                                                    STREAM_SAVE_DIR, SAMPLE_SAVE_DIR,
    #                                                                    FRAME_SAVE_DIR,
    #                                                                    CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
    #         m_id = monitor.monitor.remote()
    #         ray.get(m_id)
    #         print(ray.errors(all_jobs=True))
    #     except Exception as e:
    #         print(e)

    if MONITOR == MonitorType.PROCESS_THREAD_BASED:
        monitor = detection.EmbeddingControlBasedThreadAndProcessMonitor(VIDEO_CONFIG_DIR / 'video.json',
                                                                         STREAM_SAVE_DIR, SAMPLE_SAVE_DIR,
                                                                         FRAME_SAVE_DIR,
                                                                         CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
        monitor.monitor()

    elif MONITOR == MonitorType.PROCESS_BASED:
        monitor = detection.EmbeddingControlBasedProcessMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
                                                                SAMPLE_SAVE_DIR,
                                                                FRAME_SAVE_DIR,
                                                                CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)

        monitor.monitor()
    elif MONITOR == MonitorType.TASK_BASED:
        monitor = detection.EmbeddingControlBasedTaskMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
                                                             SAMPLE_SAVE_DIR,
                                                             FRAME_SAVE_DIR,
                                                             CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
        monitor.monitor()
    else:
        monitor = detection.EmbeddingControlBasedThreadMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
                                                               SAMPLE_SAVE_DIR,
                                                               FRAME_SAVE_DIR,
                                                               CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
        monitor.monitor()

# process = Process(target=monitor.monitor)
# process.start()
# process.join()

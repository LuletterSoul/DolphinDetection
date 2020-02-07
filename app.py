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
import multiprocessing as mp
# from multiprocessing import Process
from config import *
import sys
from interface import *
from stream.http import HttpServer
from classfy.model import model
import traceback
import numpy as np


class DolphinDetectionServer:

    def __init__(self, cfg: ServerConfig, vcfgs: List[VideoConfig], switcher_options) -> None:
        self.cfg = cfg
        self.vcfgs = [c for c in vcfgs if switcher_options[str(c.index)]]
        self.monitor = detection.EmbeddingControlBasedTaskMonitor(self.vcfgs,
                                                                  self.cfg.stream_save_path,
                                                                  self.cfg.sample_save_dir,
                                                                  self.cfg.frame_save_dir,
                                                                  self.cfg.candidate_save_dir,
                                                                  self.cfg.offline_stream_save_dir)
        self.http_server = HttpServer(self.cfg.http_ip, self.cfg.http_port, self.cfg.mode)

    def run(self):
        """
        System Entry
        """
        self.http_server.run()
        self.monitor.monitor()


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
    # if MONITOR == MonitorType.PROCESS_THREAD_BASED:
    #     monitor = detection.EmbeddingControlBasedThreadAndProcessMonitor(VIDEO_CONFIG_DIR / 'video.json',
    #                                                                      STREAM_SAVE_DIR, SAMPLE_SAVE_DIR,
    #                                                                      FRAME_SAVE_DIR,
    #                                                                      CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
    #     monitor.monitor()
    #
    # elif MONITOR == MonitorType.PROCESS_BASED:
    #     monitor = detection.EmbeddingControlBasedProcessMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
    #                                                             SAMPLE_SAVE_DIR,
    #                                                             FRAME_SAVE_DIR,
    #                                                             CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
    #
    #     monitor.monitor()
    # elif MONITOR == MonitorType.TASK_BASED:
    #     monitor = detection.EmbeddingControlBasedTaskMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
    #                                                          SAMPLE_SAVE_DIR,
    #                                                          FRAME_SAVE_DIR,
    #                                                          CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
    #     monitor.monitor()
    #
    # else:
    #     monitor = detection.EmbeddingControlBasedThreadMonitor(VIDEO_CONFIG_DIR / 'video.json', STREAM_SAVE_DIR,
    #                                                            SAMPLE_SAVE_DIR,
    #                                                            FRAME_SAVE_DIR,
    #                                                            CANDIDATE_SAVE_DIR, OFFLINE_STREAM_SAVE_DIR)
    #     monitor.monitor()
    server_config = load_server_config('vcfg/server.json')
    video_config = load_video_config('vcfg/video.json')
    switcher_options = load_json_config('vcfg/switcher.json')
    server = DolphinDetectionServer(server_config, video_config, switcher_options)
    server.run()

# process = Process(target=monitor.monitor)
# process.start()
# process.join()

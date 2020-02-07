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
import argparse
from stream.http import HttpServer
from classfy.model import DolphinClassifier
import traceback
import numpy as np


class DolphinDetectionServer:

    def __init__(self, cfg: ServerConfig, vcfgs: List[VideoConfig], switcher_options) -> None:
        self.cfg = cfg
        self.vcfgs = [c for c in vcfgs if switcher_options[str(c.index)]]
        self.classification_model = DolphinClassifier(self.cfg.classify_model_path)
        self.monitor = detection.EmbeddingControlBasedTaskMonitor(self.vcfgs,
                                                                  self.classification_model,
                                                                  self.cfg.stream_save_path,
                                                                  self.cfg.sample_save_dir,
                                                                  self.cfg.frame_save_dir,
                                                                  self.cfg.candidate_save_dir,
                                                                  self.cfg.offline_stream_save_dir)
        self.http_server = HttpServer(self.cfg.http_ip, self.cfg.http_port, self.cfg.mode, self.cfg.candidate_save_dir)

    def run(self):
        """
        System Entry
        """
        self.http_server.run()
        self.classification_model.run()
        self.monitor.monitor()


def load_cfg(args):
    server_config = load_server_config(args.cfg)
    video_config = load_video_config(args.vcfg)
    switcher_options = load_json_config(args.switcher)
    if args.http_ip is not None:
        server_config.http_ip = args.http_ip
    if args.http_port is not None:
        server_config.http_port = args.http_port
    if args.root is not None:
        server_config.set_root(args.root)
    if args.cdp is not None:
        server_config.set_candidate_save_dir(args.cdp)
    return server_config, video_config, switcher_options


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
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--cfg', type=str, default='vcfg/server.json',
                        help='Server configuration file represented by json format.')
    parser.add_argument('--vcfg', type=str, default='vcfg/video.json',
                        help='Video configuration file represented by json format.')

    parser.add_argument('--switcher', type=str, default='vcfg/switcher.json',
                        help='Definite which video should be monitored.')

    parser.add_argument('--http_ip', type=str, help='Http server ip address')
    parser.add_argument('--http_port', type=int, help='Http server listen port')
    parser.add_argument('--root', type=str,
                        help='Redirect data root path,default path is [PROJECT DIR/data/...].' +
                             'PROJECT DIR is depend on which project is deployed.' +
                             'If root param is set,data path is redirected as [root/data/...]')
    parser.add_argument('--cdp', type=str,
                        help='Dolphin video stream storage relative path.default path is [$PROJECT DIR$/data/candidate].' +
                             'If cdp param is set,stream storage path is redirected as [$root$/$cdp$] ' +
                             'or [$PROJECT DIR$]/$cdp$.')
    args = parser.parse_args()
    server_config, video_config, switcher_options = load_cfg(args)
    server = DolphinDetectionServer(server_config, video_config, switcher_options)
    server.run()

# process = Process(target=monitor.monitor)
# process.start()
# process.join()

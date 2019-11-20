#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: manager.py
@time: 2019/11/16 13:22
@version 1.0
@desc:
"""

from multiprocessing import Manager, Pool
import interface as I
from pathlib import Path
from config import VideoConfig
from typing import List
from utils import clean_dir


class DetectionMonitor(object):

    def __init__(self, video_config_path: Path, stream_path: Path, candidate_path: Path) -> None:
        super().__init__()
        self.cfgs = I.load_video_config(video_config_path)[:1]
        # Communication Pipe between detector and stream receiver
        self.pipes = [Manager().Queue() for c in self.cfgs]
        self.stream_path = stream_path
        self.candidate_path = candidate_path
        self.pool = Pool()

    def monitor(self):
        for i, cfg in enumerate(self.cfgs):
            # clean all legacy streams and candidates files before initialization
            clean_dir(self.stream_path)
            clean_dir(self.candidate_path)
            self.pool.apply_async(I.read_stream, (self.stream_path / str(cfg.index), cfg, self.pipes[i],))
            self.pool.apply_async(I.detect,
                                  (self.stream_path / str(cfg.index), self.candidate_path / str(cfg.index),
                                   self.pipes[i], cfg,))
        self.pool.close()
        self.pool.join()

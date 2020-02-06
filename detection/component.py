#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: component.py
@time: 2/4/20 3:16 PM
@version 1.0
@desc:
"""

from multiprocessing import Pipe

import cv2
import imutils
import numpy as np

import interface
from config import *

cfgs = interface.load_video_config(VIDEO_CONFIG_DIR / 'video.json')
cfgs = [c for c in cfgs if enable_options[c.index]]
stream_pipes = {}

mog2_dict = {}

for cfg in cfgs:
    # mog2_dict[cfg.index] = cv2.bgsegm.createBackgroundSubtractorGMG()
    mog2_dict[cfg.index] = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    in_pipe, out_pipe = Pipe()
    stream_pipes[cfg.index] = (in_pipe, out_pipe)


def display():
    col = 4
    row = len(stream_pipes) // col
    while True:
        if row == 0:
            frames = []
            for k, v in stream_pipes.items():
                frames.append(imutils.resize(v[1].recv(), width=500))
            frame = np.concatenate(frames, axis=0)
        else:
            frames = []
            for i in range(row):
                col_cat = []
                for j in range(col):
                    for k, v in stream_pipes.items():
                        col_cat.append(v[1].recv())
                frames.append(np.concatenate(col_cat, axis=0))
            frame = np.concatenate(frames, axis=1)
        cv2.imshow('Dolphin Monitor', frame)
        cv2.waitKey(1)

# Process(target=display).start()

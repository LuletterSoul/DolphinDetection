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
from multiprocessing import Process

stream_pipes = {}
mog2_dict = {}


def display(vcfgs):
    col = 4
    row = len(stream_pipes) // col
    while True:
        if row == 0:
            frames = []
            binary = []
            for k, v in stream_pipes.items():
                recv = v[1].recv()
                frames.append(imutils.resize(recv.frame, width=500))
                binary.append(imutils.resize(recv.binary, width=500))
            frames = np.concatenate(frames, axis=1)
            binary = np.concatenate(binary, axis=1)
        else:
            frames = []
            binary = []
            for i in range(row):
                frame_cat = []
                binary_cat = []
                for j in range(col):
                    for k, v in stream_pipes.items():
                        recv = v[1].recv()
                        binary_cat.append(imutils.resize(recv.binary, width=500))
                        frame_cat.append(imutils.resize(recv.frame, width=500))
                frames.append(np.concatenate(frame_cat, axis=1))
                binary.append(np.concatenate(binary_cat, axis=1))
            frames = np.concatenate(frames, axis=0)
            binary = np.concatenate(binary, axis=0)

        cv2.imshow('Dolphin Monitor', frames)
        cv2.imshow('Dolphin Binary Monitor', binary)
        cv2.waitKey(1)


def run_player(vcfgs):
    for cfg in vcfgs:
        # mog2_dict[cfg.index] = cv2.bgsegm.createBackgroundSubtractorGMG()
        mog2_dict[cfg.index] = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        in_pipe, out_pipe = Pipe()
        stream_pipes[cfg.index] = (in_pipe, out_pipe)
    Process(target=display, args=(vcfgs,), daemon=True).start()

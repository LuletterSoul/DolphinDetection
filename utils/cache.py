#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: FrameCache.py
@time: 3/13/20 11:39 AM
@version 1.0
@desc:
"""
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import numpy as np


class FrameCache(object):

    def __init__(self, manager: SharedMemoryManager, cache_size, unit, shape) -> None:
        self.unit = unit
        print(self.unit)
        self.shape = shape
        self.cache_size = cache_size
        self.total_bytes = self.unit * self.cache_size
        self.cache_block = manager.SharedMemory(size=self.total_bytes)

    def __getitem__(self, index):
        buf = self.get_buf(index)
        return np.ndarray(self.shape, dtype=np.uint8, buffer=buf)

    def __setitem__(self, index, frame):
        buf = self.get_buf(index)
        buf_frame = np.ndarray(self.shape, dtype=np.uint8, buffer=buf)
        buf_frame[:, :, :] = frame[:, :, :]

    def get_buf(self, index):
        cbt = self.unit * (index % self.cache_size)
        nbt = self.unit * ((index + 1) % self.cache_size)
        if nbt == 0:
            nbt = self.total_bytes
        # print((index + 1) % self.cache_size)
        buf = self.cache_block.buf[cbt:nbt]
        return buf

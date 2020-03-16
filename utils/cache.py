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
from multiprocessing import Manager
import numpy as np


class SharedMemoryFrameCache(object):
    """
    shared_memory.ShareMemory wrapper for r/w shared memory as a original List
    """

    def __init__(self, manager: SharedMemoryManager, cache_size, unit, shape) -> None:
        self.unit = unit
        print(self.unit)
        self.shape = shape
        self.cache_size = cache_size
        self.total_bytes = self.unit * self.cache_size
        self.cache_block = manager.SharedMemory(size=self.total_bytes)

    def __getitem__(self, index):
        """
        cache[index]
        :param index:
        :return:
        """
        buf = self.get_buf(index)
        return np.ndarray(self.shape, dtype=np.uint8, buffer=buf)

    def __setitem__(self, index, frame):
        """
        cache[index] = frame
        :param index:
        :param frame:
        :return:
        """
        buf = self.get_buf(index)
        buf_frame = np.ndarray(self.shape, dtype=np.uint8, buffer=buf)
        buf_frame[:, :, :] = frame[:, :, :]

    def is_closed(self):
        return self.cache_block.close()

    def get_buf(self, index):
        """
        decode bottom buffer into numpy object and r/w buffer via numpy slices operation
        :param index:
        :return:
        """
        cbt = self.unit * (index % self.cache_size)
        nbt = self.unit * ((index + 1) % self.cache_size)
        if nbt == 0:
            nbt = self.total_bytes
        # print((index + 1) % self.cache_size)
        buf = self.cache_block.buf[cbt:nbt]
        return buf

    def close(self):
        self.cache_block.close()
        self.cache_block.unlink()


class ListCache(object):
    """
    Manager().list() wrapper
    """

    def __init__(self, manager: Manager, cache_size, template) -> None:
        self.proxy = manager.list([None] * cache_size)
        self.cache_size = cache_size
        self.template = template

    def __getitem__(self, index):
        return self.proxy[index % self.cache_size]

    def __setitem__(self, index, frame):
        self.proxy[index % self.cache_size] = frame

    def append(self, frame):
        return self.proxy.append(frame)

    def pop(self):
        if not len(self.proxy):
            print('List Cache: Empty Frame.Return Template Instead.')
            return self.template
        return self.proxy.pop()

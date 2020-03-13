#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test_share_memory.py
@time: 3/13/20 11:26 AM
@version 1.0
@desc:
"""
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Pool, Manager, cpu_count, Queue
from utils.cache import FrameCache
import cv2
import time
import imutils


def blur(cache: FrameCache, type):
    index = 0
    while True:
        frame = cache[index]
        frame = cv2.GaussianBlur(frame, (3, 3), sigmaX=0, sigmaY=0)
        start = time.time()
        cache[index] = frame
        end = time.time()
        print(f'Write Into {type} cache: [{1 / (end - start)}]/FPS')
        index += 1


def receive(cache: FrameCache, type):
    index = 0
    while True:
        start = time.time()
        frame = cache[index]
        end = time.time()
        print(f'Get from {type} Cache: [{1 / (end - start)}]/FPS')
        frame = imutils.resize(frame, width=1000)
        # cv2.imshow('Blur', frame)
        # cv2.waitKey(1)
        index += 1


if __name__ == '__main__':
    smm = SharedMemoryManager()
    manager = Manager()
    smm.start()
    frame = cv2.imread("/Users/luvletteru/Documents/GitHub/DolphinDetection/data/test/0312/1.jpg")
    print(frame.dtype)
    cache_size = 300
    s1 = FrameCache(smm, cache_size, frame.nbytes, frame.shape)
    q1 = Manager().list([None] * cache_size)
    for i in range(cache_size):
        start = time.time()
        s1[i] = frame
        end = time.time()
        print(f'Write Into {type} cache: [{1 / (end - start)}]/FPS')
        # q1[i] = frame
    with Pool(cpu_count() - 1) as pool:
        r1 = pool.apply_async(blur, args=(s1, 'Shared Memory',))
        r2 = pool.apply_async(receive, args=(s1, 'Shared Memory',))
        r1.get()
        r2.get()
        # r3 = pool.apply_async(blur, args=(q1, 'List',))
        # r4 = pool.apply_async(receive, args=(q1, 'List',))
        # r3.get()
        # r4.get()

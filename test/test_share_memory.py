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
from multiprocessing import Pool, Manager, cpu_count, Queue, Value,RLock
from utils.cache import SharedMemoryFrameCache
import cv2
import time
import imutils
import numpy as np


def blur(cache: SharedMemoryFrameCache, global_index, type):
    cap = cv2.VideoCapture("/Users/luvletteru/Documents/GitHub/DolphinDetection/data/candidates/0325_cvam_21.mp4")
    grabbed, frame = cap.read()
    while grabbed:
        frame = cv2.GaussianBlur(frame, (3, 3), sigmaX=0, sigmaY=0)
        start = time.time()
        index = global_index.get()
        cache[index] = frame
        end = time.time()
        print(f'Write Into {type} cache: [{1 / (end - start)}]/FPS')
        global_index.set(index + 1)
        grabbed, frame = cap.read()


def receive(cache: SharedMemoryFrameCache, global_index, type):
    index = 0
    time.sleep(2)
    while True:
        start = time.time()
        index = global_index.get() - 5
        cache.lock_cache(index, index + 5)
        frame = cache[index]
        time.sleep(2)
        end = time.time()
        print(f'Get from {type} Cache: [{1 / (end - start)}]/FPS')
        frame = imutils.resize(frame, width=1080)
        cv2.imshow('Blur', frame)
        cv2.waitKey(1)
        cache.release()


def test_share_memory_rw():
    cache_size = 10
    init_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    global_index = Manager().Value('i', 1)
    smm = SharedMemoryManager()
    smm.start()
    s1 = SharedMemoryFrameCache(smm, cache_size, init_frame.nbytes, init_frame.shape)
    with Pool(2) as pool:
        r1 = pool.apply_async(blur, args=(s1, global_index, 'Shared Memory',))
        r2 = pool.apply_async(receive, args=(s1, global_index, 'Shared Memory',))
        r1.get()
        r2.get()
        # r3 = pool.apply_async(blur, args=(q1, 'List',))
        # r4 = pool.apply_async(receive, args=(q1, 'List',))
        # r3.get()
        # r4.get()


# def test_share_list():


if __name__ == '__main__':
    test_share_memory_rw()

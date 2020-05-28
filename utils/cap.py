#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: cat.py
@time: 4/4/20 7:09 PM
@version 1.0
@desc:
"""
import os

import vlc
import ctypes
import time
import cv2
import numpy as np
from PIL import Image
import queue

"""
a simple encapsulation of VLC module for solving packets loss and decoding errors 
based 4K RTSP streams.
"""

current_index = -1
VIDEOWIDTH = -1
VIDEOHEIGHT = -1
RUN = True
q = queue.Queue()
size = 0
buf = None
buf_p = None
m = None
mp = None

VideoLockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))


@VideoLockCb
def _lockcb(opaque, planes):
    planes[0] = buf_p


@vlc.CallbackDecorators.VideoDisplayCb
def put_queue(opaque, picture):
    """
    receive buffer data from bottom layer, convert raw image into a BGR image
    :param opaque:
    :param picture:
    :return:
    """
    global current_index, q
    current_index += 1
    img = Image.frombuffer("RGBA", (VIDEOWIDTH, VIDEOHEIGHT), buf, "raw", "RGBA", 0, 1)
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    q.put(np.array(img)[:, :, :3])
    # q.put(img)

    #img = Image.frombuffer("RGB", (VIDEOWIDTH, VIDEOHEIGHT), buf, "raw", "BGR", 0, 1)
    #q.put(np.array(img))

    # print(os.getpid())

    #img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # q.put(img)

    # q.put(np.array(img))
    # img = np.ndarray((VIDEOHEIGHT, VIDEOWIDTH, 3), dtype=np.uint8, buffer=buf)
    # q.put(numpy.array(img)[:, :, :3])
    # cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
    # cv2.imshow('image', img)
    # cv2.waitKey(1)


def read():
    """
    read a frame from queue
    :return:
    """
    global q
    try:
        return True, q.get(timeout=5)
    except:
        release()
        return False, None


def release():
    """
    release VLC media player instance
    :return:
    """
    global RUN
    RUN = False


def run(src, shape):
    """
    init VLC media player
    :param src:
    :param shape:
    :return:
    """
    global VIDEOHEIGHT, VIDEOWIDTH, buf, buf_p, RUN
    VIDEOWIDTH = shape[1]
    VIDEOHEIGHT = shape[0]
    size = VIDEOHEIGHT * VIDEOWIDTH * 4
    buf = (ctypes.c_ubyte * size)()
    buf_p = ctypes.cast(buf, ctypes.c_void_p)
    vlcInstance = vlc.Instance()
    m = vlcInstance.media_new(src)
    mp = vlc.libvlc_media_player_new_from_media(m)
    vlc.libvlc_video_set_callbacks(mp, _lockcb, None, put_queue, None)
    mp.video_set_format("RGBA", VIDEOWIDTH, VIDEOHEIGHT, VIDEOWIDTH * 4)
    mp.play()

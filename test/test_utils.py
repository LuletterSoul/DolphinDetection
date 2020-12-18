#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test_utils.py
@time: 2019/11/15 22:30
@version 1.0
@desc:
"""
from utils import *
from config import *
import interface
import cv2
import imutils


def test_crop_by_roi():
    cfg = interface.load_video_config(VIDEO_CONFIG_DIR / 'video.json')[0]
    frame_path = list(FRAME_SAVE_DIR.glob('*'))[0]
    frame = cv2.imread(str(frame_path))
    frame = imutils.resize(frame, width=cfg.resize['width'])
    if frame is None:
        return
    frame = crop_by_roi(frame, cfg.roi)
    print(frame.shape)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)


def test_label():
    labeler = Labeler(LABEL_IMAGE_PATH, LABEL_SAVE_PATH, LABEL_SAVE_PATH, LABEL_TARGET_PATH)
    labeler.label()

if __name__ == "__main__":
   img =  cv2.imread('./static/test.png')
   logo = cv2.imread('./static/logo.png',-1)

   local_time = datetime.now()
   b, g, r, a = cv2.split(logo)
   template = cv2.merge((a, a, a))
   fmt_local_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
   im_ret =  paste_logo(img,template, fmt_local_time,1.6,(1300,95),4)
   cv2.imshow('Logo',im_ret)
   cv2.waitKey(0)

    # test_crop_by_roi()

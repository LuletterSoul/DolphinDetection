#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: log.py
@time: 2019/11/16 21:09
@version 1.0
@desc:
"""

import json
import os
from pathlib import Path
from utils.crop import crop_by_se
from utils.log import logger

import cv2
import numpy as np


class Labeler(object):

    def __init__(self, img_path: Path, save_path: Path, json_path: Path, target_path: Path, json_name='samples.json',
                 template_name='samples_template.json', scale=0.5,
                 suffix='*.png') -> None:
        """

        :param img_path:
        :param save_path:Labels will be saved here
        :param json_path: Get json template from this dir
        :param target_path: Cropped target will be saved here
        :param template_name: Default template json name
        :param json_name:
        :param scale:
        :param suffix:
        """
        super().__init__()
        self.img_path = img_path
        self.save_path = save_path
        self.target_path = target_path
        self.scale = scale
        self.suffix = suffix
        self.start = None
        self.end = None
        self.center = None
        self.crop_label_img = None
        self.json_path = json_path
        self.template_name = template_name
        self.json_name = json_name

    def label(self):
        all_path = list(self.img_path.glob(self.suffix))
        if not len(all_path):
            logger.info(
                "Empty dir:[{}] or cannot match the suffix:[{}] with filenames.".format(self.img_path, self.suffix))
        self.json_path.mkdir(exist_ok=True, parents=True)
        template_json = self.json_path / self.template_name
        if not template_json.exists():
            raise Exception('Template not exists')
        # template_json, _ = buildWritableConfigFile(template_json)
        SELECT_ROI_WINDOW_NAME = 'Select Image ROI'
        SELECT_CENTER_WINDOW_NAME = 'Select ROI Center'
        json_file = open(str(self.json_path / self.json_name), mode='w')
        # sample_json, _ = buildWritableConfigFile()
        sample_jsons = {}
        for idx, p in enumerate(all_path):
            js = json.load(open(template_json))
            base_name = os.path.basename(str(p))
            img = cv2.imread(str(p))
            if img is None:
                logger.info("Load empty img error: [{}]".format(p))
            # resize img for labeling convinient
            resized = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
            # crop ROI
            self.crop_roi(SELECT_ROI_WINDOW_NAME, resized)
            # confirm selected ROI
            resized = self.crop_label_img
            # select center
            self.select_center(SELECT_CENTER_WINDOW_NAME, resized)
            cv2.destroyAllWindows()
            # back to original size
            self.start = (self.start / self.scale).astype(np.int32)
            self.end = (self.end / self.scale).astype(np.int32)
            js['start'] = self.start.tolist()
            js['end'] = self.end.tolist()
            js['center'] = self.center.tolist()
            sample_jsons[base_name] = js
            # get cropped img
            cropped = crop_by_se(img, self.start, self.end)
            cv2.imwrite(str(self.target_path / base_name), cropped)
        json_file.write(json.dumps(sample_jsons, indent=4))
        json_file.close()

    def select_center(self, window_name, img):
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_select_center,
                             EventParam(img, window_name, self))
        while True:
            cv2.imshow(window_name, img)
            if cv2.waitKey(0) & 0xFF == ord('y'):
                logger.info('Select Center')
                break
            else:
                logger.info('Re-select Center')

    def crop_roi(self, window_name, img):
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_select_roi,
                             EventParam(img, window_name, self))
        while True:
            cv2.imshow(window_name, img)
            if cv2.waitKey(0) & 0xFF == ord('y'):
                logger.info('Crop roi done')
                break
            else:
                logger.info('Re-select roi')


class EventParam(object):

    def __init__(self, img, winddow_name, labeler: Labeler) -> None:
        super().__init__()
        self.img = img
        self.window_name = winddow_name
        self.labeler = labeler


def on_select_roi(event, x, y, flags, param: EventParam):
    # global point1, point2
    # global ensure
    # global pressure
    img = param.img
    target = param.labeler
    window_name = param.window_name
    # index = param[1]
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        target.start = np.array([x, y])
        cv2.circle(img2, (x, y), 1, (0, 255, 0), 1)
        cv2.imshow(window_name, img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, (target.start[0], target.start[1]), (x, y), (255, 0, 0), 1)
        cv2.imshow(window_name, img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        target.end = np.array([x, y])
        cv2.rectangle(img2, (target.start[0], target.start[1]), (target.end[0], target.end[1]), (0, 0, 255), 1)
        logger.info('Confirm roi: start :{}, end:{} y/n?'.format(target.start, target.end))
        target.crop_label_img = img2
        cv2.imshow(window_name, img2)


def on_select_center(event, x, y, flags, param: EventParam):
    # global point1, point2
    # global ensure
    # global pressure
    img = param.img
    target = param.labeler
    window_name = param.window_name
    # index = param[1]
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        target.center = np.array([x, y])
        cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow(window_name, img2)
        logger.info('Confirm roi center: {}'.format(target.center))

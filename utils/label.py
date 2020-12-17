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
import time
from PIL import Image, ImageDraw, ImageFont


class Labeler(object):

    def __init__(self, img_path: Path, save_path: Path, json_path: Path, target_path: Path, json_name='samples.json',
                 template_name='samples_template.json', scale=0.5,
                 suffix='*.png', use_template=False) -> None:
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
        self.use_template = use_template

    def label(self):
        all_path = list(self.img_path.glob(self.suffix))
        if not len(all_path):
            logger.info(
                "Empty dir:[{}] or cannot match the suffix:[{}] with filenames.".format(self.img_path, self.suffix))
        self.json_path.mkdir(exist_ok=True, parents=True)
        if self.use_template:
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
            # js = json.load(open(template_json))
            js = {} if not self.use_template else json.load(open(template_json))
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
                if self.center is None:
                    logger.info('Pleas select a center')
                    continue
                else:
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
                if self.start is None or self.end is None:
                    logger.info('Pleas select a valid roi')
                else:
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


def add_text_logo(image, text, logo, text_params, logo_params):
    """
    params:
        image: source image.
        text: the text put on the image.
        logo: teh logo image.
        text_params: the params of text pasting.
        logo_params: the params of logo pasting.
    return:
        img_ret: the image after pasting text and logo.
    note:
        img_ret should be saved as PNG format, JPEG format maybe cause fault.
    """
    # add text
    img = image.convert("RGBA")
    img_x, img_y = img.size
    text_overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    img_draw = ImageDraw.Draw(text_overlay)
    
    text_size_x, text_size_y = img_draw.textsize(text, font=text_params["font"])
    text_location = (img_x - text_size_x - text_params["location"][0], text_params["location"][1])
    img_draw.text(text_location, text, font=text_params["font"], fill=text_params["color"])
    img_draw.text((text_location[0] + text_params["bold_offset"], text_location[1] + text_params["bold_offset"]), text, font=text_params["font"], fill=text_params["color"])
    img_draw.text((text_location[0] - text_params["bold_offset"], text_location[1] - text_params["bold_offset"]), text, font=text_params["font"], fill=text_params["color"])
    img_draw.text((text_location[0] - text_params["bold_offset"], text_location[1] + text_params["bold_offset"]), text, font=text_params["font"], fill=text_params["color"])
    img_draw.text((text_location[0] + text_params["bold_offset"], text_location[1] - text_params["bold_offset"]), text, font=text_params["font"], fill=text_params["color"])
    img_ret = Image.alpha_composite(img, text_overlay)

    # add logo
    logo = logo.convert("RGBA")
    logo_x, logo_y = logo.size
    scale = logo_params["reduce_ratio"]
    logo_scale = max(img_x / (scale * logo_x), img_y / (scale * logo_y))
    new_size = (int(logo_x * logo_scale), int(logo_y * logo_scale))
    logo = logo.resize(new_size)
    _, _, _, logo_mask = logo.split()

    img_ret.paste(logo, logo_params["location"], logo_mask)
    # img_ret.show()

    return img_ret


#text_params = dict({
#    "font": ImageFont.truetype("msyh.ttc", 50, encoding="uft-8"),
#    "color": (255, 255, 255),
#    "location": (40, 40),
#    "bold_offset": 1
#})

#logo_params = dict({
#    "location": (0, 0),
#    "reduce_ratio": 8
#})


#text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#text = "下关码头 / " + text

#if __name__ == "__main__":
#    im = Image.open("1.png")
#    logo = Image.open("eco_eye_logo.png")
#    im_ret = add_text_logo(im, text, logo, text_params=text_params, logo_params=logo_params)
#    im_ret.save("ret_2.png")

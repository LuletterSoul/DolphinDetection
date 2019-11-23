# -*- coding: utf-8 -*-

from detection.thresh import adaptive_thresh
from config import *
import interface as I 
import os.path as osp
import glob
import re
import cv2
import numpy as np


blur_ksize = 5
canny_lth = 75
canny_hth = 125
kernel_size = np.ones((3, 3), np.uint8)
temperature = 300


def evaluate_label(path):
    img_paths = glob.glob(osp.join(path, '*.png'))
    label_keys, label_vals = I.load_label_config(LABEL_SAVE_PATH / 'samples.json')

    idx, num= -1, 0
    for img_path in img_paths:
        idx += 1
        origin = cv2.imread(img_path)
        gray = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        ada_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 20)
        edges = cv2.Canny(ada_thresh, canny_lth, canny_hth)
        dilation = cv2.dilate(edges, kernel_size)
        # cv2.imshow("dila",dilation)
        # cv2.waitKey(0)
        x, y = label_vals[idx].center[0], label_vals[idx].center[1]
        flag = 0
        for i in range(temperature):
            for j in range(temperature):
                if dilation[x - int(temperature / 2) + i, y - int(temperature / 2) + j] == 255:
                    flag += 1
        if flag > 0:
            num += 1

    origin = cv2.imread(img_paths[0])
    size = origin.shape
    all_area = size[0] * size[1]
    estimate_area = temperature * temperature
    estimate_ratio = estimate_area / all_area

    return (float(num / img_paths.__len__())), estimate_ratio


if __name__ == '__main__':
    accuracy, ratio = evaluate_label(str(LABEL_IMAGE_PATH))
    print("Using additional area ratio: {:.2} %".format(ratio * 100))
    print("-->    accuracy: {:.2}".format(accuracy))
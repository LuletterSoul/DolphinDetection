# -*- coding: utf-8 -*-

import cv2
import numpy as np
from config import VideoConfig
from utils import crop_by_roi

path = '/home/cys/Codes/DolphinDetection'

blur_ksize = 5
canny_lth = 75
canny_hth = 125
kernel_size = np.ones((3, 3), np.uint8)


def adaptive_thresh(frame, cfg=None):
    # img = img[370:1080, 0:1980]
    # frame = crop_by_roi(frame, cfg.roi)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    th_atm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 20)
    # ret_otsu, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(th_atm, canny_lth, canny_hth)
    dilation = cv2.dilate(edges, kernel_size)
    return dilation

# def main():
#     init_path = osp.join(path, 'Demo/picts/test_2.jpg')
#     save_path = osp.join(path, 'Demo/picts/2.jpg')
#
#     img = cv2.imread(init_path)
#     result = adaptive_thresh(img)
#     # cv2.imwrite(save_path, dilation)
#
#
# if __name__ == '__main__':
#     main()
#     print("Done!")

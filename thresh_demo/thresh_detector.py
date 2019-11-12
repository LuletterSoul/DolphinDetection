# -*- coding: utf-8 -*-

import cv2
import os.path as osp
import numpy as np 

path = '/home/cys/Codes/DolphinDetection'

blur_ksize = 5
canny_lth = 75
canny_hth = 125
kernel_size = np.ones((3, 3), np.uint8)


def process_img(img):
    img = img[370:1080, 0:1980]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    th_atm = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 20)
    # ret_otsu, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(th_atm, canny_lth, canny_hth)
    return edges

def main():
    init_path = osp.join(path, 'Demo/picts/test_2.jpg')
    save_path = osp.join(path, 'Demo/picts/2.jpg')

    img = cv2.imread(init_path)
    result = process_img(img)
    dilation = cv2.dilate(result, kernel_size)
    cv2.imwrite(save_path, dilation)

if __name__ == '__main__':
    main()
    print("Done!")
    
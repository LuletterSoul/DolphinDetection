import cv2 as cv
import numpy as np
import os
import time


# enhance module
def resize(img, scale):
    if scale == 1:
        return img
    h, w, c = img.shape
    return cv.resize(img, (w * scale, h * scale), interpolation=cv.INTER_CUBIC)


def smooth(img):
    # # 均值滤波
    # blur_img = cv.blur(img, (5, 5))

    # 中值滤波
    # blur_img = cv.medianBlur(img, 5)

    # 高斯滤波
    # blur_img = cv.GaussianBlur(img, (5, 5), 0)

    # 双边滤波
    blur_img = cv.bilateralFilter(img, 9, 75, 75)
    return blur_img


def sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sharp_img = cv.filter2D(img, -1, kernel=kernel)
    return sharp_img


def bright(img):
    img_bright = cv.convertScaleAbs(img, alpha=1.5, beta=0)
    return img_bright


def equalize(img):
    img_norm = cv.normalize(img, dst=None, alpha=350, beta=10, norm_type=cv.NORM_MINMAX)
    return img_norm


def enhance_img(img, scale):
    # img = blur(img)
    # img = sharp(img)

    start = time.time()
    hr_img = resize(img, scale=scale)
    smooth_img = smooth(hr_img)
    sharp_img = sharp(smooth_img)
    # sharp_img = sharp(sharp_img)
    # equalize_img = equalize(sharp_img)
    # print(f'used time={time.time() - start}')

    # sharp_img = blur(sharp_img)
    # sharp_img = sharp(sharp_img)
    return sharp_img


def crop_and_up_sample_image(img, center, scale, dst_shape, task_cnt):
    """

    :param img: frame，numpy格式
    :param center: 框中心 [width, height]
    :param scale:  上采样倍数，后续会根据scale与dst_shape计算裁剪多大的区域
    :param dst_shape: 目标大小[width, height]
    :return:
    """
    img_shape = img.shape  # [height, width, channel]
    crop_shape = [int(dst_shape[0] / scale), int(dst_shape[1] / scale)]  # [width, height]
    left_top = [max(0, center[0] - int(crop_shape[0] / 2)),
                max(0, center[1] - int(crop_shape[1] / 2))]  # [width, height]
    if left_top[0] + crop_shape[0] >= img_shape[1]:
        left_top[0] = img_shape[1] - crop_shape[0]

    if left_top[1] + crop_shape[1] >= img_shape[0]:
        left_top[1] = img_shape[0] - crop_shape[1]

    croped_img = img[left_top[1]:left_top[1] + crop_shape[1], left_top[0]:left_top[0] + crop_shape[0], :]
    #print(f'Task: {task_cnt}, left_top={left_top}, crop_shape={crop_shape}, cropped img.shape={croped_img.shape}')
    upsampled_img = enhance_img(croped_img, scale)
    return upsampled_img


def test1():
    video_path = '../data/videos/original_video/12-17-13-25-27-0.mp4'
    video_capture = cv.VideoCapture(video_path)
    ret, frame = video_capture.read()
    while ret:
        upsampled_frame = crop_and_up_sample_image(frame, center=[800, 100], scale=4, dst_shape=[1920, 1080])
        cv.imshow('upsampled image', upsampled_frame)
        cv.waitKey(25)
        ret, frame = video_capture.read()


def main():
    test1()


if __name__ == '__main__':
    main()

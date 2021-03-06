# -*- coding: utf-8 -*-

import cv2
import os
from utils.log import logger
from pathlib import Path


def process_video(inpath: Path, outpath: Path, frame_num=10):
    outpath.mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(str(inpath))
    # num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not cap.isOpened():
        logger.info("Open failed.")

    cnt = 0
    count = 0
    samples = []
    while 1:
        ret, frame = cap.read()
        cnt += 1
        if cnt % frame_num == 0:
            count += 1
            cv2.imwrite(os.path.join(outpath, str(count) + ".jpg"), frame)
            samples.append(frame)
        if not ret:
            break
    return samples


if __name__ == '__main__':

    inpath = "TS/000.ts"  # 输入的视频文件路径
    outpath = "Pic"  # 输出的样本帧路径
    frame_num = 10  # 帧采样率（就是每隔多少帧保存一幅图片）

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    process_video(inpath, outpath, frame_num)

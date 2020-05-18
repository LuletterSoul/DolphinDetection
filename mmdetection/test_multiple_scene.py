from pathlib import Path

import cv2

from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import imutils

config_file = './configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py'
checkpoint_file = './work_dirs/cascade_rcnn_r50_caffe_fpn_1x_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

test_path = "/home/jt1/data/test_data/0515"
idx = 0
metrics = []


def cal_precision(TP, FP, TN, FN):
    return TP / (TP + FP)


def cal_recall(TP, FP, TN, FN):
    return TP / (TP + FN)


def cal_specificity(TP, FP, TN, FN):
    return TN / (TN + FP)


def cal_fpn(TP, FP, TN, FN):
    return FP / (TN + FP)


def cal_fnn(TP, FP, TN, FN):
    return FN / (TP + FN)


def cal_fl(precision, recall):
    return (2 * precision * recall) / (precision + recall)


for c in os.listdir(test_path):
    positive_path = os.path.join(test_path, c, 'Positive')
    negative_path = os.path.join(test_path, c, 'Negative')
    pos_sample_nums = len(os.listdir(positive_path))
    neg_sample_nums = len(os.listdir(negative_path))
    print(f'Processing Scene [{c}]')
    print(f'Positive Num: {pos_sample_nums}')
    print(f'Negative Num: {neg_sample_nums}')
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for p in os.listdir(positive_path):
        video = mmcv.VideoReader(os.path.join(positive_path, p))
        tp_set_flag = False
        fn_set_flag = False
        print(f'Processing video: {c}/{p}')
        for frame in video:
            if idx % 3 == 0:
                frame = imutils.resize(frame, width=1000)
                result = inference_detector(model, frame)
                model.show_result(frame, result, wait_time=1, show=True, font_scale=1, thickness=2,
                                  score_thr=0.8)
                if len(result[0]) and result[0][0][4] > 0.8:
                    if not tp_set_flag:
                        TP += 1
                        tp_set_flag = True
            idx += 1
        if not tp_set_flag:
            FN += 1

    for n in os.listdir(negative_path):
        video = mmcv.VideoReader(os.path.join(negative_path, n))
        fp_set_flag = False
        tn_set_flag = False
        print(f'Processing video: {c}/{n}')
        for frame in video:
            if idx % 3 == 0:
                frame = imutils.resize(frame, width=1000)
                result = inference_detector(model, frame)
                model.show_result(frame, result, wait_time=1, show=True, font_scale=1, thickness=2,
                                  score_thr=0.8)
                if len(result[0]) and result[0][0][4] > 0.8:
                    if not fp_set_flag:
                        FP += 1
                        fp_set_flag = True
            idx += 1
        if not fp_set_flag:
            TN += 1

    precision = cal_precision(TP, FP, TN, FN)
    recall = cal_recall(TP, FP, TN, FN)
    specificity = cal_specificity(TP, FP, TN, FN)
    fpn = cal_fpn(TP, FP, TN, FN)
    fnn = cal_fnn(TP, FP, TN, FN)
    f1 = cal_fl(precision, recall)
    metrics.append([precision, recall, specificity, fpn, fnn, f1])
    print(
        f'Scene [{c}]: Precision: [{round(precision, 2)}], Recall: [{round(recall, 2)}],'
        f' Specificity: [{round(specificity, 2)}], FPN: [{round(fpn, 2)}], FNN: [{round(fnn, 2)}], f1: [{round(f1, 2)}]')

avg = []
num_scenes = len(os.listdir(test_path))
for n in range(6):
    m1 = 0
    for m in range(num_scenes):
        m1 += metrics[m][n]
    avg.append(m1 / num_scenes)

print(
    f'Avg: Precision: [{round(avg[0], 2)}], Recall: [{round(avg[1], 2)}],'
    f' Specificity: [{round(avg[2], 2)}], FPN: [{round(avg[3], 2)}], FNN: [{round(avg[4], 2)}], f1: [{round(avg[5], 2)}]')

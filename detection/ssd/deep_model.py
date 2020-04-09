'''
   a deep classify_model class
   by Sun Hongwei
'''

import os
import sys

# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
import torch.nn as nn
import torch
from torch.autograd import Variable
from .data import BaseTransform, VOC_CLASSES as labelmap
from .ssd import build_ssd
import numpy as np
import time
import cv2
# from .logger import make_logger
from utils import logger as Logger

from pathlib import Path


class SSDDetector(nn.Module):
    '''
     cowfish deep classify_model
     input frames batch
     output cowfish locations and confidences
    '''

    def __init__(self, size=300, conf=0.5, logger=Logger, model_path=None,
                 device_id='3'):
        super(SSDDetector, self).__init__()

        # net size
        self.size = size
        self.device_id = device_id
        self.device = None
        if self.device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        if torch.cuda.is_available():
            # if self.device_id is not None:
            # self.device = torch.device("cuda:" + str(self.device_id))
            self.device = torch.device("cuda")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # self.net = torch.load(str(self.model_path))
        else:
            self.device = torch.device("cpu")
        self.net = build_ssd('test', self.device, self.size, 2)
        # set det confidence thresh
        self.conf = conf
        # classify_model path
        self.model_path = model_path
        # define image process
        self.transform = BaseTransform(self.net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))
        # set logger
        self.logger = logger
        # set gpu config

    def forward(self, x):
        '''
        :param x: set ：[frame1,frame2,frame3,...] frame type is numpy
        :return:
        '''
        if len(x) == 0:
            self.logger.info("x is zero frame")

        height, width = x[0].shape[:2]
        if len(x) == 1:
            x = torch.from_numpy(self.transform(x[0])[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0)).cuda()
        else:
            frames_set = list()
            for frame in x:
                frames_set.append(torch.from_numpy(self.transform(frame)[0]).permute(2, 0, 1))

            x = torch.stack(frames_set, 0)  # group a batch
            x = Variable(x).cuda()

        # t0 = time.time()
        y = self.net(x)  # forward pass output = torch.zeros(num, self.num_classes, self.top_k, 5)
        # t1 = time.time()

        # self.logger.info('classify_model detect timer: %.4f sec.' % (t1 - t0))
        detections = y.data

        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])

        frames_pts = list()
        for i in range(detections.size(0)):  # foreach batch
            j = 0
            pts = np.zeros((0, 5))
            while detections[i, 1, j, 0] >= self.conf:
                pt = (detections[i, 1, j, 1:] * scale).cpu().numpy()
                # print(pt)
                # print(';;;;')
                # print('--------', detections[i, 1, j, 0])
                pt = np.append(pt, detections[i, 1, j, 0].item())
                pts = np.row_stack((pts, pt))
                j += 1
            frames_pts.append(pts[np.newaxis, :, :])

        dets = np.concatenate(frames_pts)
        # t2 = time.time()
        # self.logger.info(' after processing  time: %.4f sec.' % (t2 - t1))
        # self.logger.info(' all  time: %.4f sec.' % (t2 - t0))

        return dets

    def load_model(self, model_path=None):
        '''
        load classify_model pars
        :return:
        '''
        if model_path is not None:
            self.model_path = model_path
            self.net.load_state_dict(torch.load(self.model_path))

    def eval(self):
        '''
        :return:
        '''
        # eval mode
        self.net.eval()

    def draw_dets(self, frame, dets):
        '''
        :param frame:
        :param dets: [[x1,y1,x2,y2,p],[],...,]
        :return:
        '''

        det_num = dets.shape[0]
        for i in range(det_num):
            pt = dets[i]
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          (0, 0, 255), 2)

            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, str(round(pt[4], 2)), (int(pt[2]), int(pt[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def run(self):
        # if self.device_id is not None:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = self.device_id

        if self.model_path is not None:
            if not self.model_path.exists():
                raise Exception(f'Model init failed: classify_model not exist at [{str(self.model_path)}].')
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(str(self.model_path)))
        else:
            self.net.load_state_dict(torch.load(str(self.model_path), map_location=torch.device('cpu')))
        self.net = self.net.to(self.device)
        self.net.eval()
        print(self.net)
        print(self.device)


def init_ssd(model_path, device_id):
    # SSD_MODEL = SSDDetector(model_path=model_path, device_id=device_id)
    # SSD_MODEL.run()
    pass

# SSD_MODEL = SSDDetector(model_path=Path("model/0220-ssd.pth"), device_id=3)
# SSD_MODEL.run()
#
# if __name__ == "__main__":
#
#     if torch.cuda.is_available():
#         if opt.DEVICE:
#             torch.set_default_tensor_type('torch.cuda.FloatTensor')
#         if not opt.DEVICE:
#             print("WARNING: It looks like you have a CUDA device, but aren't " +
#                   "using CUDA.\nRun with --cuda for optimal training speed.")
#             torch.set_default_tensor_type('torch.FloatTensor')
#     else:
#         torch.set_default_tensor_type('torch.FloatTensor')
#
#     test_file_txt = "/data/shw/dolphinDetect/JT001/ImageSets/test.txt"
#     img_path = "/data/shw/dolphinDetect/JT001/JPGImages"
#     model_path = "/home/shw/code/ZhiXing/checkpoint/Exp-2/JT001.pth"
#     test_file = open(test_file_txt, "r")
#     lines = test_file.readlines()
#     test_img = [line.strip() for line in lines]
#     # logger = make_logger("project", opt.OUTPUT_DIR, 'log')
#     model = SSDDetector(logger=logger, model_path=model_path)
#     model.load_model()
#
#     for m in test_img[:1]:
#         print("==")
#         frame = cv2.imread(os.path.join(img_path, m + ".jpg"))
#         t0 = time.time()
#         dets = model([frame, frame, frame, frame, frame, frame, frame, frame])
#         t1 = time.time()
#         print(' detect  time: %.4f sec.' % (t1 - t0))
#         # print(dets)

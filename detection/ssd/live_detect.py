from __future__ import print_function
import torch
import cv2
import os
import time
from .logger import make_logger
import sys
from os import path
from .deep_model import SSDDetector

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from .data import BaseTransform
import yaml


def get_config(config_file, logger=None):
    '''
    :param config_file:
    :param logger:
    :return:
    '''
    if logger is not None:
        logger.info('Read config from: %s', config_file)
    with open(config_file, 'r') as f:
        cfg = yaml.load(f)
    if logger is not None:
        logger.info('Config: {}'.format(cfg))
    return cfg


def detect_frame(url):
    '''
    :return:
    '''
    cap = cv2.VideoCapture(url)
    while cap.isOpened():

        ret, frame = cap.read()

        if ret is not False:
            t = str(round(time.time(), 4))

            dets = net([frame])
            print(dets)
            if len(dets) is not 0 and len(dets[0]) is not 0:
                # frame = net.draw_dets(frame, dets[0])

                if not os.path.exists(os.path.join(opt['COWFISH_SAVE_PATH'], day)):
                    os.mkdir(os.path.join(opt['COWFISH_SAVE_PATH'], day))

                cv2.imwrite(os.path.join(opt['COWFISH_SAVE_PATH'], day, t + ".jpg"), frame)
        else:
            break


def detect_img():
    '''
    :return:
    '''
    test_path = "/data/shw/cowfish/jtdome/test"
    imgs = os.listdir(test_path)

    for img in imgs:
        img_path = os.path.join(test_path, img)
        img_data = cv2.imread(img_path)

        if img_data is not None:
            t = str(round(time.time(), 4))

            dets = net([img_data])
            print(dets)
            if len(dets) is not 0 and len(dets[0]) is not 0:
                img_data = net.draw_dets(img_data, dets[0])

                if not os.path.exists(os.path.join(opt['COWFISH_SAVE_PATH'], day)):
                    os.mkdir(os.path.join(opt['COWFISH_SAVE_PATH'], day))

                cv2.imwrite(os.path.join(opt['COWFISH_SAVE_PATH'], day, t + ".jpg"), img_data)
        else:
            break


if __name__ == '__main__':

    opt = get_config(config_file='./configs/sample.yaml')

    if opt['DEVICE']:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt['DEVICE_ID']
        num_gpus = len(opt['DEVICE_ID'].split(','))

    if torch.cuda.is_available():
        if opt['DEVICE']:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not opt['DEVICE']:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("build classify_model")
    net = SSDDetector()  # initialize SSD
    print("load classify_model weight")
    net.load_model("/home/shw/code/ZhiXing/checkpoint/Exp-5/JT002.pth")
    net.eval()
    print("loaded")
    ti = time.time()
    date_ary = time.localtime(ti)
    day = str(time.strftime("%Y%m%d", date_ary))
    logger = make_logger("project", os.path.join(opt['COWFISH_SAVE_PATH'], day), 'log')
    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    # detect_img()
    detect_frame("rtsp://222.190.243.243/bp/jtdome")
    # detect_frame("/data/lxd/finished_shw_1/12-19-11-39/5/original-streams_t/12-19-13-41-37-407.mp4")
    # flag = False
    # camera = Camera(rtsp_url="/data/lxd/finished_shw_1/12-19-11-39/5/original-streams_t/12-19-17-20-29-1023.mp4")
    # frame_queue = queue.Queue()
    # run()

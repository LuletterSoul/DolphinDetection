import sys
import argparse

sys.path.append('../')
from interface import *
from detection import *
import numpy as np
from pathlib import Path
import cv2 as cv


class VideoDetector(object):
    def __init__(self, cfg_path, output_path):
        self.cfg_path = cfg_path
        self.cfg = self.load_yaml_config()
        self.detect_params = self.set_detect_params()
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def load_yaml_config(self):
        with open(self.cfg_path) as f:
            cfg_obj = yaml.safe_load(f)
            cfg = VideoConfig.from_yaml(cfg_obj)
        return cfg

    def set_detect_params(self):
        x_num = self.cfg.routine['col']
        y_num = self.cfg.routine['row']
        empty = np.zeros(self.cfg.shape).astype(np.uint8)
        frame, _ = preprocess(empty, self.cfg)
        x_step = int(frame.shape[1] / x_num)
        y_step = int(frame.shape[0] / y_num)

        detect_params = []
        for i in range(x_num):
            for j in range(y_num):
                region_detector_path = Path('/home/jt1/Desktop/test_thresh_detect/block') / (
                        str(i) + '-' + str(j))
                detect_params.append(
                    DetectorParams(x_step, y_step, i, j, self.cfg, region_detector_path))
        return detect_params

    def detect_frame(self, frame, idx, video_path=None):
        # frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        rects = []
        frame, original_frame = preprocess(frame, self.cfg)
        # print(f'frame.shape={frame.shape}, original_frame.shape={original_frame.shape}')
        for d in self.detect_params:
            # print(f'frame.shape={frame.shape}, d.start={d.start}, d.end={d.end}')
            block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                  idx, original_frame.shape)
            # TODO we have to use a more robust frontground extraction algorithm get the binary map
            #  adaptive thresh to get binay map is just a compromise,
            #  it don't work if river background is complicated
            sub_result = adaptive_thresh_with_rules(block.frame, block, d)
            # shape = sub_result.binary.shape
            # mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
            # mask[100:900, :] = 255
            # sub_result = adaptive_thresh_mask_no_filter(block.frame, mask, block, d)

            # sub_result = detect_based_task(block, d)
            for rect in sub_result.rects:
                rects.append(rect)
        return rects

    def detect_video(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(self.output_path, f'{video_name}_detected.mp4')
        video_capture = cv.VideoCapture(video_path)
        frame_size = (int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        fps = video_capture.get(cv.CAP_PROP_FPS)
        print(f'video_path=[{video_path}], frame_size={frame_size}, fps={fps}')
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        video_writer = cv.VideoWriter(output_video_path, fourcc, fps, frame_size)

        ret, frame = video_capture.read()
        idx = 0
        while ret:
            rects = self.detect_frame(frame, idx)
            temp = []
            for rect in rects:
                if 0 < rect[1] < 2160:
                    temp.append(rect)
                    cv.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)
            print(f'[{video_name}], idx={idx}, filter_rects={temp}')
            # temp_img = cv.resize(frame, dsize=(960, 540))
            # cv.imshow('res_video', temp_img)
            # cv.waitKey(1)
            video_writer.write(frame)
            ret, frame = video_capture.read()
            idx += 1
        video_capture.release()
        video_writer.release()
        print(f'[{output_video_path}] saved...')


def test_detect_frame():
    parser = argparse.ArgumentParser(description='detect video')
    parser.add_argument('--c', type=str, default='dolphin')
    args = parser.parse_args()
    cfg_path = '/home/lxd/jsy/DolphinDetection/vcfg/prod/test-37.yml'
    video_dir = f'/home/lxd/jsy/DolphinDetection/test_data/input/{args.c}'
    output_path = f'/home/lxd/jsy/DolphinDetection/test_data/output/{args.c}_detected'
    os.makedirs(output_path, exist_ok=True)
    video_detector = VideoDetector(cfg_path, output_path)
    frame_path = '/home/lxd/jsy/DolphinDetection/experiments/data/498.png'
    frame = cv.imread(frame_path)
    frame = cv.resize(frame, (1920, 1080))
    video_detector.detect_frame(frame, idx=1)


def test_detect_video():
    cfg_path = '/home/lxd/jsy/DolphinDetection/vcfg/test/video-5.yml'
    video_dir = f'/home/lxd/jsy/DolphinDetection/test_data/input/dolphin'
    output_path = f'/home/lxd/jsy/DolphinDetection/test_data/output/dolphin_detected'
    os.makedirs(output_path, exist_ok=True)
    video_detector = VideoDetector(cfg_path, output_path)
    video_name_list = os.listdir(video_dir)
    video_name_list.sort()
    for video_name in video_name_list:
        video_path_ = os.path.join(video_dir, video_name)
        video_detector.detect_video(video_path=video_path_)


def test_detect_video1():
    cfg_path = '../vcfg/test/test_thresh_detect.yml'
    video_dir = f'/home/jt1/Desktop/DolphinDataset/Dolphin/DolphinClassification'
    output_dir = f'/home/jt1/Desktop/test_thresh_detect_1'
    video_file_name_list = os.listdir(video_dir)
    res = {}
    if os.path.exists(f'{output_dir}/res.txt'):
        os.remove(f'{output_dir}/res.txt')
    f = open(f'{output_dir}/res.txt', 'w+')
    for video_file_name in video_file_name_list:
        video_dir_ = os.path.join(video_dir, video_file_name)
        output_dir_ = os.path.join(output_dir, video_file_name)
        os.makedirs(output_dir_, exist_ok=True)
        video_detector = VideoDetector(cfg_path, output_dir_)
        video_name_list = os.listdir(video_dir_)
        for video_name in video_name_list:
            video_path = os.path.join(video_dir_, video_name)
            video_detector.detect_video(video_path)
            temp = input(f'please input whether [{video_path}] detected dolphin\n')
            res[video_path] = temp
            print(f'[{video_path}]: {temp}', file=f)
    for key, value in res.items():
        print(f'{key}: {value}')


def test_detect_video2():
    cfg_path = '../vcfg/test/test_thresh_detect.yml'
    video_dir = f'/home/jt1/Desktop/DolphinDataset/Dolphin/DolphinClassification'
    output_dir = f'/home/jt1/Desktop/test_thresh_detect_1'
    video_file_name_list = os.listdir(video_dir)
    for video_file_name in video_file_name_list:
        video_dir_ = os.path.join(video_dir, video_file_name)
        output_dir_ = os.path.join(output_dir, video_file_name)
        os.makedirs(output_dir_, exist_ok=True)
        video_detector = VideoDetector(cfg_path, output_dir_)
        video_name_list = os.listdir(video_dir_)
        cnt = 0
        for video_name in video_name_list:
            print(f'{cnt+1}|{len(video_name_list)}')
            cnt += 1
            video_path = os.path.join(video_dir_, video_name)
            video_detector.detect_video(video_path)


def main():
    # test_detect_frame()
    # test_detect_video()
    test_detect_video2()


if __name__ == '__main__':
    main()

import sys

# sys.path.append('../')
from interface import *
from detection import *
import numpy as np
from pathlib import Path
import cv2 as cv


class VideoDetector(object):
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.cfg = self.load_enable_cfg()
        self.detect_params = self.set_detect_params()
        self.output_path = '/data/lxd/jsy/DolphinDetection/test_data/output'

    def load_enable_cfg(self):
        cfg_list = load_video_config(self.cfg_path)
        for cfg in cfg_list:
            if cfg.enable:
                return cfg
        else:
            print('no cfg enable...')
            exit()

    def set_detect_params(self):
        x_num = self.cfg.routine['col']
        y_num = self.cfg.routine['row']
        x_step = int(self.cfg.shape[1] / x_num)
        y_step = int(self.cfg.shape[0] / y_num)

        detect_params = []
        for i in range(x_num):
            for j in range(y_num):
                region_detector_path = Path('/data/lxd/jsy/DolphinDetection/data/candidates/test/5/blocks') / (
                        str(i) + '-' + str(j))
                detect_params.append(
                    DetectorParams(x_step, y_step, i, j, self.cfg, region_detector_path))
        return detect_params

    def detect_frame(self, frame, idx):
        frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        rects = []
        for d in self.detect_params:
            # print(f'idx={idx} ({d.x_index}, {d.y_index}) d.start={d.start}, d.end={d.end}')
            block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                  idx, frame.shape)
            sub_result = adaptive_thresh_mask_no_rules(block.frame, block, d)
            for rect in sub_result.rects:
                rects.append(rect)
        return rects

    def detect_video(self, video_path):
        print(f'video_path={video_path}')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(self.output_path, f'{video_name}_detected.mp4')
        video_capture = cv.VideoCapture(video_path)
        frame_size = (int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        fps = video_capture.get(cv.CAP_PROP_FPS)
        print(f'frame_size={frame_size}, fps={fps}')
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        video_writer = cv.VideoWriter(output_video_path, fourcc, fps, frame_size)

        ret, frame = video_capture.read()
        idx = 0
        while ret:
            rects = self.detect_frame(frame, idx)
            temp = []
            for rect in rects:
                # if rect[2] > 15 and rect[3] > 15 and 100 < rect[1] < 900:
                temp.append(rect)
                cv.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            print(f'idx={idx}, filter_rects={temp}')
            video_writer.write(frame)
            ret, frame = video_capture.read()
            idx += 1
        video_capture.release()
        video_writer.release()


class PostFastObjectFilter(object):
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.cfg = self.load_enable_cfg()
        self.detect_params = self.set_detect_params()
        self.speed_thresh_x = 20
        self.speed_thresh_y = 20

    def load_enable_cfg(self):
        cfg_list = load_video_config(self.cfg_path)
        for cfg in cfg_list:
            if cfg.enable:
                return cfg
        else:
            print('no cfg enable...')
            exit()

    def set_detect_params(self):
        x_num = self.cfg.routine['col']
        y_num = self.cfg.routine['row']
        x_step = int(self.cfg.shape[1] / x_num)
        y_step = int(self.cfg.shape[0] / y_num)

        detect_params = []
        for i in range(x_num):
            for j in range(y_num):
                region_detector_path = Path('/data/lxd/jsy/DolphinDetection/data/candidates/test/5/blocks') / (
                        str(i) + '-' + str(j))
                detect_params.append(
                    DetectorParams(x_step, y_step, i, j, self.cfg, region_detector_path))
        return detect_params

    def detect_frame(self, frame, idx):
        frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        rects = []
        for d in self.detect_params:
            block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                  idx, frame.shape)
            shape = block.frame.shape
            mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
            mask[100:900, :] = 255
            sub_result = adaptive_thresh_mask_no_rules(block.frame, mask, block, d)
            # sub_result = detect_based_task(block, d)
            for rect in sub_result.rects:
                rects.append(rect)
        return rects

    def detect_video(self, video_path):
        result_set = []
        video_capture = cv2.VideoCapture(video_path)
        ret, frame = video_capture.read()
        idx = 0
        while ret:
            temp = []
            rects = self.detect_frame(frame, idx)
            for rect in rects:
                if rect[2] > 15 and rect[3] > 15 and 100 < rect[1] < 900:
                    temp.append(rect)
            if len(temp) > 0:
                result_set.append((idx, temp))
            # logger.info(f'idx={idx}, temp={temp}, len_rects={len(rects)}')
            ret, frame = video_capture.read()
            idx += 1
        video_capture.release()
        return result_set

    def get_median(self, data):
        if len(data) == 0:
            return 0
        data.sort()
        half = len(data) // 2
        return (data[half] + data[~half]) / 2

    def get_max_continuous_time(self, data):
        if len(data) == 0:
            return 0
        return max(data)

    def post_filter(self, video_path, task_cnt):
        """
        :param video_path: the path of the generated video to be filtered
        :param task_cnt:
        :return: False: no fast-object;
                 True: exists fast-object or float
        """
        logger.info(f'Post filter [{self.cfg.index}, {task_cnt}]: started...')
        if not os.path.exists(video_path):
            logger.info(f'Post filter [{self.cfg.index}, {task_cnt}]: {video_path} is not exists...')
            return False
        result_set = self.detect_video(video_path)
        if len(result_set) <= 1:
            return False
        continuous_time_set = []
        speed_x_set = []
        speed_y_set = []
        continuous_time = 0
        for i in range(1, len(result_set)):
            pre_idx, pre_rects = result_set[i - 1]
            current_idx, current_rects = result_set[i]
            if len(pre_rects) == len(current_rects):
                if abs(current_idx - pre_idx) <= 3:
                    continuous_time += abs(current_idx - pre_idx)
                else:
                    continuous_time_set.append(continuous_time)
                    continuous_time = 0
                for j in range(len(pre_rects)):
                    pre_rect = pre_rects[j]
                    current_rect = current_rects[j]
                    pre_center_x = pre_rect[0] + pre_rect[2]
                    pre_center_y = pre_rect[1] + pre_rect[3]
                    current_center_x = current_rect[0] + current_rect[2] / 2
                    current_center_y = current_rect[1] + current_rect[3] / 2
                    speed_x = abs(pre_center_x - current_center_x) / abs(pre_idx - current_idx)
                    speed_y = abs(pre_center_y - current_center_y) / abs(pre_idx - current_idx)
                    speed_x_set.append(speed_x)
                    speed_y_set.append(speed_y)
                    # logger.info(f'speed_x={speed_x}')
                    # logger.info(f'speed_y={speed_y}')
            else:
                continuous_time_set.append(continuous_time)
                continuous_time = 0
        continuous_time_set.append(continuous_time)
        median_speed_x = self.get_median(speed_x_set)
        median_speed_y = self.get_median(speed_y_set)
        max_continuous_time = self.get_max_continuous_time(continuous_time_set)
        logger.info(
            f'Post filter [{self.cfg.index}, {task_cnt}]: median_speed_x={median_speed_x},'
            f' median_speed_y={median_speed_y}, max_continuous_time={max_continuous_time}')
        if median_speed_x > self.speed_thresh_x or median_speed_y > self.speed_thresh_y:
            logger.info(
                f'Post filter [{self.cfg.index}, {task_cnt}]: detect fast-object in [{video_path}]')
            return True
        elif max_continuous_time > 20:
            logger.info(
                f'Post filter [{self.cfg.index}, {task_cnt}]: detect float in [{video_path}]')
            return True
        else:
            return False


if __name__ == '__main__':
    cfg_path = '/data/lxd/jsy/DolphinDetection/vcfg/video-dev.json'
    video_detector = VideoDetector(cfg_path)
    video_dir = '/data/lxd/jsy/DolphinDetection/test_data/input/target'
    video_name_list = os.listdir(video_dir)
    for video_name in video_name_list:
        video_path_ = os.path.join(video_dir, video_name)
        video_detector.detect_video(video_path=video_path_)

    # cfg_path = '/data/lxd/jsy/DolphinDetection/vcfg/video-dev.json'
    # post_fast_object_filter = PostFastObjectFilter(cfg_path)
    # video_dir = '/data/lxd/jsy/DolphinDetection/test_data/input/float'
    # video_name_list = os.listdir(video_dir)
    # task_cnt_ = 0
    # for video_name in video_name_list:
    #     video_path_ = os.path.join(video_dir, video_name)
    #     if post_fast_object_filter.post_filter(video_path_, task_cnt_):
    #         logger.info(f'{task_cnt_} detect fast-object in [{video_path_}]')
    #     else:
    #         logger.info(f'{task_cnt_} does not detect fast-object in [{video_path_}]')
    #     task_cnt_ += 1

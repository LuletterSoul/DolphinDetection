#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: utils.py
@time: 2019/11/7 18:19
@version 1.0
@desc:
"""
import time
from multiprocessing import Queue
import traceback

import imutils

from config import VideoConfig
from utils import *

# from utils.log import logger

# initialize the motion saliency object and start the video stream
saliency = None
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
eps = 1e-5
# loop over frames from the video file stream
crop = [[0, 540], [1919, 539], [1919, 1079]]

#
global_mean = np.array([110, 110, 80])
global_std = None
dolphin_mean_intensity = np.array([80, 80, 80])
alpha = 15
beta = 15


class DetectionResult(object):

    def __init__(self, frame, status, regions) -> None:
        super().__init__()
        self.frame= frame
        self.status = status
        self.regions = regions


class Detector(object):
    def __init__(self, col_step, col_step_1, row_index, col_index, cfg: VideoConfig, sq: Queue, rq: Queue,
                 region_save_path: Path) -> None:
        super().__init__()
        # self.video_path = video_path
        # self.region_save_path = region_save_path
        self.cfg = cfg
        # self.routine = cfg.routine
        self.global_std = None
        self.global_mean = None
        self.row_step = col_step
        self.col_step = col_step_1
        self.col_index = row_index
        self.raw_index = col_index
        self.start = [self.raw_index * col_step_1, self.col_index * col_step]
        self.end = [(self.raw_index + 1) * col_step_1, (self.col_index + 1) * col_step]
        self.sq = sq
        self.rq = rq
        self.region_save_path = region_save_path
        self.region_save_path.mkdir(exist_ok=True, parents=True)
        logger.debug(
            'Detector [{},{}]: region save to: [{}]'.format(self.col_index, self.col_index, str(self.region_save_path)))
        self.global_mean = np.array([80, 80, 80])
        self.global_std = np.array([50, 50, 50])
        self.dolphin_mean_intensity = np.array([80, 80, 80])
        self.alpha = 15
        self.beta = 15
        self.std_thresh = 20
        self.region_cnt = 0
        self.detect_cnt = 0

    def get_frame(self):
        return self.crop(self.sq.get())

    def crop(self, frame):
        return crop_by_se(frame, self.start, self.end)

    def not_belong_do(self, candidate_mean, thresh=10):
        # global global_mean, dolphin_mean_intensity
        # bg_dist = euclidean_distance(candidate_mean, bg_mean_intensity)
        gt_dist = euclidean_distance(candidate_mean, self.dolphin_mean_intensity)
        # logger.debug('Distance with background threshold: [{}].'.format(bg_dist))
        logger.debug('Distance with Ground Truth threshold: [{}].'.format(gt_dist))
        # the smaller euclidean distance with class, it's more reliable towards the correct detection
        return gt_dist < thresh

    def do_decision(self, region_mean, region_std):
        # global global_mean, dolphin_mean_intensity, global_std
        bg_dist = euclidean_distance(region_mean, self.global_mean)
        gt_dist = euclidean_distance(region_mean, self.dolphin_mean_intensity)
        std_dist = euclidean_distance(region_std, self.global_std)
        logger.debug('Mean Distance with background threshold: [{}].'.format(bg_dist))
        logger.debug('Mean Distance with Ground Truth threshold: [{}].'.format(gt_dist))
        logger.info('Std Distance with global: [{}].'.format(std_dist))
        # the smaller euclidean distance with class, it's more reliable towards the correct detection
        is_true = gt_dist < bg_dist and (std_dist > self.std_thresh)
        return is_true, bg_dist, gt_dist

    def not_belong_bg(self, region_mean, thresh=20):
        bg_dist = euclidean_distance(region_mean, self.global_mean)
        logger.debug('Distance with background threshold: [{}].'.format(bg_dist))
        return bg_dist > thresh

    def is_in_ratio(self, area, total):
        return (area / total) * 100 <= self.cfg.filtered_ratio

    def detect(self):
        try:
            # if not isinstance(mq, Queue):
            #     raise Exception('Queue must be capable of multi-processing safety.')
            logger.info(
                'Detector: [{},{},{}] Init detection process......'.format(self.cfg.index, self.col_index,
                                                                           self.raw_index))
            # global global_mean, global_std, dolphin_mean_intensity

            frame = self.get_frame()

            logger.debug(
                'Detector [{},{}]: init saliency detector'.format(self.col_index, self.col_index))
            self.saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            self.saliency.setImagesize(frame.shape[1], frame.shape[0])
            self.saliency.init()
            self.shape = frame.shape
            # if vs.isOpened():
            #     logger.error('Video Open Failed: [{}]'.format(ts_path))
            #     continue
            while True:
                # logger.info('Detector: [{},{}] Fetch frame'.format(self.row_index, self.col_index))
                frame = self.get_frame()
                if frame is None:
                    logger.info('Detector: [{},{}] empty frame')
                    continue
                # logger.info('Detector: [{},{}] Fetch frame done..'.format(self.row_index, self.col_index))
                original_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                success, saliency_map = self.saliency.computeSaliency(gray)
                saliency_map = (saliency_map * 255).astype("uint8")
                # do dilation to connect the splited small components
                dilated = cv2.dilate(saliency_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                # dilated = saliency_map

                # 获取所有检测框
                connectivity = 8
                # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(dilated,
                                                                                                        connectivity,
                                                                                                        cv2.CV_16U,
                                                                                                        cv2.CCL_DEFAULT)
                # centroids = centroids.astype(np.int)
                if num_labels == 0:
                    logger.debug('Not found connected components')
                    continue
                # stats = [s for s in stats if in_range(s[0], frame.shape[1]) and in_range(s[1], frame.shape[0])]
                # sorted(stats, key=lambda s: s[4], reverse=False)
                # background components
                logger.debug('Components num: [{}].'.format(num_labels))
                # logger.debug('Background label: [{}]'.format(label_map[centroids[0][1], centroids[0][0]]))
                old_b = self.global_mean
                old_d = self.dolphin_mean_intensity
                # new_b = np.reshape(frame, (-1, 3)).mean(axis=0)
                new_b, self.global_std = cv2.meanStdDev(frame)
                # in case env become dark suddenly
                if np.mean(old_b - new_b) > alpha:
                    self.dolphin_mean_intensity -= beta
                    logger.info(
                        'Down dolphin mean intensity from [{}] to [{}]'.format(old_d, self.dolphin_mean_intensity))
                # in case env become light suddenly
                if np.mean(new_b - old_b) > alpha:
                    self.dolphin_mean_intensity += beta
                    logger.info(
                        'Increase dolphin mean intensity from [{}] to [{}]'.format(old_d, self.dolphin_mean_intensity))
                logger.debug(
                    'Update mean global background pixel intensity from [{}] to [{}].'.format(old_b, new_b))

                self.global_mean = new_b

                logger.debug('Current mean of bg: [{}].'.format(np.reshape(new_b, (1, -1))))
                logger.debug('Current std of bg: [{}]'.format(np.reshape(self.global_std, (1, -1))))

                regions = []
                status = []
                for idx, s in enumerate(stats):
                    # potential dolphin target components
                    if s[0] and s[1] and in_range(s[0], frame.shape[1]) and in_range(s[1], frame.shape[0]):
                        region_mean, mask_frame = cal_mean_intensity(frame, idx, label_map, s[4])
                        region = original_frame[s[1] - 10: s[1] + s[3] + 10, s[0] - 10: s[0] + s[2] + 10]
                        _, region_std = cv2.meanStdDev(region)
                        logger.info(
                            'Area: [{}], ration: [{}]'.format(s[4],
                                                              round(s[4] / (frame.shape[0] * frame.shape[1]) * 100, 3)))
                        is_in_ratio = self.is_in_ratio(s[4], self.shape[0] * self.shape[1])
                        is_dolphin, bg_dist, gt_dist = self.do_decision(region_mean, region_std)
                        if is_dolphin and is_in_ratio:
                            # logger.info('Bg dist: [{}]/ Gt dist: [{}].'.format(bg_dist, gt_dist))
                            color = np.random.randint(0, 255, size=(3,))
                            color = [int(c) for c in color]
                            cv2.rectangle(frame, (s[0] - 10, s[1] - 10), (s[0] + s[2] + 10, s[1] + s[3] + 10), color, 2)
                            # if region.shape[0] and region.shape[1]:
                            cv2.imwrite(str(self.region_save_path / (
                                    str(self.region_cnt) + '-' + str(int(region_mean[0])) + '-' + str(
                                int(region_mean[1])) + '-' +
                                    str(int(region_mean[2])) + '.png')), region)
                            regions.append(region)
                            status.append(s)
                        self.region_cnt += 1

                # display the image to our screen
                if self.cfg.show_window:
                    cv2.imshow("Frame", frame)
                    # cv2.imshow("Map", dilated)
                    key = cv2.waitKey(1) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                self.detect_cnt += 1
                logger.debug('Current mean of bg: [{}].'.format(np.reshape(new_b, (1, -1))))
                logger.debug(
                    'Detector: [{},{}] detect done [{}] frames..'.format(self.col_index, self.raw_index,
                                                                         self.detect_cnt))
                # self.rq.put(frame)
                self.rq.put(DetectionResult(frame, status, regions))

            # do a bit of cleanup
            cv2.destroyAllWindows()

            # vs.release()

        except Exception as msg:
            traceback.print_exc()
            logger.error(msg)


# def setChunks(self, frame):


class DetectorController(object):
    def __init__(self, video_path: Path, region_save_path: Path, mq, cfg: VideoConfig) -> None:
        super().__init__()
        self.video_path = video_path
        self.region_save_path = region_save_path
        self.mq = mq
        self.cfg = cfg


def fetch_component_frame(frame, label_map, idx):
    mask = (label_map == idx)
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, axis=2, repeats=3).astype(np.uint8) * 255
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)
    return cv2.bitwise_and(frame, mask)


def euclidean_distance(p1, p2):
    # if p1.shape != p2.shape:
    #     raise Exception("Point 1 shape:[{}],Point 2 shape:[{}]".format(p1.shape, p2.shape))
    return np.sqrt(np.sum((p1 - p2) ** 2))


def not_belong_do(candidate_mean, thresh=10):
    global global_mean, dolphin_mean_intensity
    # bg_dist = euclidean_distance(candidate_mean, bg_mean_intensity)
    gt_dist = euclidean_distance(candidate_mean, dolphin_mean_intensity)
    # logger.debug('Distance with background threshold: [{}].'.format(bg_dist))
    logger.debug('Distance with Ground Truth threshold: [{}].'.format(gt_dist))
    # the smaller euclidean distance with class, it's more reliable towards the correct detection
    return gt_dist < thresh


def do_decision(region_mean, region_std):
    global global_mean, dolphin_mean_intensity, global_std
    bg_dist = euclidean_distance(region_mean, global_mean)
    gt_dist = euclidean_distance(region_mean, dolphin_mean_intensity)
    std_dist = euclidean_distance(region_std, global_std)
    logger.debug('Mean Distance with background threshold: [{}].'.format(bg_dist))
    logger.debug('Mean Distance with Ground Truth threshold: [{}].'.format(gt_dist))
    logger.debug('Std Distance with global: [{}].'.format(std_dist))
    # the smaller euclidean distance with class, it's more reliable towards the correct detection
    return gt_dist < bg_dist, bg_dist, gt_dist


def not_belong_bg(region_mean, thresh=20):
    global global_mean
    bg_dist = euclidean_distance(region_mean, global_mean)
    logger.debug('Distance with background threshold: [{}].'.format(bg_dist))
    return bg_dist > thresh


def detect(video_path: Path, region_save_path: Path, mq: Queue, cfg: VideoConfig):
    # if not isinstance(mq, Queue):
    #     raise Exception('Queue must be capable of multi-processing safety.')
    logger.info('Init detection process......')

    video_path.mkdir(exist_ok=True, parents=True)
    region_save_path.mkdir(exist_ok=True, parents=True)

    global global_mean, global_std, dolphin_mean_intensity
    saliency = None
    target_cnt = 0
    save_path = Path(region_save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    while True:
        if mq.empty():
            logger.debug('Empty cache section wait....')
            time.sleep(1)
            continue
        ts_poxis = video_path / mq.get()
        ts_path = str(ts_poxis)

        if not os.path.exists(ts_path):
            logger.debug('Ts path not exist: [{}]'.format(ts_path))
            continue

        vs = cv2.VideoCapture(ts_path)

        # if vs.isOpened():
        #     logger.error('Video Open Failed: [{}]'.format(ts_path))
        #     continue

        while True:
            # grab the frame from the threaded video stream and resize it
            # to 500px (to speedup processing)
            ret, frame = vs.read()
            if not ret:
                logger.info('Read frame failed from [{}].'.format(ts_path))
                break
            frame, original_frame = preprocess(frame, cfg)

            # frame_height, frame_width, _ = frame.shape
            # if our saliency object is None, we need to instantiate it
            if saliency is None:
                saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
                saliency.setImagesize(frame.shape[1], frame.shape[0])
                saliency.init()

            # convert the input frame to grayscale and compute the saliency
            # map based on the motion model
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (success, saliency_map) = saliency.computeSaliency(gray)
            saliency_map = (saliency_map * 255).astype("uint8")

            # do dilation to connect the splited small components
            dilated = cv2.dilate(saliency_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            # dilated = saliency_map

            # 获取所有检测框
            connectivity = 8
            # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(dilated,
                                                                                                    connectivity,
                                                                                                    cv2.CV_16U,
                                                                                                    cv2.CCL_DEFAULT)
            # centroids = centroids.astype(np.int)
            if num_labels == 0:
                logger.debug('Not found connected components')
                continue
            # stats = [s for s in stats if in_range(s[0], frame.shape[1]) and in_range(s[1], frame.shape[0])]
            # sorted(stats, key=lambda s: s[4], reverse=False)
            # background components
            logger.debug('Components num: [{}].'.format(num_labels))
            # logger.debug('Background label: [{}]'.format(label_map[centroids[0][1], centroids[0][0]]))
            old_b = global_mean
            old_d = dolphin_mean_intensity
            # new_b = np.reshape(frame, (-1, 3)).mean(axis=0)
            new_b, global_std = cv2.meanStdDev(frame)
            # in case env become dark suddenly
            if np.mean(old_b - new_b) > alpha:
                dolphin_mean_intensity -= beta
                logger.info('Down dolphin mean intensity from [{}] to [{}]'.format(old_d, dolphin_mean_intensity))
            # in case env become light suddenly
            if np.mean(new_b - old_b) > alpha:
                dolphin_mean_intensity += beta
                logger.info('Increase dolphin mean intensity from [{}] to [{}]'.format(old_d, dolphin_mean_intensity))
            logger.debug(
                'Update mean global background pixel intensity from [{}] to [{}].'.format(old_b, new_b))
            global_mean = new_b

            logger.debug('Current mean of bg: [{}].'.format(np.reshape(new_b, (1, -1))))
            logger.debug('Current std of bg: [{}]'.format(np.reshape(global_std, (1, -1))))

            for idx, s in enumerate(stats):
                # potential dolphin target components
                if s[0] and s[1] and in_range(s[0], frame.shape[1]) and in_range(s[1], frame.shape[0]):
                    region_mean, mask_frame = cal_mean_intensity(frame, idx, label_map, s[4])
                    region = original_frame[s[1] - 10: s[1] + s[3] + 10, s[0] - 10: s[0] + s[2] + 10]
                    _, region_std = cv2.meanStdDev(region)
                    # logger.info(region_std)
                    is_dolphin, bg_dist, gt_dist = do_decision(region_mean, region_std)
                    if is_dolphin:
                        logger.info('Bg dist: [{}]/ Gt dist: [{}].'.format(bg_dist, gt_dist))
                        color = np.random.randint(0, 255, size=(3,))
                        color = [int(c) for c in color]
                        cv2.rectangle(frame, (s[0] - 10, s[1] - 10), (s[0] + s[2] + 10, s[1] + s[3] + 10), color, 2)
                        # candidate = original_frame[s[1] - 10: s[1] + s[3] + 10, s[0] - 10: s[0] + s[2] + 10]
                        # cv2.imwrite(str(save_path / str(target_cnt) / '.png'), candidate)
                        target_cnt += 1

            # display the image to our screen
            if cfg.show_window:
                cv2.imshow("Frame", frame)
                # cv2.imshow("Map", dilated)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

        # clean ts file
        ts_poxis.unlink()

        cv2.destroyAllWindows()
        vs.release()


def preprocess(frame, cfg):
    if cfg.resize['scale'] != -1:
        frame = cv2.resize(frame, (0, 0), fx=cfg.resize['scale'], fy=cfg.resize['scale'])
    elif cfg.resize['width'] != -1:
        frame = imutils.resize(frame, cfg.resize['width'])
    elif cfg.resize['height '] != -1:
        frame = imutils.resize(frame, cfg.resize['height'])
    frame = crop_by_roi(frame, cfg.roi)
    # frame = imutils.resize(frame, width=1000)
    # frame = frame[340:, :, :]
    # frame = frame[170:, :, :]
    original_frame = frame.copy()
    frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
    return frame, original_frame


def cal_mean_intensity(frame, idx, label_map, area, mean=None, mask_frame=None):
    mask_frame = fetch_component_frame(frame, label_map, idx)
    mean = np.reshape(mask_frame, (-1, 3)).sum(axis=0) / (area + eps)
    return np.reshape(mean, (3, 1)), mask_frame


def test_connected_components():
    g = np.array([[0, 1, 1, 1],
                  [0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [1, 1, 0, 1],
                  [1, 1, 0, 0]], dtype=np.uint8)
    g2 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]], dtype=np.uint8)

    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(g, 4,
                                                                                            cv2.CV_16U,
                                                                                            cv2.CCL_DEFAULT)
    print(label_map)
    print(num_labels)
    print(stats[0])
    print(stats[1])
    print(stats[2])
    print(stats[3])
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(g2, 4,
                                                                                            cv2.CV_16U,
                                                                                            cv2.CCL_DEFAULT)
    print(label_map)
    print(num_labels)
    print(stats[0])

# if __name__ == '__main__':
#     detect()
# test_connected_components()

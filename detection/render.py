#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: render.py
@time: 2/27/20 8:53 AM
@version 1.0
@desc:
"""
import threading
import time
from multiprocessing import Manager
from multiprocessing.queues import Queue

import cv2
import numpy as np
import os

from config import SystemStatus
# from .manager import DetectorController
from stream.websocket import creat_packaged_msg_json
from utils import logger, bbox_points, generate_time_stamp, crop_by_se
from stream.rtsp import FFMPEG_MP4Writer
import traceback
from config import VideoConfig
from detection.params import DispatchBlock, DetectorParams
from detection.detect_funcs import detect_based_task, adaptive_thresh_mask_no_rules, adaptive_thresh_with_rules
from pathlib import Path
from utils import preprocess, paint_chinese_opencv


class DetectionStreamRender(object):

    def __init__(self, cfg, detect_index, future_frames, msg_queue: Queue,
                 rect_stream_path, original_stream_path, render_frame_cache, render_rect_cache, original_frame_cache,
                 notify_queue, region_path, detect_params=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.detect_index = detect_index
        # self.rect_stream_path = controller.rect_stream_path
        # self.original_stream_path = controller.original_stream_path
        self.rect_stream_path = rect_stream_path
        self.original_stream_path = original_stream_path
        self.stream_cnt = 0
        self.index = cfg.index
        self.is_trigger_write = True
        self.write_done = True
        # self.controller = controller
        self.cache_size = cfg.cache_size
        self.future_frames = future_frames
        # self.sample_rate = controller.cfg.sample_rate
        # self.render_frame_cache = controller.render_frame_cache
        # self.render_rect_cache = controller.render_rect_cache
        # self.original_frame_cache = controller.original_frame_cache
        self.sample_rate = cfg.sample_rate
        self.render_frame_cache = render_frame_cache
        self.render_rect_cache = render_rect_cache
        self.original_frame_cache = original_frame_cache
        self.next_prepare_event = Manager().Event()
        self.next_prepare_event.set()
        self.msg_queue = msg_queue
        # self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        # self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.quit = Manager().Event()
        self.quit.clear()
        self.status = Manager().Value('i', SystemStatus.RUNNING)
        self.notify_queue = notify_queue
        self.LOG_PREFIX = f'Video Stream Render [{self.cfg.index}]: '
        self.post_filter = PostFilter(self.cfg, region_path, detect_params)

    def listen(self):
        if self.quit.wait():
            self.next_prepare_event.set()
            self.status.set(SystemStatus.SHUT_DOWN)

    def next_st(self, detect_index):
        if detect_index - self.detect_index > self.future_frames:
            return detect_index
        else:
            return self.detect_index

    def is_window_reach(self, detect_index):
        return detect_index - self.detect_index > self.future_frames

    def reset(self, detect_index):
        if self.is_window_reach(detect_index):
            self.detect_index = detect_index
            self.is_trigger_write = False
            self.write_done = False
            self.next_prepare_event.set()
            logger.info('Reset stream render')

    def loop_render_msg(self):
        logger.info(
            f'*******************************{self.LOG_PREFIX}: Init Stream Render Notify Service********************************')
        threading.Thread(target=self.listen, daemon=True).start()
        while self.status.get() == SystemStatus.RUNNING:
            try:
                index, type = self.notify_queue.get()
                if type == 'notify':
                    self.notify(index)
                if type == 'reset':
                    self.reset(index)
            except Exception as e:
                logger.error(e)
        logger.info(
            f'*******************************{self.LOG_PREFIX}: Exit Stream Render Notify Service********************************')

    def notify(self, current_index):
        # next_detect_stream_occurred = current_index - self.detect_index >= self.future_frames \
        #                               and not self.is_trigger_write
        if not self.is_trigger_write:
            if self.next_prepare_event.is_set():
                self.next_prepare_event.clear()
                # begin task asynchronously  in case blocking collector
                self.render_task(current_index, self.render_frame_cache, self.render_rect_cache,
                                 self.original_frame_cache)
                self.is_trigger_write = True
        if current_index - self.detect_index >= self.future_frames and self.write_done:
            # notify render task that the future frames(2s default required) are done
            if not self.next_prepare_event.is_set():
                self.next_prepare_event.set()
                logger.info(
                    f'Notify detection stream writer.Current frame index [{current_index}],Previous detected frame index [{self.detect_index}]...')

    def write_render_video_work(self, video_write, next_cnt, end_cnt, render_cache, rect_cache, frame_cache):
        if next_cnt < 1:
            next_cnt = 1

        render_cnt = 0
        rects = []
        for index in range(next_cnt, end_cnt + 1):
            frame = frame_cache[index]
            if frame is None:
                continue
            tmp_rects = self.render_rect_cache[index % self.cache_size]
            self.render_rect_cache[index % self.cache_size] = None
            if tmp_rects is not None:
                render_cnt = 0
                rects = tmp_rects
            is_render = render_cnt <= 36
            if is_render:
                for rect in rects:
                    color = np.random.randint(0, 255, size=(3,))
                    color = [int(c) for c in color]
                    p1, p2 = bbox_points(self.cfg, rect, frame.shape)
                    frame = paint_chinese_opencv(frame, '江豚', p1)
                    cv2.rectangle(frame, p1, p2, color, 2)
                render_cnt += 1
            next_cnt += 1
            video_write.write(frame)
        return next_cnt

    def write_original_video_work(self, video_write, next_cnt, end_cnt, frame_cache):
        if next_cnt < 1:
            next_cnt = 1
        start = time.time()
        try_times = 0
        while next_cnt < end_cnt:
            try:
                if self.status.get() == SystemStatus.SHUT_DOWN:
                    logger.info(
                        f'Video Render [{self.index}]: original task interruped by exit signal')
                    return False
                # if next_cnt in frame_cache:
                frame = frame_cache[next_cnt % self.cache_size]
                if frame is not None:
                    video_write.write(frame)
                    # frame_cache[next_cnt % self.cache_size] = None
                    next_cnt += 1
                else:
                    try_times += 1
                    time.sleep(0.5)
                    if try_times > 100:
                        try_times = 0
                        logger.info(f'Try time overflow.round to the next cnt: [{try_times}]')
                        next_cnt += 1
                    logger.info(f'Lost frame index: [{next_cnt}]')

                end = time.time()
                if end - start > 30:
                    logger.info('Task time overflow, complete previous render task.')
                    break
            except Exception as e:
                end = time.time()
                if end - start > 30:
                    logger.info('Task time overflow, complete previous render task.')
                    break
                logger.error(e)
        return next_cnt

    def render_task(self, current_idx, render_cache, rect_cache, frame_cache):
        current_time = generate_time_stamp('%m%d%H%M%S') + '_'
        rect_render_thread = threading.Thread(
            target=self.rect_render_task,
            args=(current_idx, current_time, frame_cache,
                  rect_cache, render_cache,), daemon=True)
        # rect_render_thread.setDaemon(True)
        rect_render_thread.start()
        # self.rect_render_task(current_idx, current_time, frame_cache, rect_cache, render_cache)
        # self.original_render_task(current_idx, current_time, frame_cache)
        original_render_thread = threading.Thread(
            target=self.original_render_task,
            args=(current_idx, current_time, frame_cache,), daemon=True)
        # original_render_thread.setDaemon(True)
        original_render_thread.start()
        self.write_done = True
        self.stream_cnt += 1
        return True

    def rect_render_task(self, current_idx, current_time, frame_cache, rect_cache, render_cache):
        start = time.time()
        task_cnt = self.stream_cnt
        # raw_target = self.original_stream_path / (current_time + str(self.stream_cnt) + '_raw' + '.mp4')
        target = self.rect_stream_path / (current_time + str(task_cnt) + '.mp4')
        logger.info(
            f'Video Render [{self.index}]: Rect Render Task [{task_cnt}]: Writing detection stream frame into: [{str(target)}]')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # video_write = cv2.VideoWriter(str(raw_target), self.fourcc, 24.0, (self.cfg.shape[1], self.cfg.shape[0]), True)
        video_write = FFMPEG_MP4Writer(str(target), (self.cfg.shape[1], self.cfg.shape[0]), 25)
        next_cnt = current_idx - self.future_frames
        # next_cnt = self.write_render_video_work(video_write, next_cnt, current_idx, render_cache, rect_cache,
        #                                         frame_cache)
        # the future frames count
        # next_frame_cnt = 48
        # wait the futures frames is accessable
        if not self.next_prepare_event.is_set():
            logger.info(
                f'Video Render [{self.index}]: Rect Render Task [{task_cnt}] wait frames accessible....')
            start = time.time()
            # wait the future frames prepared,if ocurring time out, give up waits
            self.next_prepare_event.wait(30)
            logger.info(
                f"Video Render [{self.cfg.index}]: Rect Render Task " +
                f"[{task_cnt}] wait [{round(time.time() - start, 2)}] seconds")
            logger.info(f'Video Render [{self.index}]: Rect Render Task [{task_cnt}] frames accessible...')

        # if not self.started:
        #     return False
        # logger.info('Render task Begin with frame [{}]'.format(next_cnt))
        # logger.info('After :[{}]'.format(render_cache.keys()))
        end_cnt = current_idx + self.future_frames
        try:
            next_cnt = self.write_render_video_work(video_write, next_cnt, end_cnt, render_cache, rect_cache,
                                                    frame_cache)
        except Exception as e:
            logger.error(e)
        video_write.release()
        # self.convert_byfile(str(raw_target), str(target))
        logger.info(
            f'Video Render [{self.index}]: Rect Render Task [{task_cnt}]: Consume [{round(time.time() - start, 2)}] ' +
            f'seconds.Done write detection stream frame into: [{str(target)}]')
        if not self.post_filter.post_filter_video(str(target), task_cnt):
            msg_json = creat_packaged_msg_json(filename=str(target.name), path=str(target), cfg=self.cfg)
            self.msg_queue.put(msg_json)
            logger.info(self.LOG_PREFIX + f'Send packaged message: {msg_json} to msg_queue...')

    def original_render_task(self, current_idx, current_time, frame_cache):
        start = time.time()
        task_cnt = self.stream_cnt
        # raw_target = self.original_stream_path / (current_time + str(self.stream_cnt) + '_raw' + '.mp4')
        target = self.original_stream_path / (current_time + str(task_cnt) + '.mp4')
        logger.info(
            f'Video Render [{self.index}]: Original Render Task [{task_cnt}]: Writing detection stream frame into: [{str(target)}]')
        # video_write = cv2.VideoWriter(str(raw_target), self.fourcc, 24.0, (self.cfg.shape[1], self.cfg.shape[0]), True)
        video_write = FFMPEG_MP4Writer(str(target), (self.cfg.shape[1], self.cfg.shape[0]), 25)
        # if not video_write.isOpened():
        #     logger.error(f'Video Render [{self.index}]: Error Opened Video Writer')

        next_cnt = current_idx - self.future_frames
        next_cnt = self.write_original_video_work(video_write, next_cnt, current_idx, frame_cache)
        # the future frames count
        # next_frame_cnt = 48
        # wait the futures frames is accessable
        if not self.next_prepare_event.is_set():
            logger.info(f'Video Render [{self.index}]: Original Render Task wait frames accessible....')
            start = time.time()
            # wait the future frames prepared,if ocurring time out, give up waits
            self.next_prepare_event.wait(30)
            logger.info(
                f"Video Render [{self.cfg.index}]: Original Render Task [{task_cnt}] wait [{time.time() - start}] seconds")
            logger.info(f'Video Render [{self.index}]: Original Render Task [{task_cnt}] frames accessible....')
        # logger.info('Render task Begin with frame [{}]'.format(next_cnt))
        # logger.info('After :[{}]'.format(render_cache.keys()))
        # if not self.started:
        #     return False
        end_cnt = next_cnt + self.future_frames
        next_cnt = self.write_original_video_work(video_write, next_cnt, end_cnt, frame_cache)
        video_write.release()
        # self.convert_byfile(str(raw_target), str(target))
        logger.info(
            f'Video Render [{self.index}]: Original Render Task [{task_cnt}]: ' +
            f'Consume [{round(time.time() - start, 2)}] seconds.Done write detection stream frame into: [{str(target)}]')
        # if raw_target.exists():
        #     raw_target.unlink()
    #
    # def convert_avi(self, input_file, output_file, ffmpeg_exec="ffmpeg"):
    #     ffmpeg = '{ffmpeg} -y -i "{infile}" -c:v libx264 -strict -2 "{outfile}"'.format(ffmpeg=ffmpeg_exec,
    #                                                                                     infile=input_file,
    #                                                                                     outfile=output_file
    #                                                                                     )
    #     f = os.popen(ffmpeg)
    #     return f.readline()
    #
    # def convert_avi_to_webm(self, input_file, output_file, ffmpeg_exec="ffmpeg"):
    #     return self.convert_avi(input_file, output_file, ffmpeg_exec="ffmpeg")
    #
    # def convert_avi_to_mp4(self, input_file, output_file, ffmpeg_exec="ffmpeg"):
    #     return self.convert_avi(input_file, output_file, ffmpeg_exec="ffmpeg")
    #
    # def convert_to_avcmp4(self, input_file, output_file, ffmpeg_exec="ffmpeg"):
    #     email = threading.Thread(target=self.convert_avi, args=(input_file, output_file, ffmpeg_exec,))
    #     email.start()
    #
    # def convert_byfile(self, from_path, to_path):
    #     if not os.path.exists(from_path):
    #         logger.info("Sorry, you must create the directory for the output files first")
    #     if not os.path.exists(os.path.dirname(to_path)):
    #         os.makedirs(os.path.dirname(to_path), exist_ok=True)
    #     # directory, file_name = os.path.split(from_path)
    #     # raw_name, extension = os.path.splitext(file_name)
    #     # print("Converting ", from_path)
    #     line = self.convert_avi_to_mp4(from_path, to_path)
    #     logger.info(line)


class PostFilter(object):
    def __init__(self, cfg: VideoConfig, region_path, detect_params=None):
        self.cfg = cfg
        self.region_path = region_path
        self.block_path = region_path / 'blocks'
        # self.detect_params = self.set_detect_params()
        self.detect_params = detect_params
        self.speed_thresh_x = self.cfg.alg['speed_x_thresh']
        self.speed_thresh_y = self.cfg.alg['speed_y_thresh']
        self.continuous_time_thresh = self.cfg.alg['continuous_time_thresh']
        self.disappear_frames_thresh = self.cfg.alg['disappear_frames_thresh']

    def set_detect_params(self):
        x_num = 1
        y_num = 1
        x_step = int(self.cfg.shape[1] / x_num)
        y_step = int(self.cfg.shape[0] / y_num)

        detect_params = []
        for i in range(x_num):
            for j in range(y_num):
                region_detector_path = self.block_path / (str(i) + '-' + str(j))
                detect_params.append(
                    DetectorParams(x_step, y_step, i, j, self.cfg, region_detector_path))
        return detect_params

    def detect_frame(self, frame, idx, video_path=None):
        # frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        rects = []
        sub_results = []
        frame, original_frame = preprocess(frame, self.cfg)
        for d in self.detect_params:
            block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                  idx, original_frame.shape)
            sub_result = adaptive_thresh_with_rules(block.frame, block, d)
            sub_results.append(sub_result)
            # shape = sub_result.binary.shape
            # mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
            # mask[100:900, :] = 255
            # sub_result = adaptive_thresh_mask_no_filter(block.frame, mask, block, d)

            # sub_result = detect_based_task(block, d)
            for rect in sub_result.rects:
                rects.append(rect)
        return rects, sub_results

    def detect_video(self, video_path):
        result_set = []
        video_capture = cv2.VideoCapture(video_path)
        ret, frame = video_capture.read()
        init = False
        idx = 0
        video_writer = None
        while ret:
            temp = []
            rects, sub_results = self.detect_frame(frame, idx, video_path)
            sub_results = sub_results[0]
            if not init:
                dirname = os.path.dirname(video_path)
                filename, extention = os.path.splitext(os.path.basename(video_path))
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out_path = os.path.join(dirname, f'{filename}_binary.mp4')
                shape = sub_results.binary.shape
                video_writer = cv2.VideoWriter(out_path, fourcc, 25, (shape[1], shape[0]))
                init = True
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(sub_results.binary, cv2.COLOR_GRAY2BGR))
            for rect in rects:
                if rect[2] > 15 and rect[3] > 15 and 100 < rect[1] < 900:
                    temp.append(rect)
            if len(temp) > 0:
                result_set.append((idx, temp))
            # logger.info(f'idx={idx}, temp={temp}, len_rects={len(rects)}')
            ret, frame = video_capture.read()
            idx += 1
        if video_writer is not None:
            video_writer.release()
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

    def post_filter_video(self, video_path, task_cnt):
        """
        input a video to filter object
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
        return self.filter_by_speed_and_continuous_time(result_set, task_cnt, video_path)

    def filter_by_speed_and_continuous_time(self, result_set, task_cnt, video_path=None):
        """
        filter thing notified by detection signal, if it's(their) speeds or continuous time
        is over threshold, will be abandoned by algorithm.
        Dolphins'appear time is mostly in 1s~2s, their motion speeds are usually less 15 pixels/s(tested in 1080P monitor),
        but floats thing are much longer, birds or insect are much fast than dolphins
        :param result_set: is set of bbox rects from a batch of video frames ,
        [[(frame_idx1,[x1,y1,x2,y2]),(frame_idx2,[x2,y2,x3,y3]),...],].Rects in a frame are multiple and complicated according to
        diversity of candidates extraction algorithm.The result set could be produced by object
        tracker or detection algorithm.
        :param task_cnt: logger need it.
        :param video_path: None otherwise input a video file.
        :return: True/False indicates current result is reliable or not.
        """
        continuous_time_set = []
        speed_x_set = []
        speed_y_set = []
        continuous_time = 0
        for i in range(1, len(result_set)):
            pre_idx, pre_rects = result_set[i - 1]
            current_idx, current_rects = result_set[i]
            # rects must be corresponding between adjacent video frames
            if len(pre_rects) == len(current_rects):
                if abs(current_idx - pre_idx) <= self.disappear_frames_thresh:
                    continuous_time += abs(current_idx - pre_idx)
                else:
                    continuous_time_set.append(continuous_time)
                    continuous_time = 0
                for j in range(len(pre_rects)):
                    pre_rect = pre_rects[j]
                    current_rect = current_rects[j]
                    pre_center_x = (pre_rect[0] + pre_rect[2]) / 2
                    pre_center_y = (pre_rect[1] + pre_rect[3]) / 2
                    current_center_x = (current_rect[0] + current_rect[2]) / 2
                    current_center_y = (current_rect[1] + current_rect[3]) / 2
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
        elif max_continuous_time > self.continuous_time_thresh:
            logger.info(
                f'Post filter [{self.cfg.index}, {task_cnt}]: detect float in [{video_path}]')
            return True
        else:
            return False

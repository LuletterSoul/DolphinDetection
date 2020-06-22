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

import math
import os
import threading
import time
from dataclasses import dataclass
from multiprocessing import Manager
from multiprocessing.queues import Queue
from typing import List

import cv2
import numpy as np
from queue import Empty
from scipy.optimize import leastsq

from config import ServerConfig, ModelType
from config import SystemStatus
from config import VideoConfig
from .detect_funcs import adaptive_thresh_with_rules
from .params import DetectorParams, DispatchBlock
from pysot.tracker.service import TrackRequester
from stream.rtsp import FFMPEG_MP4Writer
# from .manager import DetectorController
from stream.websocket import creat_packaged_msg_json, creat_detect_msg_json, creat_detect_empty_msg_json
from utils import bbox_points, generate_time_stamp, get_local_time
from utils import paint_chinese_opencv
from utils import preprocess, crop_by_se, logger
from utils.cache import SharedMemoryFrameCache


class ArrivalMsgType:
    UPDATE = 1
    DETECTION = 2


@dataclass
class ArrivalMessage(object):
    """
    Arrival message encapsulation
    """
    current_index: int
    type: int  # message type See ArriveMsgType
    no_wait: bool = False  # ignore window lock
    rects: List = None  # potential bbox


class FrameArrivalHandler(object):
    """
    execute some tasks when fixed frames arrive
    """

    def __init__(self, cfg: VideoConfig, scfg: ServerConfig, detect_index, future_frames, msg_queue: Queue,
                 rect_stream_path, original_stream_path, render_rect_cache, original_frame_cache,
                 notify_queue, region_path, preview_path=None, detect_params=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.scfg = scfg
        self.detect_index = detect_index
        self.rect_stream_path = rect_stream_path
        self.original_stream_path = original_stream_path
        self.preview_path = preview_path
        self.task_cnt = 0
        self.index = cfg.index
        self.cache_size = cfg.cache_size
        self.future_frames = future_frames
        self.sample_rate = cfg.sample_rate
        # self.render_frame_cache = render_frame_cache
        self.render_rect_cache = render_rect_cache
        self.original_frame_cache: SharedMemoryFrameCache = original_frame_cache
        # shared event lock, will be released until all future frames has come
        self.lock_window = Manager().Event()
        self.lock_window.set()
        self.msg_queue = msg_queue
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.quit = Manager().Event()
        self.quit.clear()
        self.status = Manager().Value('i', SystemStatus.RUNNING)
        self.notify_queue = notify_queue
        self.LOG_PREFIX = f'Frame Arrival Handler [{self.cfg.index}]: '
        self.post_filter = Filter(self.cfg, region_path, detect_params)
        self.last_detection = time.time()  # record the last task triggered time.
        self.pre_candidate_rect = []  # record the last rects seed for detection or tracking
        self.task_msg_queue = Manager().Queue()
        self.dis_thresh = max(self.cfg.shape[0], self.cfg.shape[1]) * 1 / 4

    def is_window_reach(self, detect_index):
        return detect_index - self.detect_index > self.future_frames

    def reset(self, msg: ArrivalMessage):
        """
        release window lock
        all frames in a fixed slide window will be skipped until the final frame arrival
        :param msg:
        :return:
        """
        if self.is_window_reach(msg.current_index):
            self.detect_index = msg.current_index
            # self.lock_window = False
            if not self.lock_window.is_set():
                self.lock_window.set()
                # some messages may be cached when new candidates appear, will be post to task handler when
                # current window released(all future frames have arrive.)
                while not self.task_msg_queue.empty():
                    msg = self.task_msg_queue.get()
                    self.task(msg)
            # logger.info(self.LOG_PREFIX + 'Release window lock')

    def next_st(self, detect_index):
        if detect_index - self.detect_index > self.future_frames:
            return detect_index
        else:
            return self.detect_index

    def notify(self, msg: ArrivalMessage):
        """
        execute some analysis tasks asynchronously before the the future frame comes
        :param msg:
        :return:
        """
        # if not self.lock_window:
        # continuous arrival signal in current window will be ignored
        if self.lock_window.is_set() or msg.no_wait:
            self.lock_window.clear()
            # skipped this detection if it is closed enough to the previous one
            # avoid generating videos frequently
            # It's common that the river situation becomes terrible if frequent
            # detections were triggered.
            if self.cfg.limit_freq and time.time() - self.last_detection < self.cfg.freq_thresh:
                logger.info(
                    self.LOG_PREFIX + f'Detection frequency: {round(time.time() - self.last_detection, 2)} is lower than the thresh,ignored.')
                self.detect_index = msg.current_index
                self.last_detection = time.time()
                return
            # occupy the whole window until a sequent task is done
            self.update_record(msg)
            self.task(msg)
        # new candidate may appear during a window
        else:
            new_rects = self.cal_potential_new_candidate(msg)
            if len(new_rects):
                self.update_record(msg)
                msg.rects = new_rects
                # self.task(msg)
                # put into msg queue, waiting processed until window released.
                self.task_msg_queue.put(msg)
                logger.info(
                    self.LOG_PREFIX + f'Appear new candidate during a window, frame index: {msg.current_index}, {new_rects}.')

    def cal_potential_new_candidate(self, msg):
        if msg.rects is None:
            return []
        else:
            new_rects = []
            for old in self.pre_candidate_rect:
                for new in msg.rects:
                    if Obj.cal_dst(old, new) > self.dis_thresh:
                        logger.info(f'Distance betweent new rect and old rect: [{Obj.cal_dst(old, new)}]')
                        new_rects.append(new)
            return new_rects

    def update_record(self, msg):
        self.detect_index = msg.current_index
        self.last_detection = time.time()
        self.pre_candidate_rect = msg.rects

    def task(self, msg: ArrivalMessage):
        """
        override by subclass,do everything what you want in a single frame window
        task must be executed asynchronously  in case blocking caller
        :param msg: arrival msg
        :return: return immediately
        """
        pass

    def wait(self, task_cnt, task_type, msg: ArrivalMessage):
        """
        wait frames arrival,support time out
        :param msg:
        :param task_cnt:
        :param task_type:
        :return:
        """
        if not self.lock_window.is_set():
            logger.debug(
                f'{self.LOG_PREFIX} {task_type} [{task_cnt}] wait frames arrival....')
            start = time.time()
            # wait the future frames prepared,if trigger time out, give up waits
            if not msg.no_wait:
                self.lock_window.wait(30)
            logger.debug(
                f"{self.LOG_PREFIX} {task_type} " +
                f"[{task_cnt}] wait [{round(time.time() - start, 2)}] seconds")
            logger.debug(f'{self.LOG_PREFIX} Rect Render Task [{task_cnt}] frames accessible...')

    def listen(self):
        if self.quit.wait():
            self.lock_window.set()
            self.status.set(SystemStatus.SHUT_DOWN)

    def loop(self):
        """
        loop message inside a sub-process
        :return:
        """
        logger.info(
            f'*******************************{self.LOG_PREFIX}: Init Frame Arrival Handle Service********************************')
        threading.Thread(target=self.listen, daemon=True).start()
        while self.status.get() == SystemStatus.RUNNING:
            try:
                # index, type = self.notify_queue.get()
                msg: ArrivalMessage = self.notify_queue.get()
                if msg.type == ArrivalMsgType.DETECTION:
                    self.notify(msg)
                if msg.type == ArrivalMsgType.UPDATE:
                    self.reset(msg)
            except Empty as e:
                # task service will timeout if track request is empty in pipe
                # ignore
                pass
            except Exception as e:
                logger.error(e)
        logger.info(
            f'*******************************{self.LOG_PREFIX}: Exit Frame Arrival Handle Service********************************')


class DetectionStreamRender(FrameArrivalHandler):
    """
    Generate a video with fixed time when notification occurs
    """

    def __init__(self, cfg: VideoConfig, scfg: ServerConfig, detect_index, future_frames, msg_queue: Queue,
                 rect_stream_path,
                 original_stream_path, render_frame_cache, original_frame_cache, notify_queue,
                 region_path, preview_path=None, detect_params=None) -> None:
        super().__init__(cfg, scfg, detect_index, future_frames, msg_queue, rect_stream_path, original_stream_path,
                         render_frame_cache, original_frame_cache, notify_queue, region_path, preview_path,
                         detect_params)

    def task(self, msg: ArrivalMessage):
        """
        generate a video in fixed window,if detection signal is triggered in a window internal,the rest detection
        frame will be merged into a video instead of producing multiple videos.
        :param msg:
        :return:
        """
        current_idx = msg.current_index
        current_time = generate_time_stamp('%m%d%H%M%S')
        post_filter_event = threading.Event()
        post_filter_event.clear()
        # use two independent threads to execute video generation sub-tasks
        rect_render_thread = threading.Thread(
            target=self.rect_render_task,
            args=(current_idx, current_time, post_filter_event, msg,), daemon=True)
        rect_render_thread.start()
        original_render_thread = threading.Thread(
            target=self.original_render_task,
            args=(current_idx, current_time, post_filter_event, msg,), daemon=True)
        original_render_thread.start()
        self.task_cnt += 1
        return True

    def write_render_video_work(self, video_write, next_cnt, end_cnt):
        """
        write rendering frame into a video, render bbox if rect cache exists the position record
        :param video_write:
        :param next_cnt: video start frame index
        :param end_cnt:
        :return:
        """
        if next_cnt < 1:
            next_cnt = 1
        render_cnt = 0
        rects = []
        for index in range(next_cnt, end_cnt + 1):
            frame = self.original_frame_cache[index].copy()
            if frame is None:
                print(f'Frame is none for [{index}]')
                continue
            # tmp_rects = self.render_rect_cache[index % self.cache_size]
            # self.render_rect_cache[index % self.cache_size] = None
            # if current frame has bbox, just update the bbox position, and clear counting
            # if tmp_rects is not None and len(tmp_rects):
            #    render_cnt = 0
            #    rects = tmp_rects
            if index in self.render_rect_cache:
                render_cnt = 0
                rects = self.render_rect_cache[index]

            # each bbox will last 1.5s in 25FPS video
            logger.debug(self.LOG_PREFIX + f'Render rect frame idx {index}, rects {rects}')
            is_render = render_cnt <= 36
            if is_render:
                for rect in rects:
                    color = np.random.randint(0, 255, size=(3,))
                    color = [int(c) for c in color]
                    # get a square bbox, the real bbox of width and height is universal as 224 * 224 or 448 * 448
                    p1, p2 = bbox_points(self.cfg, rect, frame.shape)
                    # write text
                    # frame = paint_chinese_opencv(frame, '江豚', p1)
                    cv2.rectangle(frame, p1, p2, color, 2)
                    if self.scfg.detect_mode != ModelType.CLASSIFY and len(rect) >= 5:
                        cv2.putText(frame, str(round(rect[4], 2)), (p2[0], p2[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
                render_cnt += 1
            next_cnt += 1
            video_write.write(frame)
        return next_cnt

    def write_original_video_work(self, video_write, next_cnt, end_cnt):
        """
        write original frame into video.
        :param video_write: write video into a pipe, which is directed to ffmpeg command
        :param next_cnt:
        :param end_cnt:
        :return:
        """
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
                frame = self.original_frame_cache[next_cnt % self.cache_size]
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

    def rect_render_task(self, current_idx, current_time, post_filter_event, msg: ArrivalMessage):
        """
        generate a short-time dolphin video with bbox indicator.
        :param current_idx: start frame index in a slide window
        :param current_time:
        :param post_filter_event:
        :param msg: arrival message
        :return:
        """
        start = time.time()
        task_cnt = self.task_cnt
        # raw_target = self.original_stream_path / (current_time + str(self.task_cnt) + '_raw' + '.mp4')
        # target = self.rect_stream_path / (current_time + str(task_cnt) + '.mp4')
        target = self.rect_stream_path / f'{current_time}_{self.cfg.index}_{str(task_cnt)}.mp4'
        logger.debug(
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
        self.wait(task_cnt, 'Rect Render Task', msg)
        end_cnt = current_idx + self.future_frames
        try:
            next_cnt = self.write_render_video_work(video_write, next_cnt, end_cnt)
        except Exception as e:
            logger.error(e)
        video_write.release()
        logger.info(
            f'Video Render [{self.index}]: Rect Render Task [{task_cnt}]: Consume [{round(time.time() - start, 2)}] ' +
            f'seconds.Done write detection stream frame into: [{str(target)}]')
        preview_photo = self.original_frame_cache[current_idx]
        preview_photo_path = self.preview_path / f'{current_time}_{self.cfg.index}_{str(task_cnt)}.jpg'
        cv2.imwrite(str(preview_photo_path), cv2.cvtColor(preview_photo, cv2.COLOR_RGB2BGR))
        self.post_handle(current_time, post_filter_event, target, task_cnt, preview_photo_path)
        # if msg.no_wait:
        # release lock status
        # self.original_frame_cache.release()
        # logger.info('Release original frame cache')

    def original_render_task(self, current_idx, current_time, post_filter_event, msg: ArrivalMessage):
        """
        generate a short-time dolphin video without bbox indicator.
        :param msg:
        :param post_filter_event:
        :param current_idx:
        :param current_time:
        :return:
        """
        start = time.time()
        post_filter_event.clear()
        task_cnt = self.task_cnt
        # raw_target = self.original_stream_path / (current_time + str(self.task_cnt) + '_raw' + '.mp4')
        target = self.original_stream_path / f'{current_time}_{self.cfg.index}_{str(task_cnt)}.mp4'
        logger.debug(
            f'Video Render [{self.index}]: Original Render Task [{task_cnt}]: Writing detection stream frame into: [{str(target)}]')
        # video_write = cv2.VideoWriter(str(raw_target), self.fourcc, 24.0, (self.cfg.shape[1], self.cfg.shape[0]), True)
        video_write = FFMPEG_MP4Writer(str(target), (self.cfg.shape[1], self.cfg.shape[0]), 25)
        # if not video_write.isOpened():
        #     logger.error(f'Video Render [{self.index}]: Error Opened Video Writer')

        next_cnt = current_idx - self.future_frames
        next_cnt = self.write_original_video_work(video_write, next_cnt, current_idx)
        self.wait(task_cnt, 'Original Render Task', msg)
        end_cnt = next_cnt + self.future_frames
        next_cnt = self.write_original_video_work(video_write, next_cnt, end_cnt)
        video_write.release()
        logger.debug(
            f'Video Render [{self.index}]: Original Render Task [{task_cnt}]: ' +
            f'Consume [{round(time.time() - start, 2)}] seconds.Done write detection stream frame into: [{str(target)}]')
        # notify post filter can begin its job
        post_filter_event.set()

    def post_handle(self, current_time, post_filter_event, target, task_cnt, preview):
        """
        post process for each generated video.Can do post filter according its timing information
        :param current_time:
        :param post_filter_event: sync original video generation event
        :param target: video path
        :param preview: preview photo path
        :param task_cnt: current video counting
        :return:
        """
        origin_video_path = self.original_stream_path / (current_time + str(task_cnt) + '.mp4')
        if self.cfg.post_filter:
            self.do_post_filter(origin_video_path, preview, task_cnt, post_filter_event)
        else:
            msg_json = creat_packaged_msg_json(filename=str(target.name), path=str(target), cfg=self.cfg,
                                               camera_id=self.cfg.camera_id, channel=self.cfg.channel,
                                               preview_name=str(preview.name))
            self.msg_queue.put(msg_json)
            logger.info(self.LOG_PREFIX + f'Send packaged message: {msg_json} to msg_queue...')

    def do_post_filter(self, target, preview, task_cnt, post_filter_event):
        """
        execute post filter
        :param preview:
        :param target:
        :param task_cnt:
        :param post_filter_event:
        :return:
        """

        # blocked until original video generation is done.
        post_filter_event.wait()
        # execute post filter
        is_contain_dolphin, dol_rects = self.post_filter.post_filter_video(str(target), task_cnt)
        if is_contain_dolphin:
            """
            post filter think it is a video clip with dolphin
            """
            msg_json = creat_packaged_msg_json(filename=str(target.name), path=str(target), cfg=self.cfg,
                                               camera_id=self.cfg.camera_id, channel=self.cfg.channel,
                                               preview_name=preview.name)
            self.msg_queue.put(msg_json)
            logger.info(self.LOG_PREFIX + f'Send packaged message: {msg_json} to msg_queue...')


class Obj(object):
    def __init__(self, index, cfg: VideoConfig):
        self.index = index  # 当前物体编号
        self.category = None  # 当前物体类别
        self.status = True  # True：仍需要追踪， False:停止追踪
        self.trace = []  # 用来存储当前物体的轨迹rect，每个rect=(idx, [x1, y1, x2, y2])
        self.cfg = cfg
        self.mid_avg_speed = -1
        self.continue_idx_sum = 0

    @staticmethod
    def cal_dst(rect1, rect2):
        rect1_center_x = (rect1[0] + rect1[2]) / 2.0
        rect1_center_y = (rect1[1] + rect1[3]) / 2.0
        rect2_center_x = (rect2[0] + rect2[2]) / 2.0
        rect2_center_y = (rect2[1] + rect2[3]) / 2.0
        return math.sqrt(math.pow(rect1_center_x - rect2_center_x, 2) + math.pow(rect1_center_y - rect2_center_y, 2))

    @staticmethod
    def get_median(data):
        if len(data) == 0:
            return 0
        data.sort()
        half = len(data) // 2
        return (data[half] + data[~half]) / 2

    @staticmethod
    def cal_area(rect):
        w, h = abs(rect[0] - rect[2]), abs(rect[1] - rect[3])
        return w * h

    @staticmethod
    def linear_func(param, x):
        a, b = param
        return a * x + b

    def linear_error(self, param, x, y):
        return self.linear_func(param, x) - y

    def solve_linear_lsq(self, x, y):
        p0 = np.array([0, 0])
        param = leastsq(self.linear_error, p0, args=(x, y))
        return param

    @staticmethod
    def quadratic_func(param, x):
        a, b, c = param
        return a * x * x + b * x + c

    def quadratic_error(self, param, x, y):
        return self.quadratic_func(param, x) - y

    def solve_quadratic_lsq(self, x, y):
        p0 = np.array([10, 10, 10])
        param = leastsq(self.quadratic_error, p0, args=(x, y))
        return param

    def is_quadratic_with_negative_a(self, area_list):
        if len(area_list) < 3:
            return False
        x = np.array(range(len(area_list)))
        y = np.array(area_list)
        param = self.solve_quadratic_lsq(x, y)
        a, b, c = param[0]
        if a < 0:
            return True
        else:
            return False

    def get_last_rect(self):
        return self.trace[len(self.trace) - 1]

    def append_with_rules(self, idx, rect):
        """
         根据现有轨迹判断找到的最近邻rect是否应该加入轨迹中
        """
        if not self.status:
            # 此物体已停止追踪，不加入
            return False
        if len(self.trace) == 0:
            # 初始化
            self.trace.append((idx, rect))
            return True
        else:
            idx1, rect1 = self.get_last_rect()
            if abs(idx1 - idx) > self.cfg.alg['disappear_frames_thresh']:
                # 超过3帧未出现，停止追踪此物体
                self.status = False
                # self.print_obj()
                # print(f'append idx={idx}, rect={rect}, idx1-idx>3')
                return False
            avg_dst = self.cal_dst(rect1, rect) / abs(idx1 - idx)
            if avg_dst <= 200:
                # 防止追踪剧烈闪屏
                self.trace.append((idx, rect))
                return True
            else:
                # self.print_obj()
                # print(f'append idx={idx}, rect={rect}, dst={dst}, dst>200')
                return False

    def fitting_horizontal_line(self, float_trace):
        """
        self.trace 与 float trace拟合直线，判断直线是否是近似的水平线
        :param float_trace: 与self.trace格式相同,[(idx1, rect1), (idx2, rect2), ...]
        :return: True->拟合出了近似水平线，False->没有拟合出近似水平线
        """
        x = []
        y = []
        for idx, rect in float_trace:
            x.append((rect[0] + rect[2]) / 2.0)
            y.append((rect[1] + rect[3]) / 2.0)
        for idx, rect in self.trace:
            x.append((rect[0] + rect[2]) / 2.0)
            y.append((rect[1] + rect[3]) / 2.0)
        if len(x) <= 1:
            # 数据太少，无法拟合
            return False
        x = np.array(x)
        y = np.array(y)
        param = self.solve_linear_lsq(x, y)
        a, b = param[0]
        # 转化斜率为角度
        radian = math.atan(a)  # 弧度
        angle = (180 * radian) / math.pi  # 角度
        logger.info(f'Line fitting: a: {a}, b: {b}, angle: {round(angle, 2)}')
        if abs(angle) < self.cfg.alg['angle_thresh']:
            logger.info(f'Line fitting: angle is below the thresold, fitting successful.')
            return True
        else:
            return False

    def is_match_with_history_float(self, float_trace_list):
        if len(float_trace_list) == 0:
            return False
        for float_trace in float_trace_list:
            if self.fitting_horizontal_line(float_trace):
                return True
        return False

    def predict_category(self, float_trace_list):
        """
        通过分析当前物体轨迹，预测当前物体类别，分析准则有：
        1、物体速度  速度过快是飞鸟
        2、物体持续时间  持续时间过长是漂浮物
        # 3、物体面积变化曲线 江豚面积变化曲线大致为开口向下的抛物线
        4、物体与前一个滑动窗口中的漂浮物拟合直线，如近似拟合出一条水平线，则认为是漂浮物
        :return:
        """
        # if len(self.trace) < 4:
        #     # 轨迹长度过短，无法分析物体类别
        #     self.category = 'Unknown'
        #     return
        avg_speed_list = []
        continue_idx_sum = 0
        dst_sum = 0
        area_list = [self.cal_area(self.trace[0][1])]
        for i in range(1, len(self.trace)):
            pre_idx, pre_rect = self.trace[i - 1]
            current_idx, current_rect = self.trace[i]
            continue_idx_sum += abs(current_idx - pre_idx)
            area_list.append(self.cal_area(current_rect))
            dst = self.cal_dst(pre_rect, current_rect)
            dst_sum += dst
            avg_speed = dst / abs(pre_idx - current_idx)
            avg_speed_list.append(avg_speed)
        mid_avg_speed = self.get_median(avg_speed_list)
        self.mid_avg_speed = mid_avg_speed
        self.continue_idx_sum = continue_idx_sum
        if mid_avg_speed > self.cfg.alg['speed_x_thresh']:
            self.category = 'bird'
        elif continue_idx_sum > self.cfg.alg['continuous_time_thresh']:
            self.category = 'float'
        # elif self.is_quadratic_with_negative_a(area_list):
        #    self.category = 'dolphin'
        # else:
        elif self.is_match_with_history_float(float_trace_list):
            self.category = 'float'
        elif continue_idx_sum <= 1:
            self.category = 'Unknown'
        else:
            self.category = 'dolphin'


class DetectionSignalHandler(FrameArrivalHandler):
    """
    track object from a batch of frames, and decides if let system
    send detection notification,generate render video or not
    """

    def __init__(self, cfg: VideoConfig, scfg: ServerConfig, detect_index, future_frames, msg_queue: Queue,
                 rect_stream_path,
                 original_stream_path, render_frame_cache, original_frame_cache, notify_queue,
                 region_path, preview_path, track_requester: TrackRequester, render_queue: Queue,
                 detect_params=None) -> None:
        super().__init__(cfg, scfg, detect_index, future_frames, msg_queue, rect_stream_path, original_stream_path,
                         render_frame_cache, original_frame_cache, notify_queue, region_path, preview_path,
                         detect_params)
        self.track_requester = track_requester
        self.LOG_PREFIX = f'Detection Signal Handler [{self.cfg.index}]: '
        self.render_queue = render_queue
        self.dol_id = 10000
        self.detect_num = 0
        self.post_num = 0
        self.LOG_PREFIX = f'Detection Signal Handler [{self.cfg.index}]: '

    def listen(self):
        if self.quit.wait():
            self.lock_window.set()
            self.status.set(SystemStatus.SHUT_DOWN)
            logger.info(
                f'{self.LOG_PREFIX}: All Statistic: detection number: {self.detect_num}, post number: {self.post_num}, '
                f'filter rate: {round((abs(self.detect_num - self.post_num) / self.detect_num), 4) * 100}%')

    def task(self, msg: ArrivalMessage):
        """
        generate a video in fixed window,if detection signal is triggered in a window internal,the rest detection
        frame will be merged into a video instead of producing multiple videos.
        :param msg: arrival msg
        :return:
        """
        current_idx = msg.current_index
        current_time = generate_time_stamp('%m%d%H%M%S') + '_'
        # use two independent threads to execute video generation sub-tasks
        handle_thread = threading.Thread(
            target=self.handle,
            args=(msg, current_time,), daemon=True)
        handle_thread.start()
        return True

    def handle(self, msg: ArrivalMessage, current_time):
        """
        handle detection signal
        :param msg:
        :param current_time:
        :return:
        """
        current_index = msg.current_index
        if not self.cfg.forward_filter:
            logger.info(self.LOG_PREFIX + f'The signal forward operation is disabled by configuration.')
        self.wait(self.task_cnt, 'Detection Signal Handle', msg)
        # rects = self.render_rect_cache[current_index % self.cache_size]
        # lock the whole window in case cached was covered by the future arrival frames.
        # self.original_frame_cache.lock_cache(current_index - self.cfg.future_frames,
        #                                      current_index + self.cfg.search_window_size)
        rects = msg.rects
        task_cnt = self.task_cnt
        if rects is not None:
            track_start = time.time()
            result_sets, _ = self.track_requester.request(self.cfg.index, current_index, rects)
            # is_filter = self.post_filter.filter_by_speed_and_continuous_time(result_sets, task_cnt)
            self.detect_num += 1
            is_contain_dolphin, traces = self.post_filter.filter_by_obj_match_analyze(result_sets, task_cnt)
            track_end = time.time()
            track_consume = track_end - track_start
            logger.info(f'{self.LOG_PREFIX}: Filter result: contain dolphin: {is_contain_dolphin}')
            if is_contain_dolphin:
                self.trigger_rendering(current_index, traces, track_consume)
                self.task_cnt += 1
                self.post_num += 1

    def trigger_rendering(self, current_index, traces, time_consume):
        """
        trigger rendering and ignores the window lock without waits.
        :param current_index:
        :param traces:
        :return:
        """
        self.write_bbox(traces, time_consume)
        # send message via message pipe
        self.render_queue.put(ArrivalMessage(current_index, ArrivalMsgType.DETECTION, True))
        self.render_queue.put(ArrivalMessage(current_index, ArrivalMsgType.UPDATE))

    def write_bbox(self, traces, time_consume):
        # cnt = 0
        # self.render_rect_cache[:] = [None] * self.cfg.cache_size
        for frame_idx, rects in traces.items():
            # old_rects = self.render_rect_cache[frame_idx % self.cache_size]
            if frame_idx in self.render_rect_cache:
                old_rects = self.render_rect_cache[frame_idx]
                self.render_rect_cache[frame_idx] = old_rects + rects
            else:
                self.render_rect_cache[frame_idx] = rects
            json_msg = creat_detect_msg_json(video_stream=self.cfg.rtsp, channel=self.cfg.channel,
                                             timestamp=get_local_time(time_consume), rects=rects, dol_id=self.dol_id,
                                             camera_id=self.cfg.camera_id, cfg=self.cfg)
            self.msg_queue.put(json_msg)
            # bbox rendering is post to render
            logger.debug(f'put detect message in msg_queue {json_msg}...')
            # print(rects)
        empty_msg = creat_detect_empty_msg_json(video_stream=self.cfg.rtsp,
                                                channel=self.cfg.channel,
                                                timestamp=get_local_time(time_consume), dol_id=self.dol_id,
                                                camera_id=self.cfg.camera_id)
        self.dol_id += 1
        self.msg_queue.put(empty_msg)


class Filter(object):
    """
    filter post rendering video from a video, or forward detection signal from a bbox set.
    """

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
        self.float_trace_list = []

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
            # TODO we have to use a more robust frontground extraction algorithm get the binary map
            #  adaptive thresh to get binay map is just a compromise,
            #  it don't work if river background is complicated
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
        """
        iterates the whole shot-time video and executes analysis algorithm for timing information
        :param video_path:
        :return:
        """
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
                # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                # out_path = os.path.join(dirname, f'{filename}_binary.mp4')
                # shape = sub_results.binary.shape
                # video_writer = cv2.VideoWriter(out_path, fourcc, 25, (shape[1], shape[0]))
                init = True
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(sub_results.binary, cv2.COLOR_GRAY2BGR))
            for rect in rects:
                if rect[2] - rect[0] > 15 and rect[3] - rect[1] > 15 and 100 < rect[1] < 900:
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

    @staticmethod
    def get_median(data):
        if len(data) == 0:
            return 0
        data.sort()
        half = len(data) // 2
        return (data[half] + data[~half]) / 2

    @staticmethod
    def get_max_continuous_time(data):
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
        # return self.filter_by_speed_and_continuous_time(result_set, task_cnt, video_path)
        return self.filter_by_obj_match_analyze(result_set, task_cnt, video_path)

    def filter_by_speed_and_continuous_time(self, result_set, task_cnt, video_path=None):
        """
        filter thing notified by detection signal, if it's(their) speeds or continuous time
        is over threshold, will be abandoned by algorithm.
        Dolphins'appear time is mostly in 1s~2s, their motion speeds are usually less 15 pixels/s
        (tested in 1080P monitor), but floats thing are much longer, birds or insect are much fast than dolphins.
        :param result_set: is set of bbox rects from a batch of video frames ,
        [(frame_idx1,[[x1,y1,x2,y2],[x3,y3,x4,y4],...]),(frame_idx2,[[x5,y5,x6,y6],[x7,y7,x8,y8],...]),...)].
        Rects in a frame are multiple and complicated according to
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
        if len(result_set) < 2:
            return True
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

    def filter_by_obj_match_analyze(self, result_set, task_cnt, video_path=None):
        obj_list = self.get_obj_list_from_result_set(result_set)
        flag = False
        traces = {}
        i = 0
        new_float_trace = []
        for obj in obj_list:
            obj.predict_category(self.float_trace_list)
            if obj.category == 'float':
                new_float_trace.append(obj.trace)
            logger.info(
                f'Post filter [{self.cfg.index}, {task_cnt}]:[{obj.index}th] obj is '
                f'[{obj.category}],avg speed [{obj.mid_avg_speed}], continuous time [{obj.continue_idx_sum}], category [{obj.category}], trace {obj.trace}')
            if obj.category == 'dolphin':
                flag = True
                for frame_idx, rect in obj.trace:
                    if frame_idx not in traces:
                        traces[frame_idx] = [rect]
                    else:
                        traces[frame_idx].append(rect)
        self.float_trace_list = new_float_trace.copy()
        return flag, traces

    @staticmethod
    def cal_dst(rect1, rect2):
        rect1_center_x = (rect1[0] + rect1[2]) / 2.0
        rect1_center_y = (rect1[1] + rect1[3]) / 2.0
        rect2_center_x = (rect2[0] + rect2[2]) / 2.0
        rect2_center_y = (rect2[1] + rect2[3]) / 2.0
        return math.sqrt(
            math.pow(rect1_center_x - rect2_center_x, 2) + math.pow(rect1_center_y - rect2_center_y, 2))

    def find_nearest_rect(self, obj_last_rect, rects):
        min_pos = 0
        min_dst = self.cal_dst(obj_last_rect, rects[0])
        for i in range(1, len(rects)):
            dst = self.cal_dst(obj_last_rect, rects[i])
            if dst < min_dst:
                min_pos = i
                min_dst = dst
        return min_pos

    def get_obj_list_from_result_set(self, result_set):
        obj_list = []
        if len(result_set) == 0:
            return obj_list
        elif len(result_set) == 1:
            idx, rects = result_set[0]
            for rect in rects:
                obj = Obj(len(obj_list), self.cfg)
                obj.append_with_rules(idx, rect)
                obj_list.append(obj)
            return obj_list
        else:
            idx, rects = result_set[0]
            for rect in rects:
                obj = Obj(len(obj_list), self.cfg)
                obj.append_with_rules(idx, rect)
                obj_list.append(obj)
            for i in range(1, len(result_set)):
                idx, rects = result_set[i]
                flag_list = [False] * len(rects)
                for obj in obj_list:
                    idx_obj, rect_obj = obj.get_last_rect()
                    min_pos = self.find_nearest_rect(rect_obj, rects)
                    if obj.append_with_rules(idx, rects[min_pos]):
                        flag_list[min_pos] = True
                for j in range(len(flag_list)):
                    if not flag_list[j]:
                        rect = rects[j]
                        obj = Obj(len(obj_list), self.cfg)
                        obj.append_with_rules(idx, rect)
                        obj_list.append(obj)
            return obj_list

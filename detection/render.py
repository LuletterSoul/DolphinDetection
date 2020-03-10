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
from utils import logger, bbox_points, generate_time_stamp
from stream.rtsp import FFMPEG_MP4Writer
import traceback


class DetectionStreamRender(object):

    def __init__(self, cfg, detect_index, future_frames, msg_queue: Queue,
                 rect_stream_path, original_stream_path, render_frame_cache, render_rect_cache, original_frame_cache,
                 notify_queue) -> None:
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
        self.cache_size = 5000
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
            index, type = self.notify_queue.get()
            if type == 'notify':
                self.notify(index)
            if type == 'reset':
                self.reset(index)
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
        start = time.time()
        try_times = 0
        while next_cnt < end_cnt:
            try:
                if self.status.get() == SystemStatus.SHUT_DOWN:
                    logger.info(
                        f'Video Render [{self.index}]: render task interruped by exit signal')
                    return next_cnt
                # if next_cnt in render_cache:
                frame = frame_cache[next_cnt % self.cache_size]
                render_frame = rect_cache[next_cnt % self.cache_size]
                if render_frame is not None:
                    next_cnt = self.process_nearest_neighbor_render_frames(next_cnt, end_cnt, frame, rect_cache,
                                                                           render_cache,
                                                                           video_write)
                    # elif next_cnt in frame_cache:
                elif frame is not None:
                    video_write.write(frame)
                    next_cnt += 1
                    # frame_cache[next_cnt % self.cache_size] = None
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
                traceback.print_stack()
        return next_cnt

    def process_nearest_neighbor_render_frames(self, next_cnt, end_cnt, frame, rect_cache, render_cache, video_write):
        forward_cnt = next_cnt + self.sample_rate
        if forward_cnt > end_cnt:
            forward_cnt = end_cnt
        while forward_cnt > next_cnt:
            # if forward_cnt in render_cache:
            if rect_cache[forward_cnt % self.cache_size] is not None:
                break
            forward_cnt -= 1
        # current forward pointer
        if forward_cnt - next_cnt <= 1:
            # logger.info('Enter singal')
            render_frame = render_cache[next_cnt % self.cache_size]
            if render_frame is not None:
                video_write.write(render_frame)
                render_cache[next_cnt % self.cache_size] = None
            next_cnt += 1
            return next_cnt
        elif forward_cnt - next_cnt > 1:
            # logger.info('Enter internal')
            return self.render_internal_frames(forward_cnt, next_cnt, frame, video_write)

    def render_internal_frames(self, forward_cnt, next_cnt, frame, video_write):
        step = forward_cnt - next_cnt
        first_rects = self.render_rect_cache[next_cnt % self.cache_size]
        last_rects = self.render_rect_cache[forward_cnt % self.cache_size]
        self.render_rect_cache[next_cnt % self.cache_size] = None
        self.render_rect_cache[forward_cnt % self.cache_size] = None
        len_is_equal = len(last_rects) != len(first_rects)
        if frame is not None and len_is_equal:
            for i in range(step):
                try:
                    frame = self.original_frame_cache[next_cnt].copy()
                    for j in range(min(len(first_rects), len(last_rects))):
                        first_rect = first_rects[j]
                        last_rect = last_rects[j]
                        delta_x = (last_rect[0] - first_rect[0]) / step
                        delta_y = (last_rect[1] - first_rect[1]) / step
                        if abs(delta_x) > 100 / step or abs(delta_y) > 100 / step:
                            break
                        color = np.random.randint(0, 255, size=(3,))
                        color = [int(c) for c in color]
                        p1, p2 = bbox_points(self.cfg, first_rect, frame.shape, int(delta_x), int(delta_y))
                        cv2.putText(frame, 'Asaeorientalis', p1,
                                    cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, p1, p2, color, 2)
                    video_write.write(frame)
                    next_cnt += 1
                except Exception as e:
                    next_cnt += 1
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
        # raw_target = self.original_stream_path / (current_time + str(self.stream_cnt) + '_raw' + '.mp4')
        target = self.rect_stream_path / (current_time + str(self.stream_cnt) + '.mp4')
        logger.info(
            f'Video Render [{self.index}]: Rect Render Task [{self.stream_cnt}]: Writing detection stream frame into: [{str(target)}]')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # video_write = cv2.VideoWriter(str(raw_target), self.fourcc, 24.0, (self.cfg.shape[1], self.cfg.shape[0]), True)
        video_write = FFMPEG_MP4Writer(str(target), (self.cfg.shape[1], self.cfg.shape[0]), 25)
        next_cnt = current_idx - self.future_frames
        next_cnt = self.write_render_video_work(video_write, next_cnt, current_idx, render_cache, rect_cache,
                                                frame_cache)
        # the future frames count
        # next_frame_cnt = 48
        # wait the futures frames is accessable
        if not self.next_prepare_event.is_set():
            logger.info(
                f'Video Render [{self.index}]: Rect Render Task [{self.stream_cnt}] wait frames accessible....')
            start = time.time()
            # wait the future frames prepared,if ocurring time out, give up waits
            self.next_prepare_event.wait(30)
            logger.info(
                f"Video Render [{self.cfg.index}]: Rect Render Task " +
                f"[{self.stream_cnt}] wait [{round(time.time() - start, 2)}] seconds")
            logger.info(f'Video Render [{self.index}]: Rect Render Task [{self.stream_cnt}] frames accessible...')

        # if not self.started:
        #     return False
        # logger.info('Render task Begin with frame [{}]'.format(next_cnt))
        # logger.info('After :[{}]'.format(render_cache.keys()))
        end_cnt = next_cnt + self.future_frames
        next_cnt = self.write_render_video_work(video_write, next_cnt, end_cnt, render_cache, rect_cache,
                                                frame_cache)
        video_write.release()
        # self.convert_byfile(str(raw_target), str(target))
        logger.info(
            f'Video Render [{self.index}]: Rect Render Task [{self.stream_cnt}]: Consume [{round(time.time() - start, 2)}] ' +
            f'seconds.Done write detection stream frame into: [{str(target)}]')
        msg_json = creat_packaged_msg_json(filename=str(target.name), path=str(target), cfg=self.cfg)
        # if raw_target.exists():
        #     raw_target.unlink()
        self.msg_queue.put(msg_json)
        logger.info(f'put packaged message in the msg_queue...')

    def original_render_task(self, current_idx, current_time, frame_cache):
        start = time.time()
        # raw_target = self.original_stream_path / (current_time + str(self.stream_cnt) + '_raw' + '.mp4')
        target = self.original_stream_path / (current_time + str(self.stream_cnt) + '.mp4')
        logger.info(
            f'Video Render [{self.index}]: Original Render Task [{self.stream_cnt}]: Writing detection stream frame into: [{str(target)}]')
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
                f"Video Render [{self.cfg.index}]: Original Render Task [{self.stream_cnt}] wait [{time.time() - start}] seconds")
            logger.info(f'Video Render [{self.index}]: Original Render Task [{self.stream_cnt}] frames accessible....')
        # logger.info('Render task Begin with frame [{}]'.format(next_cnt))
        # logger.info('After :[{}]'.format(render_cache.keys()))
        # if not self.started:
        #     return False
        end_cnt = next_cnt + self.future_frames
        next_cnt = self.write_original_video_work(video_write, next_cnt, end_cnt, frame_cache)
        video_write.release()
        # self.convert_byfile(str(raw_target), str(target))
        logger.info(
            f'Video Render [{self.index}]: Original Render Task [{self.stream_cnt}]: ' +
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

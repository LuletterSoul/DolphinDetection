#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: pool.py
@time: 3/16/20 10:34 PM
@version 1.0
@desc:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import cv2
import traceback
from queue import Empty
from multiprocessing import Value

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from multiprocessing import Pool, Manager, Queue, Process, Lock
from utils.cache import SharedMemoryFrameCache
from stream.rtsp import FFMPEG_MP4Writer
from typing import List
import torch
from utils import logger, to_bbox_wh
from config import SystemStatus, VideoConfig
import cv2


class TrackRequest(object):
    """
    Post a tracking request
    """

    def __init__(self, request_id, monitor_index, frame_index, rect, rect_id) -> None:
        self.monitor_index = monitor_index
        self.frame_index = frame_index
        self.rect = rect
        self.rect_id = rect_id
        self.request_id = request_id


class TrackResult(object):
    """
    Tracking result
    """

    def __init__(self, rect_id, result) -> None:
        """

        :param rect_id: bbox id
        :param result: list object [[x1,y1,x2,y2],[x2,y2,x3,y3]],
        indicates tracking bbox of each frame
        """
        self.rect_id = rect_id
        self.result = result


def track_service(model_index, video_cfgs, checkpoint,
                  frame_caches,
                  recv_pipe: Queue,
                  output_pipes, status, lock):
    """
    Each track service maintains a tracker model instance, which must be init inside a subprocess
    :param model_index: model index
    :param video_cfgs: input video configurations from top tie
    :param device: cuda device object
    :param checkpoint: tracker model parameters
    :param frame_caches: global video frames cache
    :param recv_pipe: receive tracking request from the other processes,it's multi-processing shared queue
    :param output_pipes: output tracking result into a monitor-specified pipe independently
    :param status: system status, such as SHUT_DOWN,RESUME, RUNNING
    :param lock: gpu lock
    :return:
    """

    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(model_index))
    else:
        device = torch.device('cpu')

    # Build model instance
    model = ModelBuilder()
    tracker = build_tracker(model, device)
    model.load_state_dict(torch.load(checkpoint,
                                     map_location=lambda storage, loc: storage.cpu()))
    # now model is running inside a sub-process
    model.eval().to(device)
    logger.info(
        f'Tracker [{model_index}]: Running Tracker Services: checkpoint from {checkpoint}')
    LOGGER_PREFIX = f'Tracker [{model_index}]: '
    while True:
        try:
            if status.get() == SystemStatus.SHUT_DOWN:
                logger.info(f'Tracker [{model_index}]: Exit Tracker Service')
                break
            # fetch a tracking request from a global sync queue
            # monitor_index, frame_index, rect, rect_id = recv_pipe.get()
            req: TrackRequest = recv_pipe.get(timeout=5)
            # threshold for filtering low confidence bbox
            track_confidence = video_cfgs[req.monitor_index].alg['track_confidence']
            # frame number of each tracking request
            track_window_size = video_cfgs[req.monitor_index].search_window_size
            show_windows = video_cfgs[req.monitor_index].show_window
            logger.info(
                f'Tracker [{model_index}]: From monitor [{req.monitor_index}] track request, track confidence '
                f'[{track_confidence}, track window size [{track_window_size}]')
            init_frame = frame_caches[req.monitor_index][req.frame_index]
            if init_frame is None:
                logger.info(
                    f'Tracker [{model_index}]: Empty frame from cache [{req.frame_index}] of monitor [{req.monitor_index}].')
                continue
            # lock the whole model in case it is busy and throw exception if multiple requests post
            with lock:
                s = time.time()
                tracker.init(init_frame, to_bbox_wh(req.rect))
                end_index = req.frame_index + track_window_size + 1
                result = []
                result.append((req.frame_index, req.rect))
                # fetch frames ASAP in case history caches were covered by the future frames
                frames = []
                for i in range(req.frame_index + 1, end_index):
                    frame = frame_caches[req.monitor_index][i]
                    if frame is None:
                        continue
                    frames.append((i, frame))

                # track in a slice windows
                video_writer = None
                if show_windows:
                    video_writer = FFMPEG_MP4Writer(f'track_{req.monitor_index}_{req.request_id}.mp4',
                                                    (init_frame.shape[1], init_frame.shape[0]),
                                                    25)

                for i, frame in frames:
                    track_res = tracker.track(frame)
                    best_score = track_res['best_score']
                    if best_score > track_confidence:
                        result.append((i, track_res['bbox']))
                        if show_windows:
                            frame = cv2.rectangle(frame.copy(),
                                                  (int(track_res['bbox'][0]), int(track_res['bbox'][1])),
                                                  (int(track_res['bbox'][2]), int(track_res['bbox'][3])),
                                                  color=(0, 0, 255), thickness=3)
                            cv2.namedWindow(f'Track Result {model_index}', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.imshow(f'Track Result {model_index}', frame)
                            cv2.waitKey(1)
                            video_writer.write(frame)

                if show_windows and video_writer is not None:
                    video_writer.release()
                # output results into the corresponding pipe of each monitor
                output_pipes[req.monitor_index][req.request_id] = TrackResult(req.rect_id, result)
                e = time.time() - s
                logger.info(f'{LOGGER_PREFIX} tracking consumes: {round(e, 2)} seconds')
        except Empty as e:
            # task service will timeout if track request is empty in pipe
            # ignore
            pass
        except Exception as e:
            traceback.print_exc()
            # catch Broken pipe exception possibly
            return


class TrackRequester(object):

    def __init__(self, recv_pipe, output_pipes):
        super().__init__()
        self.rec_pipe = recv_pipe
        self.output_pipes = output_pipes
        self.r_id = Manager().Value('i', 0)

    def request(self, monitor_index, frame_index, rects):
        """
        post a tracking request to a GPU model, request will be blocked until tracker finishing
        services for all rects tracking
        :param monitor_index: monitor index
        :param frame_index: frame unique id
        :param rects: rects template
        :return:  res: WINDOW_SIZE * (frame_idx,[[x1,y1,x2,y2],[x3,y3,x4,y4],...](one frame with all rects);
                  seq: N * WINDOW_SIZE * (frame_idx,[x1,y1,x2,y2]); N is rects' number
        """
        res = []
        organize = {}
        seq = [[]] * len(rects)
        req_ids = []
        if not len(rects):
            return [], []
        else:
            for rect_id, rect in enumerate(rects):
                c_id = self.r_id.get()
                self.r_id.set(c_id + 1)
                req_ids.append(c_id)
                # post to a single
                self.rec_pipe.put(TrackRequest(c_id, monitor_index, frame_index, rect, rect_id))
            for i in range(len(rects)):
                # frames of each monitor arrival in order, track request is also in order
                # in corresponding receive pipe
                while req_ids[i] not in self.output_pipes[monitor_index]:
                    logger.debug(f'Wait for tracking result for request id [{req_ids[i]}]')
                    time.sleep(1)
                logger.info(f'Tracking result done for request id [{req_ids[i]}]')
                track_result: TrackResult = self.output_pipes[monitor_index][req_ids[i]]
                # but track result may be out of order at each request, should sort by original rect id
                seq[track_result.rect_id] = track_result.result
                frame_seq = track_result.result
                for f_idx, rect in frame_seq:
                    # logger.info(f'frame index {f_idx} rect {rect}')
                    if f_idx not in organize:
                        organize[f_idx] = []
                    organize[f_idx].append(rect)
            # organize data structure for filter input
            for k, v in organize.items():
                if len(v):
                    res.append((k, v))
            # logger.info(f'tracking results: {result_set}')
        return res, seq


class TrackingService(object):
    """
    Handles tracking requests from all monitor processes,and dispatch all requests into a GPU pool.
    """

    def __init__(self, model_cfg_path, video_configs: List[VideoConfig], checkpoint, frame_caches,
                 size=3) -> None:
        super().__init__()
        if not os.path.exists(model_cfg_path):
            raise Exception('Track model configuration not found.')

        if not os.path.exists(checkpoint):
            raise Exception('Track model checkpoint not found.')

        cfg.merge_from_file(model_cfg_path)
        logger.info(cfg.BACKBONE.TYPE)

        # self.devices = []
        self.models = []
        self.pipe_manager = Manager()
        self.frame_caches = {}
        self.video_configs = {}
        self.output_pipes = {}
        for idx, c in enumerate(video_configs):
            self.frame_caches[c.index] = frame_caches[idx]
            self.video_configs[c.index] = c
            self.output_pipes[c.index] = self.pipe_manager.dict()
            # self.video_configs = video_configs
        self.checkpoint = checkpoint
        # if torch.cuda.is_available():
        #     gpu_num = torch.cuda.device_count()
        #     device_num = min(size, gpu_num)
        #     for i in range(device_num):
        #         self.devices.append(torch.device('cuda:' + str(i)))
        # else:
        #     devices = [torch.device('cpu') for i in range(size)]
        #     devices.append(torch.device('cpu'))
        self.device_num = size
        # self.rec_pipes = [self.pipe_manager.Queue() for i in range(self.device_num)]
        self.rec_pipe = self.pipe_manager.Queue()
        # self.rec_pipe = recv_pipe
        self.status = self.pipe_manager.Value('i', SystemStatus.RUNNING)
        # self.output_pipes = [self.pipe_manager.Queue() for i in range(self.device_num)]
        self.gpu_locks = [Lock() for i in range(self.device_num)]
        # self.tracker_pool = Pool(len(self.devices))
        self.proc_instances = []

    def run(self):
        """
        run tracking post service
        :return:
        """
        for idx in range(self.device_num):
            p = Process(target=track_service,
                        args=(idx, self.video_configs, self.checkpoint, self.frame_caches, self.rec_pipe,
                              self.output_pipes,
                              self.status, self.gpu_locks[idx]),
                        daemon=True)
            p.start()
            self.proc_instances.append(p)

            # self.tracker_pool.

    def get_request_instance(self):
        return TrackRequester(self.rec_pipe, self.output_pipes)

    def cancel(self):
        self.status.set(SystemStatus.SHUT_DOWN)
        for p in self.proc_instances:
            p.join()
            p.close()

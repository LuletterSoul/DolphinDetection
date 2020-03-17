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
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from multiprocessing import Pool, Manager, Queue, Process, Lock
from utils.cache import SharedMemoryFrameCache
from typing import List
import torch
from utils import logger
from config import SystemStatus, VideoConfig


class TrackRequest(object):
    """
    Post a tracking request
    """

    def __init__(self, monitor_index, frame_index, rect, rect_id) -> None:
        self.monitor_index = monitor_index
        self.frame_index = frame_index
        self.rect = rect
        self.rect_id = rect_id


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


def track_service(index, video_cfgs: List[VideoConfig], device, checkpoint,
                  frame_caches: List[SharedMemoryFrameCache],
                  recv_pipe: Queue,
                  output_pipes: List[Queue], status, lock):
    """
    Each track service maintains a tracker model instance, which must be init inside a subprocess
    :param index: model index
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

    # Build model instance
    model = ModelBuilder()
    tracker = build_tracker(model)
    model.load_state_dict(torch.load(checkpoint,
                                     map_location=lambda storage, loc: storage.cpu()))
    # now model is running inside a sub-process
    model.eval().to(device)
    logger.info(
        f'Tracker [{index}]: Running Tracker Services: checkpoint from {checkpoint}')
    while True:
        if status.get() == SystemStatus.SHUT_DOWN:
            logger.info(f'Tracker [{index}]: Exit Tracker Services')
            break
        # fetch a tracking request from a global sync queue
        # monitor_index, frame_index, rect, rect_id = recv_pipe.get()
        req: TrackRequest = recv_pipe.get()
        # threshold for filtering low confidence bbox
        track_confidence = video_cfgs[req.monitor_index].alg['track_confidence']
        # frame number of each tracking request
        track_window_size = video_cfgs[req.monitor_index].alg['search_windows']
        logger.info(
            f'Tracker [{index}]: From monitor [{req.monitor_index}] track request, track confidence '
            f'[{track_confidence}, track window size [{track_window_size}]')
        init_frame = frame_caches[req.monitor_index][req.frame_index]
        if init_frame is None:
            logger.info(
                f'Tracker [{index}]: Empty frame from cache [{req.frame_index}] of monitor [{req.monitor_index}].')
            continue
        # lock the whole model in case it is busy and throw exception if multiple requests post
        with lock:
            tracker.init(init_frame, req.rect)
            end_index = req.frame_index + track_window_size + 1
            result = []
            result.append(req.rect)
            # track in a slice windows
            for i in range(req.frame_index + 1, end_index):
                frame = frame_caches[req.monitor_index][i]
                if frame is None:
                    continue
                track_res = tracker.track(frame)
                best_score = track_res['best_score']
                if best_score > track_confidence:
                    result.append((i, track_res))
            # output results into the corresponding pipe of each monitor
            logger.info(f'Tracker [{index}]: tracking results: {result}')
            output_pipes[req.monitor_index].put(TrackResult(req.rect_id, result))


class TrackingRequestPoster(object):
    """
    Handles tracking requests from all monitor processes,and dispatch all requests into a GPU pool.
    """

    def __init__(self, video_configs: List[VideoConfig], size, checkpoint, frame_caches) -> None:
        super().__init__()
        self.devices = []
        self.models = []
        self.pipe_manager = Manager()
        self.video_configs = video_configs
        self.frame_caches = frame_caches
        if torch.cuda.is_available():
            gpu_num = torch.cuda.device_count()
            device_num = min(size, gpu_num)
            for i in range(device_num):
                self.devices.append('cuda:' + str(i))
        else:
            devices = [torch.device('cpu') for i in range(size)]
            devices.append(torch.device('cpu'))
        self.device_num = len(self.devices)
        # self.rec_pipes = [self.pipe_manager.Queue() for i in range(self.device_num)]
        self.rec_pipe = self.pipe_manager.Queue()
        self.output_pipes = {}
        self.status = self.pipe_manager.Value('i', SystemStatus.RUNNING)
        for c in self.video_configs:
            self.output_pipes[c.index] = self.pipe_manager.Queue()
        # self.output_pipes = [self.pipe_manager.Queue() for i in range(self.device_num)]
        self.gpu_locks = [Lock() for i in range(self.device_num)]
        # self.tracker_pool = Pool(len(self.devices))
        self.proc_instances = []
        for idx, device in enumerate(self.devices):
            p = Process(target=track_service,
                        args=(idx, video_configs, device, checkpoint, frame_caches, self.rec_pipe, self.output_pipes,
                              self.status, self.gpu_locks[idx]),
                        daemon=True)
            p.start()
            self.proc_instances.append(p)

            # self.tracker_pool.

    def request(self, monitor_index, frame_index, rects):
        """
        post a tracking request to a GPU model, request will be blocked until tracker finishing
        services for all rects tracking
        :param monitor_index: monitor index
        :param frame_index: frame unique id
        :param rects: rects template
        :return: result set [[(frame_idx1,[x1,y1,x2,y2]),(frame_idx2,[x2,y2,x3,y3]),...],]
        """
        result_set = [None] * len(rects)
        if not len(rects):
            return []
        else:
            for rect in rects:
                # post to a single
                self.rec_pipe.put(TrackRequest(monitor_index, frame_index, rect))
            for i in range(len(rects)):
                # frames of each monitor arrival in order, track request is also in order
                # in corresponding receive pipe
                track_result: TrackResult = self.output_pipes[monitor_index].get()
                # but track result may be out of order at each request, should sort by original rect id
                result_set[track_result.rect_id] = track_result.result
        return [r for r in result_set if r is not None and len(r)]

    def cancel(self):
        self.status.set(SystemStatus.SHUT_DOWN)
        for p in self.proc_instances:
            p.join()
            p.close()

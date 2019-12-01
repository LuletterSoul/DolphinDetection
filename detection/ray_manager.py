#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: ray.py
@time: 2019/11/28 13:26
@version 1.0
@desc:
"""
import time
from pathlib import Path
import numpy as np
import cv2
import imutils
import ray
import time

import stream
from config import VideoConfig
from detection import EmbeddingControlMonitor, VideoOnlineSampleBasedRayCapture, VideoOfflineRayCapture, \
    DetectorController, RayDetector, ReconstructResult
from utils import logger, crop_by_roi


@ray.remote
class EmbeddingControlBasedRayMonitor(EmbeddingControlMonitor):

    def __init__(self, video_config_path: Path, stream_path: Path, sample_path: Path, frame_path: Path,
                 region_path: Path, offline_path: Path = None) -> None:
        super().__init__(video_config_path, stream_path, sample_path, frame_path, region_path, offline_path,
                         build_pool=False)
        # self.caps_queue = [ray.put(Manager().Queue(maxsize=500)) for c in self.cfgs]
        # self.pipes = [ray.put(Manager().Queue()) for c in self.cfgs]
        self.caps_queue = None
        self.pipes = None

        self.controllers = self.init_controllers()
        self.caps = self.init_caps()
        self.stream_receivers = self.init_stream_receivers()
        self.futures = []

    def init_stream_receivers(self):
        return [
            stream.StreamRayReceiver.remote(self.stream_path / str(c.index), self.offline_path, c, self.caps[idx],
                                            None) for
            idx, c in
            enumerate(self.cfgs)]

    def init_caps(self):
        caps = []
        for idx, c in enumerate(self.cfgs):
            if c.online == 'http':
                caps.append(
                    VideoOnlineSampleBasedRayCapture.remote(self.stream_path / str(c.index),
                                                            self.sample_path / str(c.index),
                                                            None,
                                                            None,
                                                            c,
                                                            self.controllers[idx],
                                                            c.sample_rate))
            else:
                caps.append(
                    VideoOfflineRayCapture.remote(self.stream_path / str(c.index), self.sample_path / str(c.index),
                                                  self.offline_path,
                                                  None,
                                                  # self.pipes[idx],
                                                  # self.caps_queue[idx],
                                                  None,
                                                  c, self.offline_path / str(c.index),
                                                  c.sample_rate,
                                                  delete_post=False))
        return caps

    def call(self):
        # Init stream receiver firstly, ensures video index that is arrived before detectors begin detection..
        while True:
            if self.quit:
                break
            receiver_futures = [self.init_stream_receiver(i) for i, cfg in enumerate(self.cfgs)]

        # Run video capture from stream
        # caps_future = [self.caps[i].read.remote() for i, cfg in enumerate(self.cfgs)]
        # self.futures.append(caps_future)

        # Init detector controller
        # controller_futures = self.init_controllers()
        # self.futures.append(controller_futures)

    def init_controllers(self):
        controllers = [
            RayBasedDetectorController.remote(cfg, self.stream_path / str(cfg.index), self.region_path, self.frame_path,
                                              None,
                                              None) for
            idx, cfg in enumerate(self.cfgs)]

        # for i, cfg in enumerate(self.cfgs):
        #     logger.info('Init detector controller [{}]....'.format(cfg.index))
        #     controller_futures.append(self.controllers[i].start.remote(None))
        #     logger.info('Done init detector controller [{}]....'.format(cfg.index))
        return controllers

    def init_stream_receiver(self, i):
        # return self.process_pool.apply_async(self.stream_receivers[i].receive_online)
        return self.stream_receivers[i].receive_task.remote()

    def wait(self):
        super().wait()
        logger.info('Ray Wait')
        for future in self.futures:
            if isinstance(future, list):
                f_ids, r_ids = ray.wait(future)

            else:
                f_id = ray.get(future)
        logger.info('Ray done')
        # wait all remote future arrivied in Thread Pool Executor


class PreProcessor(object):

    def __init__(self, cfg: VideoConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def preprocess(self, frame):
        original_frame = frame.copy()
        if self.cfg.resize['scale'] != -1:
            frame = cv2.resize(frame, (0, 0), fx=self.cfg.resize['scale'], fy=self.cfg.resize['scale'])
        elif self.cfg.resize['width'] != -1:
            frame = imutils.resize(frame, width=self.cfg.resize['width'])
        elif self.cfg.resize['height'] != -1:
            frame = imutils.resize(frame, height=self.cfg.resize['height'])
        frame = crop_by_roi(frame, self.cfg.roi)
        # frame = imutils.resize(frame, width=1000)
        # frame = frame[340:, :, :]
        # frame = frame[170:, :, :]
        frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
        return frame, original_frame


class FrameDispatcher(object):

    def __init__(self, detectors) -> None:
        self.detectors = detectors

    def dispatch_ray_frame(self, frame):
        # start = time.time()
        # while True:
        #     if self.quit:
        #         break
        # frame = self.frame_queue.get()
        detect_futures = [d.detect_task.remote(frame) for d in self.detectors]
        # for sp in self.send_pipes:
        # frame_id = ray.put(ray.get(frame))
        # sp.put(frame_id)
        # logger.info('Dispatch frame to all detectors....')
        return detect_futures


class FrameReconstructor(object):

    def __init__(self, cfg, col, row, col_step, row_step) -> None:
        super().__init__()
        self.col = col
        self.row = row
        self.cfg = cfg
        self.col_step = col_step
        self.row_step = row_step
        self.start = time.time()
        self.cnt = 0

    def construct_ray_frame(self, results):
        logger.info('Waiting [{}].....'.format(self.cnt))
        res_ready_ids, remaining_ids = ray.wait(results, num_returns=len(results))
        logger.info('Constructing [{}].....'.format(self.cnt))
        # results = [ray.get(obj_id) for obj_id in res_ready_ids]
        results = ray.get(res_ready_ids)
        sub_frames = [r.frame for r in results]
        sub_binary = [r.binary for r in results]
        sub_thresh = [r.thresh for r in results]
        constructed_frame = self.construct_rgb(sub_frames)
        constructed_binary = self.construct_gray(sub_binary)
        constructed_thresh = self.construct_gray(sub_thresh)
        if self.cnt % 100 == 0:
            end = time.time() - self.start
            logger.info(
                'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
                    self.cfg.index, round(end, 2), round(end / 100, 2)))
            self.start = time.time()

        # self.handle_persist(constructed_frame, results)
        self.cnt += 1
        return ReconstructResult(constructed_frame, constructed_binary, constructed_thresh)

    def construct_rgb(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.col, self.row, self.col_step, self.row_step, 3))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3, 4))
        constructed_frame = np.reshape(sub_frames, (self.col * self.col_step, self.row * self.row_step, 3))
        return constructed_frame

    def construct_gray(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.col, self.row, self.col_step, self.row_step))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3))
        constructed_frame = np.reshape(sub_frames, (self.col * self.col_step, self.row * self.row_step))
        return constructed_frame


class VideoPlayer(object):
    def __init__(self, cfg: VideoConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def display(self, res):
        if self.cfg.show_window:
            cv2.imshow('Reconstructed Frame', res.reconstruct_frame)
            cv2.waitKey(1)
        return True


class FilePersister(object):

    def __init__(self, result_path: Path) -> None:
        super().__init__()
        self.result_path = result_path
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.result_cnt = 0

    def persist(self, reconstruct_result, results):
        res_ready_ids, remaining_ids = ray.wait(results, num_returns=len(results))
        results = ray.get(res_ready_ids)
        for r in results:
            if len(r.regions):
                self.result_cnt += 1
                current_time = time.strftime('%m-%d-%H:%M-', time.localtime(time.time()))
                target = self.result_path / (current_time + str(self.result_cnt) + '.png')
                reconstruct_frame = reconstruct_result.reconstruct_frame
                if reconstruct_frame is not None:
                    cv2.imwrite(str(target), reconstruct_frame)
                # self.result_queue.put(idx)
                # self.result_queue.put(r.original_frame)
        return True


@ray.remote(num_cpus=1)
class RayBasedDetectorController(DetectorController):

    def __init__(self, cfg: VideoConfig, stream_path: Path, region_path: Path, frame_path: Path, frame_queue=None,
                 index_pool=None) -> None:
        super().__init__(cfg, stream_path, region_path, frame_path, frame_queue, index_pool)
        self.result_path = self.region_path / str(self.cfg.index) / 'frames'
        self.detectors = []
        self.pre_processor = PreProcessor(self.cfg)
        self.file_persister = FilePersister(self.result_path)
        self.video_player = VideoPlayer(self.cfg)
        self.col_step = 0
        self.row_step = 0
        # self.frame_reconstructor = FrameReconstructor.remote(self.col, self.row, self.col_step, self.row_step)
        self.frame_reconstructor = None
        self.frame_dispatcher = None
        # self.result_path = None
        # self.send_pipes = [ray.put(Manager().Queue()) for i in range(self.col * self.raw)]
        # self.receive_pipes = [ray.put(Manager().Queue()) for i in range(self.col * self.raw)]
        # self.index_pool = ray.get(index_pool)
        # self.frame_queue = ray.get(frame_queue)
        # self.result_queue = Manager().Queue()
        self.futures = []

    def init_step_after_stream(self, frame):
        frame, original_frame = self.pre_processor.preprocess(frame)
        self.col_step = int(frame.shape[0] / self.col)
        self.row_step = int(frame.shape[1] / self.row)
        return frame.shape

    def init_detectors_after_stream(self, shape):
        detectors = []
        logger.info('Init total [{}] detectors....'.format(self.col * self.row))
        # self.detectors = []
        for i in range(self.col):
            for j in range(self.row):
                region_detector_path = self.region_path / str(self.cfg.index) / (str(i) + '-' + str(j))
                index = self.col * i + j
                logger.info(index)
                detectors.append(
                    RayDetector.remote(self.col_step, self.row_step, i, j, self.cfg, None,
                                       None,
                                       region_detector_path, shape))
        logger.info('Detectors init done....')
        return detectors

    def start_stream_task(self, frame):
        if self.frame_reconstructor is None:
            shape = self.init_step_after_stream(frame)
            self.frame_reconstructor = FrameReconstructor(self.cfg, self.col, self.row, self.col_step,
                                                          self.row_step)
            self.detectors = self.init_detectors_after_stream(shape)
            self.frame_dispatcher = FrameDispatcher(self.detectors)

        frame, original_frame = self.pre_processor.preprocess(frame)
        # res = pool.apply_async(self.collect_and_reconstruct, ())
        # futures = []
        # logger.info('Controller [{}] Stream Task: Dispatch'.format(self.cfg.index))
        results = self.frame_dispatcher.dispatch_ray_frame(frame)
        reconstruct_result = self.frame_reconstructor.construct_ray_frame(results)
        logger.info('Controller [{}] Stream Task: Reconstruct'.format(self.cfg.index))
        # futures.append(cr_future)
        # pool.apply_async(self.dispatch, ())
        # futures.append(results)
        # thread_res.append(thread_pool.submit(self.write_work))
        # logger.info('Controller [{}] Stream Task: Persist'.format(self.cfg.index))
        self.file_persister.persist(reconstruct_result, results)
        logger.info('Controller [{}] Stream Task: Persist'.format(self.cfg.index))
        self.video_player.display(reconstruct_result)
        logger.info('Controller [{}] Stream Task: Display'.format(self.cfg.index))

        # futures.append(write_future)
        # for idx, d in enumerate(self.detectors):
        #     logger.info('Submit detector [{},{},{}] task..'.format(self.cfg.index, d.col_index, d.raw_index))
        #     detect_proc_res.append(pool.apply_async(d.detect, ()))
        # futures.append(d.detect.remote())
        # detect_proc_res.append(pool.submit(d.detect, ()))
        # logger.info('Done detector [{},{},{}]'.format(self.cfg.index, d.col_index, d.raw_index))
        return True

    #
    # @ray.method(num_return_vals=2)
    # def preprocess(self, frame):
    #     original_frame = frame.copy()
    #     if self.cfg.resize['scale'] != -1:
    #         frame = cv2.resize(frame, (0, 0), fx=self.cfg.resize['scale'], fy=self.cfg.resize['scale'])
    #     elif self.cfg.resize['width'] != -1:
    #         frame = imutils.resize(frame, width=self.cfg.resize['width'])
    #     elif self.cfg.resize['height'] != -1:
    #         frame = imutils.resize(frame, height=self.cfg.resize['height'])
    #     frame = crop_by_roi(frame, self.cfg.roi)
    #     # frame = imutils.resize(frame, width=1000)
    #     # frame = frame[340:, :, :]
    #     # frame = frame[170:, :, :]
    #     frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
    #     return frame, original_frame

    # def persist(self, reconstruct_result, results):
    #     res_ready_ids, remaining_ids = ray.wait(results, num_returns=len(results))
    #     results = ray.get(res_ready_ids)
    #     for r in results:
    #         if len(r.regions):
    #             self.result_cnt += 1
    #             current_time = time.strftime('%m-%d-%H:%M-', time.localtime(time.time()))
    #             target = self.result_path / (current_time + str(self.result_cnt) + '.png')
    #             reconstruct_frame = ray.get(reconstruct_result).reconstruct_result
    #             if reconstruct_frame is not None:
    #                 cv2.imwrite(str(target), reconstruct_frame)
    #             # self.result_queue.put(idx)
    #             # self.result_queue.put(r.original_frame)
    #     return True

    #
    # def collect_ray_frame(self):
    #     res = []
    #     for rp in self.receive_pipes:
    #         res.append(rp.get())
    #     # res_ready_ids, remaining_ids = ray.wait(res, num_returns=len(res))
    #     # res = [ray.get(obj_id) for obj_id in res_ready_ids]
    #     logger.info('Collect sub-frames from all detectors....')
    #     return res

    #
    # @ray.method(num_return_vals=1)
    # def dispatch_ray_frame(self, frame):
    #     # start = time.time()
    #     # while True:
    #     #     if self.quit:
    #     #         break
    #     # frame = self.frame_queue.get()
    #     frame_id, original_frame_id = self.preprocess.remote(frame)
    #     detect_futures = [d.detect_task.remote(frame_id) for d in self.detectors]
    #     # for sp in self.send_pipes:
    #     # frame_id = ray.put(ray.get(frame))
    #     # sp.put(frame_id)
    #     logger.info('Dispatch frame to all detectors....')
    #     return detect_futures

    # internal = (time.time() - start) / 60
    # if int(internal) == self.cfg.sample_internal:
    #     cv2.imwrite(str(self.frame_path / ))

    def collect_and_reconstruct_ray(self, results):
        logger.info('Detection controller [{}] start collect and construct'.format(self.cfg.index))
        cnt = 0
        start = time.time()
        # while True:
        #     if self.quit:
        #         break
        logger.info('Done collected from detectors.....')
        # results = self.collect_ray_frame.remote()
        frame_id, binary_id, thresh_id = self.construct_ray_frame.remote(results)
        logger.info('Constructing sub-frames into a original frame....')
        # frame, binary, thresh = self.construct(results)
        cnt += 1
        if cnt % 100 == 0:
            end = time.time() - start
            logger.info(
                'Detection controller [{}]: Operation Speed Rate [{}]s/100fs, unit process rate: [{}]s/f'.format(
                    self.cfg.index, round(end, 2), round(end / 100, 2)))
            start = time.time()
        if self.cfg.draw_boundary:
            self.play.remote(frame_id)
        # logger.info('Done constructing of sub-frames into a original frame....')
        return True

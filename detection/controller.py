#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: manager.py
@time: 2019/11/16 13:22
@version 1.0
@desc:
"""

import os.path as osp
import time
import traceback
from multiprocessing import Pool

import imutils
import numpy as np

from classfy.model import DolphinClassifier
from config import ModelType
from .render import ArrivalMessage, ArrivalMsgType
from stream.websocket import *
from utils.cache import SharedMemoryFrameCache
from . import Detector
from .capture import *
from .detect_funcs import *
from .params import DispatchBlock
# from utils import NoDaemonPool as Pool
from .ssd import SSDDetector


# from pynput.keyboard import Key, Controller, Listener
# import keyboard
# from capture import *


class DetectorController(object):
    """

    Monitor will build multiple video stream receivers according the video configuration
    """

    def __init__(self, cfg: VideoConfig, stream_path: Path, candidate_path: Path, frame_path: Path,
                 frame_queue: Queue,
                 index_pool: Queue,
                 msg_queue: Queue, frame_cache: SharedMemoryFrameCache) -> None:
        super().__init__()
        self.cfg = cfg
        self.dol_id = 10000
        self.dol_gone = True
        self.stream_path = stream_path
        self.frame_path = frame_path
        self.candidate_path = candidate_path
        self.block_path = candidate_path / 'blocks'
        self.result_path = self.candidate_path / 'frames'
        self.crop_result_path = self.candidate_path / 'crops'
        self.rect_stream_path = self.candidate_path / 'render-streams'
        self.original_stream_path = self.candidate_path / 'original-streams'
        self.test_path = self.candidate_path / 'tests'
        self.preview_path = self.candidate_path / 'preview'
        self.create_workspace()

        self.result_cnt = 0
        self.x_num = cfg.routine['col']
        self.y_num = cfg.routine['row']
        self.x_step = 0
        self.y_step = 0
        self.block_info = BlockInfo(self.y_num, self.x_num, self.y_step, self.x_step)

        self.frame_queue = frame_queue
        self.msg_queue = msg_queue
        self.result_queue = Manager().Queue(self.cfg.max_streams_cache)
        self.quit = Manager().Event()
        self.quit.clear()

        self.status = Manager().Value('i', SystemStatus.SHUT_DOWN)
        self.global_index = Manager().Value('i', 0)

        self.next_prepare_event = Manager().Event()
        self.next_prepare_event.clear()
        self.pre_detect_index = -self.cfg.future_frames
        self.history_write = False

        # frame cache
        self.cache_size = self.cfg.cache_size
        self.original_frame_cache = Manager().list()
        self.original_frame_cache = frame_cache

        # bbox cache, retrieval by frame index,none represent this frame don't exist bbox
        # self.render_rect_cache = Manager().list()
        self.render_rect_cache = Manager().dict()
        # self.render_rect_cache[:] = [None] * self.cache_size
        self.LOG_PREFIX = f'Controller [{self.cfg.index}]: '
        self.save_cache = {}

    def cancel(self):
        pass

    def init_control_range(self):
        # read a frame, record frame size before running detectors
        frame = self.frame_queue.get()
        frame, original_frame = preprocess(frame, self.cfg)
        self.x_step = int(frame.shape[1] / self.x_num)
        self.y_step = int(frame.shape[0] / self.y_num)
        self.block_info = BlockInfo(self.y_num, self.x_num, self.y_step, self.x_step)

    def init_detectors(self):
        logger.info('Init total [{}] detectors....'.format(self.x_num * self.y_num))
        self.detectors = []
        for i in range(self.x_num):
            for j in range(self.y_num):
                region_detector_path = self.block_path / (str(i) + '-' + str(j))
                index = self.x_num * i + j
                self.detectors.append(
                    Detector(self.x_step, self.y_step, i, j, self.cfg, self.send_pipes[index],
                             self.receive_pipes[index],
                             region_detector_path))
        logger.info('Detectors init done....')

    def create_workspace(self):

        self.rect_stream_path.mkdir(exist_ok=True, parents=True)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.crop_result_path.mkdir(exist_ok=True, parents=True)
        self.original_stream_path.mkdir(exist_ok=True, parents=True)
        self.block_path.mkdir(exist_ok=True, parents=True)
        self.test_path.mkdir(exist_ok=True, parents=True)
        self.preview_path.mkdir(exist_ok=True, parents=True)

    def start(self, pool):
        self.status.set(SystemStatus.RUNNING)
        self.init_control_range()
        self.init_detectors()
        return None

    def write_frame_work(self):
        """
        Write Key frame into disk if detection event occurs
        :return:
        """
        logger.info(
            '*******************************Controler [{}]: Init detection frame frame routine********************************'.format(
                self.cfg.index))
        if not self.cfg.save_box:
            logger.info(
                '*******************************Controler [{}]: Frame writing routine disabled********************************'.format(
                    self.cfg.index))
            return

        while True:
            if self.status.get() == SystemStatus.SHUT_DOWN and self.result_queue.empty():
                logger.info(
                    '*******************************Controller [{}]: Frame write routine exit********************************'.format(
                        self.cfg.index))
                break
            try:
                # r = self.get_result_from_queue()
                if not self.result_queue.empty():
                    # result_queue = self.result_queue.get(timeout=1)
                    frame_index, rects = self.result_queue.get(timeout=1)
                    frame = self.original_frame_cache[frame_index]
                    self.result_cnt += 1
                    current_time = generate_time_stamp() + '_'
                    img_name = current_time + str(self.result_cnt) + '.png'
                    target = self.result_path / img_name
                    cv2.imwrite(str(target), frame)
                    self.label_crop(frame, img_name, rects)
                    self.save_bbox(img_name, rects)
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
        return True

    def dispatch(self):
        pass

    def collect(self, args):
        res = []
        for rp in self.receive_pipes:
            res.append(rp.get())
        logger.info('Collect sub-frames from all detectors....')
        return res

    def construct_rgb(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.x_num, self.y_num, self.x_step, self.y_step, 3))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3, 4))
        constructed_frame = np.reshape(sub_frames, (self.x_num * self.x_step, self.y_num * self.y_step, 3))
        return constructed_frame

    def construct_gray(self, sub_frames):
        sub_frames = np.array(sub_frames)
        sub_frames = np.reshape(sub_frames, (self.y_num, self.x_num, self.y_step, self.x_step))
        sub_frames = np.transpose(sub_frames, (0, 2, 1, 3))
        constructed_frame = np.reshape(sub_frames, (self.y_num * self.y_step, self.x_num * self.x_step))
        return constructed_frame

    def save_bbox(self, frame_name, boundary_rect):
        bbox_path = str(self.candidate_path / 'bbox.json')
        self.save_cache[frame_name] = boundary_rect

        if not osp.exists(bbox_path):
            fw = open(bbox_path, 'w')
            fw.write(json.dumps(self.save_cache, indent=4))
            fw.close()

        if len(self.save_cache) == 2:
            fr = open(bbox_path, 'r')
            save_file = json.load(fr)
            fr.close()

            for key in self.save_cache:
                save_file[key] = self.save_cache[key]

            fw = open(bbox_path, 'w')
            fw.write(json.dumps(save_file, indent=4))
            fw.close()

            self.save_cache = {}

    def construct(self, *args) -> ConstructResult:
        pass

    def label_crop(self, frame, label_name, rects):
        for idx, rect in enumerate(rects):
            crop_path = self.crop_result_path / (str(idx) + '_' + label_name)
            cropped = crop_by_rect(self.cfg, rect, frame)
            cv2.imwrite(str(crop_path), cropped)


class TaskBasedDetectorController(DetectorController):
    """
    Main frame controller, consumes some heavy services,such as detection notification,forward or post handle,
    video rendering,cache writing or reading.
    It will communicate all frames with video capture by shared memory
    """

    def __init__(self, server_cfg: ServerConfig, cfg: VideoConfig, stream_path: Path, candidate_path: Path,
                 frame_path: Path, frame_queue: Queue, index_pool: Queue, msg_queue: Queue, streaming_queue: List,
                 render_notify_queue, frame_cache: SharedMemoryFrameCache,recoder) -> None:
        super().__init__(cfg, stream_path, candidate_path, frame_path, frame_queue, index_pool, msg_queue, frame_cache)
        # self.construct_params = ray.put(
        #     ConstructParams(self.result_queue, self.original_frame_cache, self.render_frame_cache,
        #                     self.render_rect_cache, self.stream_render, 500, self.cfg))
        self.server_cfg = server_cfg
        self.construct_params = ConstructParams(self.result_queue, self.original_frame_cache,
                                                self.render_rect_cache, None, 500, self.cfg)
        # self.pool = ThreadPoolExecutor()
        # self.threads = []
        self.push_stream_queue = streaming_queue
        self.display_pipe = Manager().Queue(1000)
        self.detect_params = []
        self.detectors = []
        self.args = Manager().list()
        self.frame_stack = Manager().list()
        self.detect_handler = None
        # self.stream_render = stream_render
        self.render_notify_queue = render_notify_queue
        self.recorder = recoder
        self.init_control_range()
        self.init_detectors()

    def init_detect_handler(self, handler):
        self.detect_handler = handler

    def listen(self):
        """
        listen shutdown signal
        :return:
        """
        if self.quit.wait():
            self.status.set(SystemStatus.SHUT_DOWN)
            # self.stream_render.quit.set()

    def init_detectors(self):
        """
        init all well-defined and neccessary parameters about detection
        :return:
        """
        logger.info(
            '*******************************Controller [{}]: Init total [{}] detectors********************************'.format(
                self.cfg.index,
                self.x_num * self.y_num))
        for i in range(self.x_num):
            for j in range(self.y_num):
                region_detector_path = self.block_path / (str(i) + '-' + str(j))
                # index = self.col * i + j
                # self.detectors.append(
                #     TaskBasedDetector(self.col_step, self.row_step, i, j, self.cfg, self.send_pipes[index],
                #                       self.receive_pipes[index],
                #                       region_detector_path))
                self.detect_params.append(
                    DetectorParams(self.x_step, self.y_step, i, j, self.cfg, region_detector_path))
        logger.info(
            '*******************************Controller [{}]: detectors init done ********************************'.format(
                self.cfg.index))

    def init_control_range(self):
        """
        init frame blocks range
        :return:
        """
        # read a frame, record frame size before running detectors
        empty = np.zeros(self.cfg.shape).astype(np.uint8)
        frame, _ = preprocess(empty, self.cfg)
        self.x_step = int(frame.shape[1] / self.x_num)
        self.y_step = int(frame.shape[0] / self.y_num)
        self.block_info = BlockInfo(self.y_num, self.x_num, self.y_step, self.x_step)

    def collect(self, args):
        return [f.result() for f in args]

    def collect_and_reconstruct(self, *args):
        """
        collect detection result from detector
        if using dividend and conquer principle, we have to reconstruct sun-frame blocks into a whole original frame,
        which is usually time-consuming
        # TODO running detection algorithm in different process and reconstruct sub-frames
        # TODO a small size shared memory is appropriate to post sub-frame,could decrease communication cost between
            processes compared to pipes serialization
        :param args: (List[DetectResult],Detection Model Ref,Original Frame)
        :return: construct result
        """
        # results = self.collect(args)
        s = time.time()
        construct_result: ConstructResult = self.construct(*args)
        e = 1 / (time.time() - s)
        # logger.debug(self.LOG_PREFIX + f'Construct Speed: [{round(e, 2)}]/FPS')
        if construct_result is not None:
            frame = construct_result.frame
            if self.cfg.draw_boundary:
                frame, _ = preprocess(frame, self.cfg)
                frame = draw_boundary(frame, self.block_info)
                # logger.info('Done constructing of sub-frames into a original frame....')
            if self.cfg.show_window:
                pass
            # if self.cfg.show_window:
            #     frame = imutils.resize(frame, width=800)
            #     self.display_pipe.put(frame)
            # cv2.imshow('Reconstructed Frame', frame)
            # cv2.waitKey(1)
        else:
            logger.error('Empty reconstruct result.')
            traceback.print_exc()
        return construct_result

    def post_detect(self, frame, idx) -> List[DetectionResult]:
        sub_results = []
        for d in self.detect_params:
            block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                  idx, frame.shape)
            sub_results.append(detect_based_task(block, d))
        return sub_results

    def construct(self, *args) -> ConstructResult:
        # sub_frames = [r.frame for r in results]
        results = args[0]
        _model = args[1]
        original_frame = args[-1]
        # sub_binary = [r.binary for r in results]
        # sub_thresh = [r.thresh for r in results]
        # constructed_frame = self.construct_rgb(sub_frames)
        # constructed_binary = self.construct_gray(sub_binary)
        # constructed_thresh = self.construct_gray(sub_thresh)
        # logger.debug(f'Controller [{self.cfg.index}]: Construct frames into a original frame....')
        try:
            current_index = results[0].frame_index
            render_frame = original_frame.copy()
            push_flag = False
            for r in results:
                if len(r.rects):
                    self.result_queue.put((r.frame_index, r.rects))
                    rects = []
                    if len(r.rects) >= 5:
                        logger.info(f'To many rect candidates: [{len(r.rects)}].Abandoned..... ')
                        return ConstructResult(original_frame, None, None, frame_index=current_index)
                    for rect in r.rects:
                        start = time.time()
                        detect_result = True
                        if not self.cfg.cv_only:
                            candidate = crop_by_rect(self.cfg, rect, render_frame)
                            obj_class, output = _model.predict(candidate)
                            detect_result = (obj_class == 0)
                            logger.debug(
                                self.LOG_PREFIX + f'Model Operation Speed Rate: [{round(1 / (time.time() - start), 2)}]/FPS')
                        if detect_result:
                            logger.debug(
                                f'============================Controller [{self.cfg.index}]: Dolphin Detected============================')
                            self.dol_gone = False
                            push_flag = True
                            rects.append(rect)
                    r.rects = rects
                    if push_flag:
                        json_msg = creat_detect_msg_json(video_stream=self.cfg.rtsp, channel=self.cfg.channel,
                                                         timestamp=current_index, rects=r.rects, dol_id=self.dol_id,
                                                         camera_id=self.cfg.camera_id, cfg=self.cfg)
                        self.msg_queue.put(json_msg)
                        if self.cfg.render:
                            # self.render_rect_cache[current_index % self.cache_size] = r.rects
                            self.render_rect_cache[current_index] = r.rects
                        self.forward_filter(current_index, rects)
                        self.notify_render(current_index)
                    else:
                        if not self.dol_gone:
                            empty_msg = creat_detect_empty_msg_json(video_stream=self.cfg.rtsp,
                                                                    channel=self.cfg.channel,
                                                                    timestamp=current_index, dol_id=self.dol_id,
                                                                    camera_id=self.cfg.camera_id)
                            self.msg_queue.put(empty_msg)
                            logger.info(self.LOG_PREFIX + f'Send empty msg: {empty_msg}')
                            self.dol_id += 1
                            self.dol_gone = True
            self.update_render(current_index)
            self.update_detect_handler(current_index)
            return ConstructResult(render_frame, None, None, detect_flag=push_flag, results=results,
                                   frame_index=current_index)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

    def control(self):
        """
        Handle all frames from stream producers or frame receivers as fast as possible
        Here we are using the Python 3.8 newest feature,Shared Memory as global frame cache,
        which can reach 60FPS+ speed when processing 4K video frames
        :return:
        """
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{int(self.cfg.index) % 4}'

        from mmdetection import init_detector
        import torch
        torch.set_num_threads(1)
        classifier = None
        model = None

        # init different detection models according configuration inside the SUB-PROCESS
        # every frame looper will occupy single model instance by now
        # TODO less model instances,but could be shared by all detectors
        if self.server_cfg.detect_mode == ModelType.SSD:
            model = SSDDetector(model_path=self.server_cfg.detect_model_path, device_id='0')
            model.run()
            logger.info(
                f'*******************************Capture [{self.cfg.index}]: Running SSD Model********************************')
        elif self.server_cfg.detect_mode == ModelType.CLASSIFY and not self.cfg.cv_only:
            classifier = DolphinClassifier(model_path=self.server_cfg.classify_model_path,
                                           device_id=self.server_cfg.dt_id)
            classifier.run()
            logger.info(
                f'*******************************Capture [{self.cfg.index}]: Running Classifier Model********************************')
        elif self.server_cfg.detect_mode == ModelType.CASCADE:
            os.environ["CUDA_VISIBLE_DEVICES"] = f'{int(self.cfg.index) % 4}'
            cascade_model_cfg = self.server_cfg.cascade_model_cfg
            cascade_model_path = self.server_cfg.cascade_model_path
            if self.cfg.alg['cascade_model_cfg'] != '':
                cascade_model_cfg = self.cfg.alg['cascade_model_cfg']
                cascade_model_path = self.cfg.alg['cascade_model_path']
            model = init_detector(cascade_model_cfg, cascade_model_path)
            logger.info(
                f'*******************************Capture [{self.cfg.index}]: Running Cascade-RCNN Model********************************')

        logger.info(
            '*******************************Controller [{}]: Init Loop Stack********************************'.format(
                self.cfg.index))
        pre_index = 0
        while self.status.get() == SystemStatus.RUNNING:
            try:
                # always obtain the newest reach frame when detection is slower than video stream receiver
                current_index = self.global_index.get() - 1
                # but sometimes detection process could be faster than stream receiving
                # so always keep the newest frame index and don't rollback
                # otherwise it will flicker terribly to push stream
                if current_index <= pre_index:
                    continue
                pre_index = current_index
                # logger.info(f'Current index: {current_index}')
                frame = self.original_frame_cache[current_index]
                s = time.time()
                self.dispatch(frame, None, model, classifier, current_index)
                e = 1 / (time.time() - s)
                logger.debug(self.LOG_PREFIX + f'Detection Process Speed: [{round(e, 2)}]/FPS')
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
        logger.info(
            '*******************************Controller [{}]: Loop Stack Exit********************************'.format(
                self.cfg.index))

    def put_cache(self, *args):
        """
        callback function.video capture will pass frame
        into the global cache every time it calls this function.
        :param args:
        :return:
        """
        frame = args[0]
        s = time.time()
        if frame.shape[1] > self.cfg.shape[1]:
            frame = imutils.resize(frame, width=self.cfg.shape[1])
        self.original_frame_cache[self.global_index.get()] = frame
        self.global_index.set(self.global_index.get() + 1)
        e = 1 / (time.time() - s)
        logger.debug(self.LOG_PREFIX + f'Global Cache Writing Speed: [{round(e, 2)}]/FPS')

    def dispatch(self, *args):
        """
        Dispatch frame to key detection handler
        :param args: [Original Frame, None, SSD Model Ref, Classification Model Ref, Frame Index]
        :return:
        """
        original_frame = args[0]
        self.pre_cnt = args[-1]
        # select different detection methods according the configuration
        if self.server_cfg.detect_mode == ModelType.CLASSIFY:  # using classifier
            self.classify_based(args, original_frame.copy())
        # using SSD or Cascade-RCNN
        elif self.server_cfg.detect_mode == ModelType.SSD or self.server_cfg.detect_mode == ModelType.CASCADE:
            self.model_based(args, original_frame.copy())
        elif self.server_cfg.detect_mode == ModelType.FORWARD:  # done nothing
            self.forward(args, original_frame)

    # self.clear_original_cache()

    def forward(self, args, original_frame):
        """
        do nothing, just forward video stream into target server
        :param args:
        :param original_frame:
        :return:
        """
        self.post_stream_req(None, original_frame)

        # self.push_stream_queue.append((original_frame, None, self.pre_cnt))

    def model_based(self, args, original_frame):
        model_instance = args[2]
        if self.pre_cnt % self.cfg.sample_rate == 0:
            start = time.time()
            frames_results = self.get_model_result(original_frame, model_instance, self.server_cfg)
            logger.debug(
                self.LOG_PREFIX + f'Model [{self.cfg.index}]: Operation Speed Rate: [{round(1 / (time.time() - start), 2)}]/FPS')
            # render_frame = original_frame.copy()
            detect_results = []
            detect_flag = False
            current_index = self.pre_cnt
            rects = []
            if self.cfg.show_window:
                cv2.namedWindow(str(self.cfg.index), cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
                frame = original_frame
                if len(frames_results):
                    for rect in frames_results[0]:
                        if rect[4] > self.cfg.alg['ssd_confidence']:
                            cv2.imwrite(f'data/frames/{self.cfg.index}_{current_index}.png',
                                        cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                            color = np.random.randint(0, 255, size=(3,))
                            color = [int(c) for c in color]
                            # get a square bbox, the real bbox of width and height is universal as 224 * 224 or 448 * 448
                            p1, p2 = bbox_points(self.cfg, rect, original_frame.shape)
                            # write text
                            frame = paint_chinese_opencv(frame, '江豚', p1)
                            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
                            cv2.putText(frame, str(round(rect[4], 2)), (p2[0], p2[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
                cv2.imshow(str(self.cfg.index), frame)
                cv2.waitKey(1)

            if len(frames_results):
                for frame_result in frames_results:
                    if len(frame_result):
                        rects = [r for r in frame_result if
                                 r[4] > self.cfg.alg['ssd_confidence']]
                        if len(rects):
                            self.result_queue.put((current_index, rects))
                            if len(rects) >= 3:
                                logger.info(f'To many rect candidates: [{len(rects)}].Abandoned..... ')
                                return ConstructResult(original_frame, None, None, frame_index=self.pre_cnt)
                            detect_results.append(DetectionResult(rects=rects))
                            detect_flag = True
                            self.dol_gone = False
                            logger.info(
                                f'============================Controller [{self.cfg.index}]: Dolphin Detected in frame [{current_index}]============================')
                    if detect_flag:
                        # json_msg = creat_detect_msg_json(video_stream=self.cfg.rtsp, channel=self.cfg.channel,
                        #                                 timestamp=current_index, rects=rects, dol_id=self.dol_id,
                        #                                 camera_id=self.cfg.camera_id)
                        # self.msg_queue.put(json_msg)
                        # logger.debug(f'put detect message in msg_queue {json_msg}...')
                        # self.render_frame_cache[current_index % self.cache_size] = render_frame
                        #if self.cfg.render:
                        #    self.render_rect_cache[current_index % self.cache_size] = rects
                        self.forward_filter(current_index, rects)
                        self.notify_render(current_index)
                        self.recorder.record()
                    # else:
                    #    if not self.dol_gone:
                    # empty_msg = creat_detect_empty_msg_json(video_stream=self.cfg.rtsp,
                    #                                        channel=self.cfg.channel,
                    #                                        timestamp=current_index, dol_id=self.dol_id,
                    #                                        camera_id=self.cfg.camera_id)
                    # self.dol_id += 1
                    # self.msg_queue.put(empty_msg)
                    #        self.dol_gone = True
            self.update_render(current_index)
            self.update_detect_handler(current_index)
            # threading.Thread(target=self.str.notify, args=(current_index,), daemon=True).start()
            construct_result = ConstructResult(None, None, None, None, detect_flag, detect_results,
                                               frame_index=self.pre_cnt)
            self.post_stream_req(construct_result, original_frame)
        else:
            # if self.cfg.push_stream:
            #     self.push_stream_queue.append((original_frame, None, self.pre_cnt))
            self.post_stream_req(None, original_frame)

    def update_detect_handler(self, current_index):
        """
        update frame index maintained by detect handler
        :param current_index:
        :return:
        """
        if self.cfg.forward_filter:
            self.detect_handler.reset(ArrivalMessage(current_index, ArrivalMsgType.UPDATE))

    def update_render(self, current_index):
        """
        update frame index maintained by video render service
        :param current_index:
        :return:
        """

        # if forward filtering is enabled, video render notification is under control by DetectionSignalHandler
        if self.cfg.render and not self.cfg.forward_filter:
            self.render_notify_queue.put(ArrivalMessage(current_index, ArrivalMsgType.UPDATE))

    def notify_render(self, current_index):
        """
        notify video render service to generate video clip
        :param current_index: arrival frame index
        :return:
        """
        # if forward filtering is enabled, video render notification is under control by DetectionSignalHandler
        if self.cfg.render and not self.cfg.forward_filter:
            self.render_notify_queue.put(ArrivalMessage(current_index, ArrivalMsgType.DETECTION))

    def forward_filter(self, current_index, rects):
        """
        forward filter
        :param rects:
        :param current_index:
        :return:
        """
        if self.cfg.forward_filter:
            self.detect_handler.notify(ArrivalMessage(current_index, ArrivalMsgType.DETECTION, rects=rects))
        # TODO control instant detection signal commit

    def get_model_result(self, original_frame, model, server_cfg: ServerConfig, inference_detector=None):
        """
        get detection result from ssd model
        :param original_frame: numpy frame
        :param model: model instance
        :return:
        """
        if self.cfg.ssd_divide_four:
            """
            if divide original frame into four blocks(patches) or not 
            """
            # TODO bug to be fixed
            b1, b2, b3, b4 = split_img_to_four(original_frame)
            if server_cfg.detect_mode == 'ssd':
                block_res = model([b1, b2, b3, b4])
            if len(block_res):
                decode(original_frame, block_res, 0)
                decode(original_frame, block_res, 1)
                decode(original_frame, block_res, 2)
                decode(original_frame, block_res, 3)
            return block_res
        else:
            if server_cfg.detect_mode == 'ssd':

                return model([original_frame])
            elif server_cfg.detect_mode == 'cascade':
                from mmdetection import inference_detector
                result_cascade = inference_detector(model, original_frame)
                if len(result_cascade[0]):
                    return result_cascade
                else:
                    return []

    def post_stream_req(self, construct_result, original_frame):
        """
        pass frame to video streamer via pipe
        :param construct_result:
        :param original_frame:
        :return:
        """
        if self.cfg.push_stream:
            if self.cfg.use_sm:
                self.push_stream_queue[0][0] = original_frame
                self.push_stream_queue[1].append((construct_result, self.pre_cnt))
            else:
                self.push_stream_queue.append((original_frame, construct_result, self.pre_cnt))

    def classify_based(self, args, original_frame):
        """
        classification-based detection method
        :param args:
        :param original_frame:
        :return:
        """
        if self.pre_cnt % self.cfg.sample_rate == 0:
            # logger.debug('Controller [{}]: Dispatch frame to all detectors....'.format(self.cfg.index))
            async_futures = []
            frame, original_frame = preprocess(original_frame, self.cfg)
            if self.cfg.show_window:
                cv2.namedWindow(str(self.cfg.index), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(str(self.cfg.index), frame)
                cv2.waitKey(0)
            s = time.time()
            for d in self.detect_params:
                block = DispatchBlock(crop_by_se(frame, d.start, d.end),
                                      self.pre_cnt, original_frame.shape)
                async_futures.append(detect_based_task(block, d))
                # TODO Perform detection acceleration by dispatching frame block to multiple processes
            e = 1 / (time.time() - s)
            # logger.debug(self.LOG_PREFIX + f'Coarser Detection Speed: [{round(e, 2)}]/FPS')
            proc_res: ConstructResult = self.collect_and_reconstruct(async_futures, args[3], original_frame)
            frame = proc_res.frame
            proc_res.frame = None
            self.post_stream_req(proc_res, frame)
        else:
            self.post_stream_req(None, original_frame)

    #
    # def display(self):
    #     logger.info(
    #         '*******************************Controller [{}]: Init video player********************************'.format(
    #             self.cfg.index))
    #     while True:
    #         if self.status.get() == SystemStatus.SHUT_DOWN:
    #             logger.info(
    #                 '*******************************Controller [{}]: Video player exit********************************'.format(
    #                     self.cfg.index))
    #             break
    #         try:
    #             if not self.display_pipe.empty():
    #                 frame = self.display_pipe.get(timeout=1)
    #                 cv2.imshow('Controller {}'.format(self.cfg.index), frame)
    #                 cv2.waitKey(1)
    #         except Exception as e:
    #             logger.error(e)
    #     return True

    def start(self, pool: Pool):
        try:
            self.status.set(SystemStatus.RUNNING)
            threading.Thread(target=self.listen, daemon=True).start()
            threading.Thread(target=self.write_frame_work, daemon=True).start()
            # threading.Thread(target=self.display, daemon=True).start()
            # threading.Thread(target=self.loop_stack, daemon=True).start()
            self.control()
            return True
        except Exception as e:
            logger.error(e)
            traceback.print_exc()

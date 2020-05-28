#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: capture.py
@time: 2019/11/24 11:25
@version 1.0
@desc:
"""
import os
import shutil
import threading
import time
from multiprocessing import Manager
from multiprocessing.queues import Queue
from pathlib import Path
import traceback

import cv2

from config import VideoConfig, SystemStatus
from utils import logger, cap


class VideoCaptureThreading:
    """
    read video frames based cv2.VideoCapture
    """

    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 idx,
                 sample_rate=5, width=640, height=480, delete_post=True):
        self.cfg = cfg
        self.video_path = video_path
        self.sample_path = sample_path
        self.index_pool = index_pool
        self.idx = idx
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.grabbed, self.frame = self.cap.read()
        self.status = Manager().Value('i', SystemStatus.SHUT_DOWN)
        self.src = -1
        self.cap = None
        self.sample_rate = sample_rate
        self.frame_queue = frame_queue
        self.delete_post = delete_post
        self.runtime = 0
        self.posix = None
        self.quit = Manager().Event()
        self.quit.clear()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def __start__(self, *args):
        """
        capture initialization
        :param args:
        :return:
        """
        if self.status.get() == SystemStatus.RUNNING:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.update_capture(0)
        self.status.set(SystemStatus.RUNNING)
        threading.Thread(target=self.listen, args=(), daemon=True).start()
        time.sleep(5)  # wait detection service init done
        self.update(*args)
        return self

    def listen(self):
        """
        controller capture process to shutdown
        :return:
        """
        logger.info('Video Capture [{}]: Start listen event'.format(self.cfg.index))
        if self.quit.wait():
            logger.info('Video Capture [{}]: Receive quit signal'.format(self.cfg.index))
            self.cancel()

    def cancel(self):
        self.status.set(SystemStatus.SHUT_DOWN)

    def load_next_src(self):
        """
        load next video source
        :return:
        """
        logger.debug('Loading video stream from video index pool....')
        self.posix = self.get_posix()
        self.src = str(self.posix)
        if self.posix == -1:
            return self.src
        basename = os.path.basename(self.src)
        filename, extention = os.path.splitext(basename)
        if extention == '.mp4' or extention == '.mov':
            return self.src
        else:
            return 0

    def get_posix(self):
        """
        a base pointer,  can be overrived by sub-class
        :return:
        """
        # DEPRECATED
        return self.video_path / self.index_pool.get()

    def update(self, *args):
        """
        video capture service, loop video frames from a rtsp stream
        :param args:
        :return:
        """
        cnt = 0
        start = time.time()
        logger.info('*******************************Init video capture [{}]********************************'.format(
            self.cfg.index))
        ssd_detector = None
        classifier = None
        server_cfg = args[0]
        # if server_cfg.detect_mode == ModelType.SSD:
        #     ssd_detector = SSDDetector(model_path=server_cfg.detect_model_path, device_id=server_cfg.cd_id)
        #     ssd_detector.run()
        #     logger.info(
        #         f'*******************************Capture [{self.cfg.index}]: Running SSD Model********************************')
        # elif server_cfg.detect_mode == ModelType.CLASSIFY:
        #     classifier = DolphinClassifier(model_path=server_cfg.classify_model_path, device_id=server_cfg.dt_id)
        #     classifier.run()
        #     logger.info(
        #         f'*******************************Capture [{self.cfg.index}]: Running Classifier Model********************************')
        while self.status.get() == SystemStatus.RUNNING:
            # with self.read_lock:
            s = time.time()
            grabbed, frame = self.read_frame()
            e = 1 / (time.time() - s)
            logger.debug(f'Video capture [{self.cfg.index}]: Receive Speed Rate [{round(e, 2)}]/FPS')
            # if e > 25:
            #    sleep_time = 1 / (e / 25) / 2
            #    time.sleep(sleep_time)
            #    logger.info(
            #        f'Video capture [{self.cfg.index}]: too quick receive speed rate [{e}/FPS],sleep [{sleep_time}] seconds.')
            s = time.time()
            if not grabbed:
                # if current video source is end,load the next video sources
                self.update_capture(cnt)
                end = time.time()
                logger.info('Current src consumes time: [{}] seconds'.format(end - start))
                start = time.time()
                cnt = 0
                continue
            # if cnt % self.sample_rate == 0:
            self.pass_frame(frame, args[0], ssd_detector, classifier)
            e = 1 / (time.time() - s)
            logger.debug(f'Video capture [{self.cfg.index}]: Operation Speed Rate [{round(e, 2)}]/FPS')
            cnt += 1
            self.post_frame_process(frame)
            self.runtime = time.time() - start

        logger.info(
            '*******************************Video capture [{}] exit********************************'.format(
                self.cfg.index))
        # logger.info('Video Capture [{}]: cancel..'.format(self.cfg.index))

    def read_frame(self):
        return self.cap.read()

    def pass_frame(self, *args):
        self.frame_queue.put(args[0], block=True)
        # logger.info('Passed frame...')

    def update_capture(self, cnt):
        """
        release old video capture instance and init a new video capture when update video source
        :param cnt:
        :return:
        """
        logger.debug('Read frame done from [{}].Has loaded [{}] frames'.format(self.src, cnt))
        logger.debug('Read next frame from video ....')
        self.handle_history()
        while True:
            src = self.load_next_src()
            if src == str(-1):
                self.cancel()
                return False
            elif src == 0:
                continue
            else:
                break
        self.reload_cap(src)
        return True

    def reload_cap(self, src):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(src)

    def handle_history(self):
        if self.posix.exists() and self.delete_post:
            self.posix.unlink()

    def post_frame_process(self, frame):
        pass

    def read(self, *args):
        """
        :param args:
        :return:
        """
        try:
            if self.status.get() == SystemStatus.SHUT_DOWN:
                self.__start__(*args)
            return True
        except Exception as e:
            logger.error(e)
            traceback.print_exc()
        # with self.read_lock:
        # frame = self.frame.copy()
        # grabbed = self.grabbed
        # return self.grabbed, self.frame

    def stop(self):
        self.status.set(SystemStatus.SHUT_DOWN)
        self.cap.release()
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


class VideoOfflineCapture(VideoCaptureThreading):
    """
    Read video frames from a offline video file
    """

    def __init__(self, video_path: Path, sample_path: Path, offline_path: Path, index_pool: Queue, frame_queue: Queue,
                 cfg: VideoConfig, idx, sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, sample_rate, width, height,
                         delete_post)
        self.offline_path = offline_path
        self.streams_list = list(self.offline_path.glob('*'))
        self.pos = -1

    def get_posix(self):
        """
        maintain a video file pointer,could loop a batch videos of a directory.
        :return:
        """
        if self.pos >= len(self.streams_list):
            logger.info('Load completely for [{}]'.format(str(self.offline_path)))
            return -1
        if self.cfg.cap_loop:
            return self.streams_list[0]
        self.pos += 1
        return self.streams_list[self.pos]

    def handle_history(self):
        """
        delete video file when reading done.
        :return:
        """
        if self.delete_post:
            self.posix.unlink()


class VideoOfflineCallbackCapture(VideoOfflineCapture):
    """
    Read video frames from a offline video file,
    """

    def __init__(self, video_path: Path, sample_path: Path, offline_path: Path, index_pool: Queue, frame_queue: Queue,
                 cfg: VideoConfig, idx, controller, shut_down_event, sample_rate=5, width=640, height=480,
                 delete_post=True):
        super().__init__(video_path, sample_path, offline_path, index_pool, frame_queue, cfg, idx, sample_rate, width,
                         height, delete_post)
        self.controller = controller
        self.shut_down_event = shut_down_event

    def pass_frame(self, *args):
        assert len(args) >= 2
        # self.controller.dispatch_frame(*args)
        self.controller.put_cache(*args)

    def cancel(self):
        super().cancel()
        if not self.shut_down_event.is_set():
            self.shut_down_event.set()


class VideoOfflineVlcCapture(VideoOfflineCallbackCapture):
    """
    Read video frames from a offline video file,
    """

    def reload_cap(self, src):
        threading.Thread(target=cap.run, args=(src, self.cfg.shape,), daemon=True).start()

    def read_frame(self):
        return cap.read()

    def cancel(self):
        super().cancel()
        cap.release()
        if not self.shut_down_event.is_set():
            self.shut_down_event.set()


class VideoOnlineSampleCapture(VideoCaptureThreading):
    """
    enhance video capture functionality, which could record or post handle a frame.
    it was designed to sample and write a frame into disks at special time.
    """

    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 idx,
                 sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, sample_rate, width, height,
                         delete_post)
        self.sample_cnt = 0
        self.sample_path.mkdir(exist_ok=True, parents=True)

    def handle_history(self):
        if int(self.runtime / 60 + 1) % self.cfg.sample_internal == 0:
            current_time = time.strftime('%m-%d-%H:%M-', time.localtime(time.time()))
            filename = os.path.basename(str(self.posix))
            target = self.sample_path / (current_time + filename)
            logger.info('Sample video stream into: [{}]'.format(target))
            shutil.copy(self.posix, target)
        super().handle_history()


# # @ray.remote
# TODO delete legacy code
# class VideoOfflineRayCapture(VideoCaptureThreading):
#     def __init__(self, video_path: Path, sample_path: Path, offline_path: Path, index_pool: Queue, frame_queue: Queue,
#                  cfg: VideoConfig, idx, sample_rate=5, width=640, height=480, delete_post=True):
#         super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, sample_rate, width, height,
#                          delete_post)
#         self.offline_path = offline_path
#         self.streams_list = list(self.offline_path.glob('*'))
#         self.pos = 0
#
#     def get_posix(self):
#         if self.pos >= len(self.streams_list):
#             logger.info('Load completely for [{}]'.format(str(self.offline_path)))
#             return -1
#         return self.streams_list[self.pos]

#
# TODO delete legacy code
# # @ray.remote(num_cpus=0.5)
# class VideoOnlineSampleBasedRayCapture(VideoCaptureThreading):
#     def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
#                  idx,
#                  controller_actor,
#                  sample_rate=5, width=640, height=480, delete_post=True):
#         super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, sample_rate, width, height,
#                          delete_post)
#         # if ray_index_pool is None:
#         #     raise Exception('Invalid index pool object id.')
#         # self.ray_index_pool = ray.get(ray_index_pool)
#         self.controller_actor = controller_actor
#         self.current = 0
#         self.stream_futures = []
#         # self.frame_queue = ray.get(frame_queue)
#
#     def pass_frame(self, frame):
#         # put the ray id of frame into global shared memory
#         # frame_id = ray.put(frame)
#         # self.frame_queue.put(frame_id)
#         self.stream_futures.append(self.controller_actor.start_stream_task.remote(frame))
#         logger.info('Passing frame [{}]'.format(self.current))
#         self.current += 1
#         # if self.current > 100:
#         #     logger.info('Blocked cap wait stream complete.')
#         #     ray.wait(self.stream_futures)
#         #     logger.info('Release cap.')
#         #     self.current = 0
#
#     def remote_update(self, src):
#         cnt = 0
#         start = time.time()
#         self.set_posix(src)
#         self.cap = cv2.VideoCapture(str(self.posix))
#         while True:
#             # with self.read_lock:
#             grabbed, frame = self.cap.read()
#             # logger.info('Video Capture [{}]: cnt ..'.format(cnt))
#             if not grabbed:
#                 # self.update_capture(cnt)
#                 break
#             if (cnt + 1) % 20 == 0:
#                 time.sleep(1)
#             if cnt % self.sample_rate == 0:
#                 self.pass_frame(frame)
#             cnt += 1
#             self.runtime = time.time() - start
#         self.handle_history()
#         self.cap.release()
#         # logger.info('Video Capture [{}]: cancel..'.format(self.cfg.index))
#         return self.posix
#
#     def set_posix(self, src):
#         self.posix = self.video_path / src
#

class VideoRtspCapture(VideoOnlineSampleCapture):
    """
     Read video frames from rtsp stream,could call back controller's method, passing the frame in a global cache.
    """

    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 idx,
                 sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, sample_rate, width, height,
                         delete_post)
        self.sample_path.mkdir(exist_ok=True, parents=True)
        self.saved_time = ""
        self.sample_cnt = 0
        self.frame_cnt = 0

    def load_next_src(self):
        logger.debug("Loading next video rtsp stream ....")
        return self.cfg.rtsp

    def handle_history(self):
        pass

    def update(self, *args):
        cnt = 0
        start = time.time()
        logger.info('*******************************Init video capture [{}]********************************'.format(
            self.cfg.index))
        # TODO delete legacy code
        # TODO refactor args structure
        ssd_detector = None
        classifier = None
        server_cfg = args[0]
        # if server_cfg.detect_mode == ModelType.SSD:
        #     ssd_detector = SSDDetector(model_path=server_cfg.detect_model_path, device_id=server_cfg.cd_id)
        #     ssd_detector.run()
        #     logger.info(
        #         f'*******************************Capture [{self.cfg.index}]: Running SSD Model********************************')
        # elif server_cfg.detect_mode == ModelType.CLASSIFY:
        #     classifier = DolphinClassifier(model_path=server_cfg.classify_model_path, device_id=server_cfg.dt_id)
        #     classifier.run()
        #     logger.info(
        #         f'*******************************Capture [{self.cfg.index}]: Running Classifier Model********************************')
        while self.status.get() == SystemStatus.RUNNING:
            # with self.read_lock:
            s = time.time()
            grabbed, frame = self.read_frame()
            e = 1 / (time.time() - s)
            # logger.info(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            # logger.info(self.cap.getRTPTimeStampTs())
            logger.debug(
                'Video capture [{}]: Receive Rate [{}]/FPS'.format(
                    self.cfg.index, round(e, 2)))
            s = time.time()
            if not grabbed:
                self.update_capture(cnt)
                cnt = 0
                continue
            # if cnt % self.sample_rate == 0:
            self.pass_frame(frame, args[0], ssd_detector, classifier)
            e = 1 / (time.time() - s)
            logger.debug(
                'Video capture [{}]: Operation Speed Rate [{}]/FPS'.format(
                    self.cfg.index, round(e, 2)))
            self.post_frame_process(frame)
            cnt += 1
            self.runtime = time.time() - start
            # if self.quit:
            #     self.cancel()
        # logger.info('Video Capture [{}]: cancel..'.format(self.cfg.index))
        logger.info(
            '*******************************Video capture [{}] exit********************************'.format(
                self.cfg.index))

    def post_frame_process(self, frame):
        self.sample_cnt += 1
        if self.sample_cnt % self.cfg.rtsp_saved_per_frame and self.cfg.enable_sample_frame:
            current_time = time.strftime('%m-%d-%H-%M-', time.localtime(time.time()))
            self.sample_cnt = 0
            # if current_time != self.saved_time:
            #     self.sample_cnt = 0
            # self.saved_time = current_time
            self.frame_cnt += 1
            target = self.sample_path / (current_time + str(self.frame_cnt) + '.png')
            logger.info("Sample rtsp video stream into: [{}]".format(target))
            cv2.imwrite(str(target), frame)


class VideoRtspCallbackCapture(VideoRtspCapture):
    """
    Read video frames from rtsp stream,could call back controller's method, passing frame in a global cache
    """

    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 idx,
                 controller,
                 sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, sample_rate, width, height,
                         delete_post)
        self.controller = controller

    def pass_frame(self, *args):
        assert len(args) >= 2
        # self.controller.dispatch_frame(*args)
        self.controller.put_cache(*args)


class VideoRtspVlcCapture(VideoRtspCallbackCapture):
    def __init__(self, video_path: Path, sample_path: Path, index_pool: Queue, frame_queue: Queue, cfg: VideoConfig,
                 idx, controller, sample_rate=5, width=640, height=480, delete_post=True):
        super().__init__(video_path, sample_path, index_pool, frame_queue, cfg, idx, controller, sample_rate, width,
                         height, delete_post)

    def reload_cap(self, src):
        threading.Thread(target=cap.run, args=(src, self.cfg.shape,), daemon=True).start()

    def read_frame(self):
        return cap.read()

    def cancel(self):
        super().cancel()
        cap.release()

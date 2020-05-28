#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: rtsp.py
@time: 2/25/20 3:16 PM
@version 1.0
@desc:
"""
import subprocess
import threading
import traceback
from multiprocessing import Manager
from threading import Thread
from typing import List

import numpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os
import subprocess as sp
from moviepy.compat import PY3, DEVNULL
from moviepy.config import get_setting
import cv2
import time

import numpy as np
from config import VideoConfig, SystemStatus
from utils import logger, bbox_points, generate_time_stamp


class FFMPEG_VideoStreamer(FFMPEG_VideoWriter):

    def __init__(self, rtsp_addr, size, fps, filename='default.mp4', codec="libx264", audiofile=None, preset="medium",
                 bitrate=None,
                 withmask=False, logfile=None, threads=None, ffmpeg_params=None):
        super().__init__(filename, size, fps, codec, audiofile, preset, bitrate, withmask, logfile, threads,
                         ffmpeg_params)
        if logfile is None:
            logfile = sp.PIPE

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]

        # order is important
        cmd = [
            get_setting("FFMPEG_BINARY"),
            '-y',
            '-loglevel', 'error' if logfile == sp.PIPE else 'info',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgba' if withmask else 'rgb24',
            '-s', '%dx%d' % (size[0], size[1]),
            '-r', '%.02f' % fps,
            '-i', '-', '-an',
            '-f', 'rtsp'
        ]
        if audiofile is not None:
            cmd.extend([
                '-i', audiofile,
                '-acodec', 'copy'
            ])
        cmd.extend([
            '-vcodec', codec,
            '-preset', preset,
        ])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend([
                '-b', bitrate
            ])

        if threads is not None:
            cmd.extend(["-threads", str(threads)])
        #
        # if ((codec == 'libx264') and
        #         (size[0] % 2 == 0) and
        #         (size[1] % 2 == 0)):
        #     cmd.extend([
        #         '-pix_fmt', 'yuv420p'
        #     ])
        # cmd.extend([
        #     filename
        # ])
        cmd.extend([
            rtsp_addr
        ])

        popen_params = {"stdout": DEVNULL,
                        "stderr": logfile,
                        "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        self.proc = sp.Popen(cmd, **popen_params)


class FFMPEG_MP4Writer(FFMPEG_VideoWriter):

    def write(self, frame):
        self.write_frame(frame)

    def release(self):
        self.close()

    def __init__(self, filename, size, fps, codec="libx264", audiofile=None, preset="ultrafast", bitrate=None,
                 withmask=False, logfile=None, threads=2, ffmpeg_params=None):
        if logfile is None:
            logfile = sp.PIPE

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]

        # order is important
        cmd = [
            get_setting("FFMPEG_BINARY"),
            '-y',
            '-loglevel', 'error' if logfile == sp.PIPE else 'info',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (size[0], size[1]),
            '-pix_fmt', 'rgba' if withmask else 'rgb24',
            '-r', '%.02f' % fps,
            '-i', '-', '-an',
        ]
        if audiofile is not None:
            cmd.extend([
                '-i', audiofile,
                '-acodec', 'copy'
            ])
        cmd.extend([
            '-vcodec', codec,
            '-preset', preset,
        ])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend([
                '-b', bitrate
            ])

        if threads is not None:
            cmd.extend(["-threads", str(threads)])

        if ((codec == 'libx264') and
                (size[0] % 2 == 0) and
                (size[1] % 2 == 0)):
            cmd.extend([
                '-pix_fmt', 'yuv420p'
            ])
        cmd.extend([
            filename
        ])

        popen_params = {"stdout": DEVNULL,
                        "stderr": logfile,
                        "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        self.proc = sp.Popen(cmd, **popen_params)

        # super().__init__(filename, size, fps, codec, audiofile, preset, bitrate, withmask, logfile, threads,
        #                  ffmpeg_params)


class Live(object):
    def __init__(self, ffmpeg_path, file_path, rtsp_url, proto="udp", logger=None):
        '''
        :param ffmpeg_path: ffmpeg path
        :param file_path: path of video needed to be sended
        :param rtsp_url: transfer stream url
        :param proto: udp or tcp
        '''
        self.ffmpeg = ffmpeg_path
        self.rtsp_url = rtsp_url
        self.file_path = file_path
        self.protocol = proto
        self.logger = logger

        self.command = "{} -re -i {} -rtsp_transport {} -vcodec h264 -f rtsp {}".format(
            self.ffmpeg,
            self.file_path,
            self.protocol,
            self.rtsp_url)

        self.thread = None
        self.thread_running = False

    def read_frame(self):
        pass

    def push_frame(self):
        pass

    def execute_com(self):
        '''
        :return:
        '''

        # self.logger.info(command)
        ffmpeg_p = subprocess.Popen(self.command.split(' '), stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)

        output, _ = ffmpeg_p.communicate()
        print(output)

    def run(self):
        '''
        :return:
        '''
        if self.thread is None:
            self.thread = Thread(target=self.execute_com, )
        if self.thread_running is False:
            self.thread.start()
            self.thread_running = True

    def stop(self):
        '''
        :return:
        '''

        if self.thread_running is True:
            self.thread.join()
            self.thread_running = False


def push_rtsp():
    video_path = "12.mp4"
    test_video_path = '/data/lxd/project/DolphinDetection/data/offline/6x/22_1080P.mp4'
    video_cap = cv2.VideoCapture(test_video_path)
    # rtsp_url = "rtsp://221.226.81.54:6006/test1"
    # rtsp_url = "rtsp://221.226.81.54:6006/test1"
    rtsp_url = "rtsp://192.168.0.116/test1"
    # size = (1920, 1080)
    size = (1920, 1080)
    streamer = FFMPEG_VideoStreamer(rtsp_url, codec='h264', fps=24, size=size)
    # print('~~~~~~~~~~~~~~')
    while True:
        try:
            grabbed, frame = video_cap.read()
            if grabbed:
                # time.sleep(0.2)
                streamer.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        except Exception as e:
            print(e)
            video_cap.release()
    video_cap.release()


video_streamer = FFMPEG_VideoStreamer("rtsp://192.168.0.116/test2", size=(1920, 1072), fps=24, codec='h264', )


def push_by_live():
    ffmpeg_path = "ffmpeg"
    test_video_path = '/Users/luvletteru/Documents/GitHub/DolphinDetection/data/offline/6/22_1080P.mp4'
    rtsp_url = "rtsp://221.226.81.54:6006/test1"
    live = Live(ffmpeg_path, test_video_path, rtsp_url)
    live.run()


if __name__ == "__main__":
    # push_by_live()

    # video_path = "11.mp4"
    push_rtsp()
    # rtsp_url = "rtsp://192.168.0.116/test2"

    # live = Live(ffmpeg_path, test_video_path, rtsp_url)
    # live.run()


class PushStreamer(object):
    def __init__(self, cfg: VideoConfig, stream_stack: List) -> None:

        super().__init__()
        self.cfg = cfg
        self.stream_stack = stream_stack
        self.LOG_PREFIX = f'Push Streamer [{self.cfg.index}]: '
        self.quit = Manager().Event()
        self.quit.clear()
        self.status = Manager().Value('i', SystemStatus.RUNNING)

    def listen(self):
        if self.quit.wait():
            self.status.set(SystemStatus.SHUT_DOWN)
            # self.stream_render.quit.set()

    def push_stream(self):
        logger.info(
            f'*******************************Controller [{self.cfg.index}]: Init push stream service********************************')
        draw_cnt = 0
        tmp_results = []
        video_streamer = FFMPEG_VideoStreamer(self.cfg.push_to, size=(self.cfg.shape[1], self.cfg.shape[0]), fps=25,
                                              codec='h264', )
        video_streamer.write_frame(np.zeros((self.cfg.shape[1], self.cfg.shape[0], 3), dtype=np.uint8))
        # time.sleep(6)
        pre_index = 0
        threading.Thread(target=self.listen, daemon=True).start()
        while self.status.get() == SystemStatus.RUNNING:
            try:
                ps = time.time()
                # if self.status.get() == SystemStatus.SHUT_DOWN:
                #     video_streamer.close()
                #     break
                # se = 1 / (time.time() - ps)
                # logger.debug(self.LOG_PREFIX + f'Get Signal Speed Rate: [{round(se, 2)}]/FPS')
                # gs = time.time()
                if self.cfg.use_sm:
                    # 4K frame has large size, get it from shared memory instead pipe serialization between
                    # multi-processes
                    frame = self.stream_stack[0][0]
                    # current index and other info are smaller than frame buffer,so we can use Manager().list()
                    if not len(self.stream_stack[1]):
                        continue
                    proc_res, frame_index = self.stream_stack[1].pop()
                else:
                    # if frame shape is 4K, obtain FPS from Manager().list is around 10,which is over slow than video fps(25)
                    # it causes much latent when pushing stream
                    if not len(self.stream_stack):
                        continue
                    frame, proc_res, frame_index = self.stream_stack.pop()
                    logger.debug(f'Push Streamer [{self.cfg.index}]: Cache queue size: [{len(self.stream_stack)}]')

                if not self.cfg.use_sm and len(self.stream_stack) > 1000:
                    self.stream_stack[:] = []
                    logger.info(self.LOG_PREFIX + 'Too much frames blocked in stream queue.Cleared')
                    continue

                if pre_index < frame_index:
                    pre_index = frame_index
                else:
                    continue

                # end = 1 / (time.time() - gs)
                # logger.debug(self.LOG_PREFIX + f'Get Frame Speed Rate: [{round(end, 2)}]/FPS')
                detect_flag = (proc_res is not None and proc_res.detect_flag)
                # logger.info(f'Draw cnt: [{draw_cnt}]')
                # if proc_res is not None:
                #     logger.info(f'Detect flag: [{proc_res.detect_flag}]')
                # ds = time.time()
                if detect_flag:
                    # logger.info('Detect flag~~~~~~~~~~')
                    draw_cnt = 0
                    tmp_results = proc_res.results
                is_draw_over = draw_cnt <= 36
                if is_draw_over:
                    # logger.info('Draw next frames~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    for r in tmp_results:
                        for rect in r.rects:
                            color = np.random.randint(0, 255, size=(3,))
                            color = [int(c) for c in color]
                            p1, p2 = bbox_points(self.cfg, rect, frame.shape)
                            # p1 = (int(rect[0]), int(rect[1]))
                            # p2 = (int(rect[2]), int(rect[3]))

                            cv2.putText(frame, 'Asaeorientalis', p1,
                                        cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                            cv2.rectangle(frame, p1, p2, color, 2)
                            # if self.server_cfg.detect_mode == ModelType.SSD:
                            #     cv2.putText(frame, str(round(r[4], 2)), (p2[0], p2[1]),
                            #                 cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
                    draw_cnt += 1
                if self.cfg.write_timestamp:
                    time_stamp = generate_time_stamp("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, time_stamp, (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # de = 1 / (time.time() - ds)
                # logger.debug(self.LOG_PREFIX + f'Draw Speed Rate: [{round(de, 2)}]/FPS')
                # logger.info(f'Frame index [{frame_index}]')
                # if frame_index % self.cfg.sample_rate == 0:
                #     for _ in range(2):
                #         video_streamer.write_frame(frame)
                # else:
                #     video_streamer.write_frame(frame)
                # end = 1 / (time.time() - ps)
                # ws = time.time()
                video_streamer.write_frame(frame)
                # w_end = 1 / (time.time() - ws)
                end = 1 / (time.time() - ps)
                # logger.debug(self.LOG_PREFIX + f'Writing Speed Rate: [{round(w_end, 2)}]/FPS')
                logger.info(f'Streamer [{self.cfg.index}]: Streaming Speed Rate: [{round(end, 2)}]/FPS')
            except Exception as e:
                logger.error(e)
                traceback.print_stack()
                # logger.warning(e)
        logger.info(
            '*******************************Controller [{}]:  Push stream service exit********************************'.format(
                self.cfg.index))

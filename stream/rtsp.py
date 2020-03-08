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
from threading import Thread
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os
import subprocess as sp
from moviepy.compat import PY3, DEVNULL
from moviepy.config import get_setting
import cv2
import time


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
        self.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def release(self):
        self.close()

    def __init__(self, filename, size, fps, codec="libx264", audiofile=None, preset="medium", bitrate=None,
                 withmask=False, logfile=None, threads=None, ffmpeg_params=None):
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

#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: app.py
@time: 11/22/19 11:11 AM
@version 1.0
@desc:
"""

import argparse
import time
import warnings

import detection.monitor
from interface import *
# from multiprocessing import Process
from utils import sec2time
# from apscheduler.schedulers.background import BackgroundScheduler
from utils.scheduler import ClosableBlockingScheduler

warnings.filterwarnings("ignore")


# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.schedulers import blocking


class DolphinDetectionServer:

    def __init__(self, cfg: ServerConfig, vcfgs: List[VideoConfig], switcher_options, cd_id, dt_id) -> None:
        self.cfg = cfg
        self.cfg.cd_id = cd_id
        self.cfg.dt_id = dt_id
        self.vcfgs = [c for c in vcfgs if switcher_options[str(c.index)]]
        if not len(self.vcfgs):
            raise Exception('Empty video stream configuration enabling.')
        # self.classifier = DolphinClassifier(model_path=self.cfg.classify_model_path, device_id=cd_id)
        self.classifier = None
        self.ssd_detector = None
        self.dt_id = dt_id
        logger.setLevel(self.cfg.log_level)
        self.monitor = detection.monitor.EmbeddingControlBasedTaskMonitor(self.vcfgs, self.cfg,
                                                                          self.cfg.stream_save_path,
                                                                          self.cfg.sample_save_dir,
                                                                          self.cfg.frame_save_dir,
                                                                          self.cfg.candidate_save_dir,
                                                                          self.cfg.offline_stream_save_dir)
        self.scheduler = ClosableBlockingScheduler(stop_event=self.monitor.shut_down_event)
        # self.http_server = HttpServer(self.cfg.http_ip, self.cfg.http_port, self.cfg.env, self.cfg.candidate_save_dir)
        if not self.cfg.run_direct:
            self.scheduler.add_job(self.monitor.monitor, 'cron',
                                   month=self.cfg.cron['start']['month'],
                                   day=self.cfg.cron['start']['day'],
                                   hour=self.cfg.cron['start']['hour'],
                                   minute=self.cfg.cron['start']['minute'])

    def run(self):
        """
        System Entry
        """
        # ray.init()
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(start_time))
        logger.info(
            f'*******************************Dolphin Detection System: Running Environment [{self.cfg.env}] at '
            f'[{start_time_str}]********************************')
        # self.http_server.run()
        if self.cfg.run_direct:
            self.monitor.monitor()
        else:
            self.scheduler.start()
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(end_time))
        run_time = sec2time(end_time - start_time)
        logger.info(
            f'*******************************Dolphin Detection System: Shut down at [{end_time_str}].'
            f'Total Running Time '
            f'[{run_time}]********************************')


def load_cfg(args):
    if args.env is not None:
        if args.env == Environment.DEV:
            args.cfg = 'vcfg/dev/server.yml'
        if args.env == Environment.TEST:
            args.cfg = 'vcfg/test/server.yml'
        if args.env == Environment.PROD:
            args.cfg = 'vcfg/prod/server.yml'
        args.vcfg = get_video_cfgs(args.env)
    server_config = load_server_config(args.cfg)
    video_config = load_video_config(args.vcfg)
    switcher_options = load_yaml_config(args.sw)
    if args.log_level is not None:
        server_config.log_level = args.log_level
    if args.http_ip is not None:
        server_config.http_ip = args.http_ip
    if args.http_port is not None:
        server_config.http_port = args.http_port
    if args.wc_ip is not None:
        server_config.wc_ip = args.wc_ip
    if args.wc_port is not None:
        server_config.wc_port = args.wc_port
    if args.root is not None:
        server_config.set_root(args.root)
    if args.cdp is not None:
        server_config.set_candidate_save_dir(args.cdp)
    if args.run_direct is not None:
        server_config.run_direct = args.run_direct
    if args.send_msg is not None:
        server_config.send_msg = args.send_msg
    if args.enable is not None:
        enables = args.enable.split(',')
        for e in enables:
            switcher_options[e] = True
    if args.disable is not None:
        disables = args.disable.split(',')
        for d in disables:
            switcher_options[d] = False
    if args.use_sm is not None:
        enables = args.use_sm.split(',')
        enables = [int(e) for e in enables]
        for cfg in video_config:
            if cfg.index in enables:
                cfg.use_sm = True

    if args.enable_forward_filter is not None:
        enables = args.enable_forward_filter.split(',')
        enables = [int(e) for e in enables]
        for cfg in video_config:
            if cfg.index in enables:
                cfg.forward_filter = True

    if args.enable_post_filter is not None:
        enables = args.enable_post_filter.split(',')
        enables = [int(e) for e in enables]
        for cfg in video_config:
            if cfg.index in enables:
                cfg.post_filter = True

    if args.push_stream is not None:
        enables = args.push_stream.split(',')
        enables = [int(e) for e in enables]
        for cfg in video_config:
            if cfg.index in enables:
                cfg.push_stream = True

    if args.show_windows is not None:
        enables = args.show_windows.split(',')
        enables = [int(e) for e in enables]
        for cfg in video_config:
            if cfg.index in enables:
                cfg.show_window = True

    if args.limit_freq is not None:
        enables = args.limit_freq.split(',')
        enables = [int(e) for e in enables]
        for cfg in video_config:
            if cfg.index in enables:
                cfg.limit_freq = True
    return server_config, video_config, switcher_options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='dev',
                        help='System environment.')
    parser.add_argument('--log_level', default='INFO', help='control log output level')
    parser.add_argument('--cfg', type=str, default='vcfg/server-dev.json',
                        help='Server configuration file represented by json format.')
    parser.add_argument('--vcfg', type=str, default='vcfg/video-dev.json',
                        help='Video configuration file represented by json format.')
    parser.add_argument('--sw', type=str, default='vcfg/switcher.yml',
                        help='Control video switcher')
    parser.add_argument('--http_ip', type=str, help='Http server ip address')
    parser.add_argument('--http_port', type=int, help='Http server listen port')
    parser.add_argument('--wc_ip', type=str, help='Websocket server ip address')
    parser.add_argument('--wc_port', type=int, help='Websocket server listen port')
    parser.add_argument('--root', type=str,
                        help='Redirect data root path,default path is [PROJECT DIR/data/...].' +
                             'PROJECT DIR is depend on which project is deployed.' +
                             'If root param is set,data path is redirected as [root/data/...]')
    parser.add_argument('--cdp', type=str,
                        help='Dolphin video stream storage relative path.default path is [$PROJECT DIR$/data/candidate].' +
                             'If cdp param is set,stream storage path is redirected as [$root$/$cdp$] ' +
                             'or [$PROJECT DIR$]/$cdp$.')
    parser.add_argument('--cd_id', type=int, default=1, help='classifier GPU device id')
    parser.add_argument('--dt_id', type=int, default=2, help='detection GPU device id')
    parser.add_argument('--run_direct', action='store_true', default=False,
                        help='timing start or run directly.default is False,system will be blocked until time arrivals.')
    parser.add_argument('--enable', type=str, default="5",
                        help='Enable video using index,should input a index at least.')
    parser.add_argument('--enable_forward_filter', type=str, default=None,
                        help='Enable forward filter, default forward filter services of all monitors are disabled.')
    parser.add_argument('--enable_post_filter', type=str, default=None,
                        help='Enable post filter, default post filter services of all monitors are disabled.')
    parser.add_argument('--disable', type=str, default=None,
                        help='Disable video using index,default all videos are disabled.')
    parser.add_argument('--send_msg', action='store_true', default=False,
                        help='send detection msg to websocket server or not')
    parser.add_argument('--use_sm', default=None, help='use share memory to cache frames,default uses slower'
                                                       'Queue() as caches')
    parser.add_argument('--push_stream', default=None, help='push stream or not')
    parser.add_argument('--show_windows', default=None, help='display windows')
    parser.add_argument('--limit_freq', default=None, help='enable frequency of gerneration control')
    args = parser.parse_args()
    server_config, video_config, switcher_options = load_cfg(args)
    server = DolphinDetectionServer(server_config, video_config, switcher_options, args.cd_id, args.dt_id)
    server.run()

# process = Process(target=monitor.monitor)
# process.start()
# process.join()

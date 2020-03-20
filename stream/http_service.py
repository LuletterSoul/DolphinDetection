#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: http.py
@time: 2/7/20 11:43 AM
@version 1.0
@desc:
"""
import argparse
from multiprocessing import Process

from flask import Flask

from config import Environment

app = Flask(__name__)


class HttpServer(object):

    def __init__(self, host_ip="127.0.0.1", host_port="8080", env='dev', root="data/candidates"):
        self.host_ip = host_ip
        self.host_port = host_port
        self.env = env
        self.set_root(root)

    @staticmethod
    @app.route('/video/<date>/<channel>/<v_id>', methods=['GET'])
    def video(date, channel, v_id):
        # channel = request.args.get('channel')
        # date = request.args.get("date")
        # video_id = request.args.get("v_id")
        url = "{}/{}/render-streams/{}".format(date, channel, v_id)
        return app.send_static_file(url)

    @staticmethod
    @app.route('/', methods=['GET'])
    def root():
        return "Welcome to dolphin detection system!"

    @staticmethod
    def set_root(root):
        app.static_folder = root

    def run(self):
        if self.env == Environment.DEV:
            Process(target=app.run, args=(self.host_ip, self.host_port, False,), daemon=True).start()
        elif self.env == Environment.TEST:
            Process(target=app.run, args=(self.host_ip, self.host_port, False,), daemon=True).start()
        elif self.env == Environment.PROD:
            Process(target=app.run, args=(self.host_ip, self.host_port, False,), daemon=True).start()

    def run_front(self):
        if self.env == Environment.DEV:
            app.run(self.host_ip, self.host_port, True)
            # Process(target=app.run, args=(self.host_ip, self.host_port, False,), daemon=True).start()
        elif self.env == Environment.TEST:
            app.run(self.host_ip, self.host_port, True)
            # Process(target=app.run, args=(self.host_ip, self.host_port, False,), daemon=True).start()
        elif self.env == Environment.PROD:
            app.run(self.host_ip, self.host_port, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='prod',
                        help='System environment.')
    parser.add_argument('--http_ip', type=str, default="10.196.122.94", help='Http server ip address')
    parser.add_argument('--http_port', type=int, default=8080, help='Http server listen port')
    parser.add_argument('--root', type=str, default="data/candidates", help='Http server listen port')
    args = parser.parse_args()
    http = HttpServer(args.http_ip, args.http_port)
    http.run_front()

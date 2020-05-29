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
import os
import re
from multiprocessing import Process

import shutil
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path

from config import Environment
from utils import logger

app = Flask(__name__)
cors = CORS(app)


def copy_from_src(filenames, source_origin_dir, source_render_dir, target_original_dir, target_render_dir,
                  sample_type):
    """
    copy generated videos from source directory to the another directory for classification by WEB request
    :param filenames:
    :param source_origin_dir:
    :param source_render_dir:
    :param target_original_dir:
    :param target_render_dir:
    :param sample_type:
    :return:
    """
    target_render_type_dir = target_render_dir / sample_type
    target_origin_type_dir = target_original_dir / sample_type
    if not target_render_type_dir.exists():
        target_render_type_dir.mkdir(parents=True, exist_ok=True)
    if not target_origin_type_dir.exists():
        target_origin_type_dir.mkdir(parents=True, exist_ok=True)
    for f in filenames:
        file_origin = source_origin_dir / f
        file_render = source_render_dir / f
        if file_origin.exists():
            shutil.copy(file_origin, target_origin_type_dir)
            logger.info(f'Copy {str(file_origin)} into {str(target_original_dir)}')
        if file_render.exists():
            shutil.copy(file_render, target_render_type_dir)
            logger.info(f'Copy {str(file_origin)} into {str(target_render_type_dir)}')


class VideoType:
    ORIGIN = 'original-streams'
    RENDER = 'render-streams'


def query_directory(dir_name):
    """
    return a directory folder tree recursively, exit if meet a file endpoint.
    :param dir_name:
    :return:
    """
    sun_dir_names = [l for l in os.listdir(dir_name) if not l.startswith('.')]
    if not len(sun_dir_names):
        return []
    sun_dir_names = sort_humanly(sun_dir_names)
    return [{'value': d, 'label': d, 'children': query_directory(os.path.join(dir_name, d))} for d in sun_dir_names if
            os.path.isdir(os.path.join(dir_name, d))]


def query_directory_child_list(dir_name):
    """
    return a directory folder tree recursively, exit if meet a file endpoint.
    :param dir_name:
    :return:
    """
    sun_dir_names = [l for l in os.listdir(dir_name) if not l.startswith('.')]
    if not len(sun_dir_names):
        return []
    sun_dir_names = sort_humanly(sun_dir_names)
    return [{'content': d, 'label': sort_humanly([str(ls) for ls in list(Path(os.path.join(dir_name, d)).glob('*'))])}
            for d in
            sun_dir_names]


def tryint(s):  # 将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):  # 将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):  # 以分割后的list为单位进行排序
    """
    sort list strings according string and number
    :param v_list:
    :return:
    """
    return sorted(v_list, key=str2int, reverse=True)


class HttpServer(object):

    def __init__(self, host_ip="127.0.0.1", host_port="8080", env='dev', root="data/candidates"):
        self.host_ip = host_ip
        self.host_port = host_port
        self.env = env
        self.set_root(root)

    @staticmethod
    @app.route('/video/<date>/<channel>/<v_id>', methods=['GET'])
    def video(date, channel, v_id: str):
        """
        default return render videos
        :param date:
        :param channel:
        :param v_id:
        :return:
        """
        if not v_id.endswith('.mp4'):
            return 'Not supported file format.Must be mp4 file.'
        url = "{}/{}/render-streams/{}".format(date, channel, v_id)
        return app.send_static_file(url)

    @staticmethod
    @app.route('/video/<date>/<channel>/<video_type>/<v_id>', methods=['GET'])
    def video_by_type(date, channel, video_type, v_id: str):
        """
        can choose different video types
        :param date:
        :param channel:
        :param video_type:
        :param v_id:
        :return:
        """
        if not v_id.endswith('.mp4'):
            return 'Not supported file format.Must be mp4 file.'
        url = "{}/{}/{}/{}".format(date, channel, video_type, v_id)
        return app.send_static_file(url)

    @staticmethod
    @app.route('/preview/<date>/<channel>/<p_id>', methods=['GET'])
    def preview(date, channel, p_id: str):
        """
        can choose different video types
        :param date:
        :param channel:
        :param p_id:
        :return:
        """
        url = "{}/{}/preview/{}".format(date, channel, p_id)
        return app.send_static_file(url)

    @staticmethod
    @app.route('/files/<date>/<channel>/<video_type>', methods=['POST'])
    def post_samples_classification(date, channel, video_type):
        """
        can choose different video types
        :param date:
        :param channel:
        :param video_type:
        :param v_id:
        :return:
        """
        json_data = json.loads(request.get_data().decode('utf-8'))
        filenames = json_data['filenames']
        sample_type = json_data['sample_type']

        root = Path(app.static_folder)
        data_root = os.path.dirname(app.static_folder)

        print(f'Data root {data_root}')

        source_render_dir = root / date / channel / VideoType.RENDER
        source_origin_dir = root / date / channel / VideoType.ORIGIN

        target_sample_dir = Path(os.path.join(data_root, 'samples'))

        target_render_dir = target_sample_dir / date / channel / VideoType.RENDER
        target_original_dir = target_sample_dir / date / channel / VideoType.ORIGIN

        copy_from_src(filenames, source_origin_dir, source_render_dir, target_original_dir,
                      target_render_dir, sample_type)
        return jsonify(filenames)

    @staticmethod
    @app.route('/', methods=['GET'])
    def root():
        return "Welcome to dolphin detection system!"

    @staticmethod
    @app.route('/files', methods=['GET'])
    def folders():
        return jsonify(query_directory(app.static_folder))

    @staticmethod
    @app.route('/files/<date>/<channel>/<video_type>', methods=['GET'])
    def video_filenames(date, channel, video_type):
        file_names = os.listdir(os.path.join(app.static_folder, date, channel, video_type))
        file_names = sort_humanly(file_names)
        return jsonify(file_names)

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

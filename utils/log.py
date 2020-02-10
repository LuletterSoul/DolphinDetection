#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: log.py
@time: 2019/11/15 21:09
@version 1.0
@desc:
"""
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from config import LOG_DIR, LOG_LEVER

from .common import generate_time_stamp


def setup_logging(log_dir, logger_name='main', level=logging.DEBUG):
    log_dir = log_dir / generate_time_stamp()
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file_format = '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    log_console_format = '[%(levelname)s]: %(message)s'

    # Main logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_info_handler = RotatingFileHandler(str(log_dir / 'info.log'), maxBytes=10 ** 6, backupCount=5)
    exp_file_info_handler.setLevel(logging.INFO)
    exp_file_info_handler.setFormatter(Formatter(log_file_format))

    exp_file_debug_handler = RotatingFileHandler(str(log_dir / 'debug.log'), maxBytes=10 ** 6, backupCount=5)
    exp_file_debug_handler.setLevel(logging.DEBUG)
    exp_file_debug_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(str(log_dir / 'error.log').format(log_dir), maxBytes=10 ** 6,
                                                  backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    logger.addHandler(console_handler)
    logger.addHandler(exp_file_debug_handler)
    logger.addHandler(exp_errors_file_handler)
    return logger


logger = setup_logging(LOG_DIR, level=LOG_LEVER)

#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: ClosableBlockingScheduler.py
@time: 3/10/20 12:50 AM
@version 1.0
@desc:
"""

from apscheduler.schedulers.blocking import BlockingScheduler
import threading


class ClosableBlockingScheduler(BlockingScheduler):

    def __init__(self, gconfig={}, stop_event=None, **options):
        super().__init__(gconfig, **options)
        self.stop_event = stop_event
        threading.Thread(target=self.listen, daemon=True).start()

    def listen(self):
        if self.stop_event.wait():
            if self.running:
                self.shutdown()

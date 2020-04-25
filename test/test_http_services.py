#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test_http_services.py
@time: 3/24/20 9:54 AM
@version 1.0
@desc:
"""

from stream.http_service import query_directory

if __name__ == '__main__':
    root = '../data/candidates'
    print(query_directory(root))

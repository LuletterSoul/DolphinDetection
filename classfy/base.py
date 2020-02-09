#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: transform.py
@time: 2/6/20 9:16 AM
@version 1.0
@desc:
"""

from torchvision import transforms

train_trainsforms = transforms.Compose(
    [transforms.Resize(244), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])

test_trainsforms = transforms.Compose(
    [transforms.Resize(244), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])

to_pil = transforms.ToPILImage()

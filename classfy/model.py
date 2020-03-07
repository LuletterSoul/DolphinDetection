#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: test.py
@time: 2/6/20 9:15 AM
@version 1.0
@desc:
"""
import os

import torch

from classfy.base import *
from config import PROJECT_DIR
from utils import logger
from pathlib import Path
from utils import normalization

"""
    =========================== 模型预测与使用 ================================
"""
# 测试数据集，设置图像大小与训练一项
data_dir2 = os.path.join(PROJECT_DIR, 'data/test/0206')


# test_trainsforms = transforms.Compose([transforms.Resize((224, 224)),
#                                        transforms.ToTensor(), ])
class DolphinClassifier(object):

    def __init__(self, model_path: Path, device_id='1') -> None:
        self.model_path = model_path
        self.device = None
        self.model = None
        self.device_id = device_id

    def run(self):
        # if self.device_id is not None:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = self.device_id
        if not self.model_path.exists():
            raise Exception(f'Model init failed: model not exist at [{str(self.model_path)}].')

        if self.device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # self.device = torch.device("cuda:" + str(self.device_id))
            self.model = torch.load(str(self.model_path))
        else:
            self.device = torch.device("cpu")
            self.model = torch.load(str(self.model_path), map_location="cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(self.model)
        print(self.device)

    def predict(self, image):
        image = to_pil(image)
        image_tensor = test_trainsforms(image).float()
        image_tensor = image_tensor.unsqueeze(0)
        input = image_tensor.to(self.device)
        output = self.model(input)
        output = output.data.cpu().numpy()
        index = output.argmax()
        return index, normalization(output)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# classify_model = torch.load(os.path.join(PROJECT_DIR, 'classify_model/bc-classify_model.pth'))
# classify_model.eval()
# print(classify_model)


# def predict(image):
#     image = to_pil(image)
#     image_tensor = test_trainsforms(image).float()
#     image_tensor = image_tensor.unsqueeze(0)
#     print(image_tensor.size())
#     input = image_tensor.to(device)
#     output = classify_model(input)
#     index = output.data.cpu().numpy().argmax()
#     return index
#
#
# # 获取随机图像
# def get_random_images(num):
#     data = datasets.ImageFolder(data_dir2, transform=test_trainsforms)
#     classes = data.classes
#     print("classes:", classes)
#
#     indices = list(range(len(data)))
#     np.random.shuffle(indices)
#     idx = indices[:num]  # 获取随机的预测数据
#
#     from torch.utils.data.sampler import SubsetRandomSampler
#     sampler = SubsetRandomSampler(idx)  # 获取的预测数据再次打乱
#     loader = DataLoader(data, sampler=sampler, batch_size=num)
#     dataiter = iter(loader)
#
#     images, labels = dataiter.next()
#     return images, labels, classes

# if __name__ == '__main__':
#     images, labels, classes = get_random_images(10)
#     fig = plt.figure(figsize=(10, 10))
#     for i in range(len(images)):
#         index = predict(images[i])
#         # sub = fig.add_subplot(1, len(images), i + 1)
#         res = int(labels[i]) == index
#         print(index)
#         # str(sub.set_title(str(classes[index]))) + ":" + str(res)
#         # plt.axis('off')
#         # plt.imshow(image)
#     # plt.show()

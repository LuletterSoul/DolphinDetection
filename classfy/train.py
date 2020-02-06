#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: classfy.py
@time: 2/4/20 9:19 PM
@version 1.0
@desc:
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from config import PROJECT_DIR
from utils import logger
from classfy.base import *

"""
    =========================== 模型训练与验证 ================================
"""
# 定义train/validation数据集加载器

data_dir = os.path.join(PROJECT_DIR, 'data/class')


def load_split_train_test(datadir, valid_size=0.2):
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    train_data = datasets.ImageFolder(datadir, transform=train_trainsforms)
    # print("train_data大小：",train_data[0][0].size())       # 查看resize(确保图像都有3通道)
    test_data = datasets.ImageFolder(datadir, transform=test_trainsforms)

    num_train = len(train_data)  # 训练集数量
    indices = list(range(num_train))  # 训练集索引

    split = int(np.floor(valid_size * num_train))  # 获取20%数据作为验证集
    np.random.shuffle(indices)  # 打乱数据集

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]  # 获取训练集，测试集
    train_sampler = SubsetRandomSampler(train_idx)  # 打乱训练集，测试集
    test_sampler = SubsetRandomSampler(test_idx)

    # ============数据加载器：加载训练集，测试集===================
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=64, num_workers=4)
    return train_loader, test_loader


train_loader, test_loader = load_split_train_test(data_dir, 0.2)
print(train_loader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = models.resnet50(pretrained=True)
# print(model)

"""
   首先，我们必须冻结预训练过的层，因此在训练期间它们不会进行反向传播。
   然后，我们重新定义最后的全连接层，即使用我们的图像来训练的图层。
        我们还创建了标准（损失函数）并选择了一个优化器（在这种情况下为Adam）和学习率。
"""
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 2),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.002)
model.to(device)

epochs = 20
steps = 0
running_loss = 0
train_losses, test_losses = [], []

for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        steps += 1

        if (steps + 1) % 5 == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    out2 = model(inputs)
                    batch_loss = criterion(out2, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(out2)
                    top_pred, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))

            print(f"Epoch {epoch + 1}/{epochs}"
                  f"Train loss: {running_loss / 5:.3f}",
                  f"Test loss: {test_loss / len(test_loader):.3f} "
                  f"Test accuracy: {accuracy / len(test_loader):.3f}")
            running_loss = 0
            model.train()

torch.save(model, os.path.join(PROJECT_DIR, "bc-model.pth"))
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

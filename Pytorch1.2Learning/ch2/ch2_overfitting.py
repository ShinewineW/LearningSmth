# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch2_overfitting
# Description:  
# Author:       Administrator
# Date:         2021/1/28
# -------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from  torchvision import datasets, transforms
import torch.utils.data
from visdom import Visdom
import numpy as np
import os
import matplotlib.pyplot as plt

batch_size = 64

# 如果要在一个完备的 train_loader上做切割，我们可以在导入之后，先不指定batchsize，然后在切割好之后
# 再使用 DataLoader加载为完备数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
     shuffle=True)
# 在得到 train_loader之后，我们再人为的划分成为 train he val 部分
print(len(train_loader))
train_db,val_db = torch.utils.data.random_split(train_loader,[50000,10000])
print(len(train_db))
print(len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size = batch_size, shuffle= True)
val_db = torch.utils.data.DataLoader(
    val_db,
    batch_size = batch_size,shuffle= True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)
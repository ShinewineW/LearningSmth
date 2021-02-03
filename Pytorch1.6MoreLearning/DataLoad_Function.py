# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         DataLoad_Function
# Description:  
# Author:       Administrator
# Date:         2021/2/3
# -------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import matplotlib.pyplot as plt

voc2012_images = datasets.VOCSegmentation('./data',
                                        year='2012', image_set='trainval', download=False, transform=transforms.ToTensor(),
                                        target_transform=transforms.ToTensor(), transforms=None)

# 我们在得到一整个数据集之后，同时在完成transform之后，可以使用Dataloader功能来将整个数据集转换为 一个一个batch
# voc2012 = DataLoader(voc2012_images, batch_size=32, shuffle=True,
#            drop_last=False)

# sample =iter(voc2012)
# sample = next(sample)
# print(sample[0][0].shape, sample[0][1].shape)


# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Vision_Function
# Description:  本文件稍微讲述了如何使用Vision模块，在这个模块中主要集成了数据集操作，已有模型，已经最最重要的简单数据增强操作
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

## 1. Datasets 模块  读入数据，将数据导入成为datasetLoader模式
# 此外还有很多其他的数据集可供选择，具体参看网址 https://pytorch.org/docs/1.6.0/torchvision/index.html
# 例如下载一个 voc2012 数据集 为之后的图像分割训练做准备

# 第一次下载，执行这个命令即可
voc2012_images = datasets.VOCSegmentation('./data',
                                        year='2012', image_set='trainval', download=False, transform=None,
                                        target_transform=None, transforms=None)

# 返回的数据是一个列表，然后每个列表中是一个元组，图片和标签一一对应
print(len(voc2012_images))
print(voc2012_images[0])

plt.clf()
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.imshow(voc2012_images[i][0])
plt.show()

## 2. models模块 提供很多现成的已经训练好的神经网络模型
# 主要分为classcification segmentation等
# 稍微看了一下，网络结果并不是很好
# 用处不大 而且都是ResNet作为 bottleneck


## 3. 最重要的模块 提供基本的数据强化方法TORCHVISION.TRANSFORMS
# 具体参看 https://pytorch.org/docs/1.6.0/torchvision/transforms.html
# 如果需要更加进阶一些
# https://pytorch.org/docs/1.6.0/torchvision/transforms.html#module-torchvision.transforms.functional 使用这个tf来进行自定义的数据变换

# 一般方法： 使用 compose 来对一系列操作进行整合
transform = transforms.Compose([

    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(p= 0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.ColorJitter(brightness= 0.2,contrast=0.1,saturation=0.1),
    transforms.RandomGrayscale(p= 0.01), #随机转换为灰度图
    transforms.RandomRotation(5),transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
                    ])
# 注意 这里的 Normalize过程一定要写在 ToTensor之后，否则会报错




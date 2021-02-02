# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch3_nnModule
# Description:  
# Author:       Administrator
# Date:         2021/2/1
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

## 简单实现一个 ResNet中的残差模块
class ResBlk(nn.Module):
    def __init__(self,ch_in,chout):
        '''
        对ResBlk中使用到的各个层进行初始化
        :param ch_in:
        :param chout:
        '''
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in,chout,kernel_size= 3,stride= 1,padding= 1)
        self.bn1 = nn.BatchNorm2d(chout)
        self.conv2 = nn.Conv2d(chout,chout,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(chout)

        if chout != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,chout,kernel_size=1,stride=1),
                nn.BatchNorm2d(chout)
            )


    def forwar(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out

        return out

## 从nn.Modul 中继承已有的，也可以使用容器 来对某些简单网络的堆叠，来实现一个简单的子网络
# 然后我们可以通过 parameters来非常方便的获得这个网络所有的可训练参数
# 可以通过 to(device) 非常方便的转移到  device中
# 可以使用sava和load 来进行保存


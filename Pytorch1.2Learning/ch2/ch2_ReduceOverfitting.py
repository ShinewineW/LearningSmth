# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch2_ReduceOverfitting
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

# 一般基础的  正则化方法 也就是优化器中的 weight_decay选项
#            带动量的优化器 方法 例如 Adam  SGMD 就不再赘述
# 这里主要讲解类似于keras 中的 平原优化率改变，或者一个scheme对学习率进行按epoch进行衰减

# 一般这里都要使用一个类  这个类掌控着学习率的改变


# torch.optim.lr_scheduler 这个类提供了若干根据当前epoch来进行学习率改变的方法
# torch.optim.lr_scheduler.ReduceLROnPlateau 而这个类允许你根据val数据集上的表现，来进行动态调整学习率

# 方法1
# torch.optim.lr_scheduler.ReduceLROnPlateau(
# optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
# threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# 使用方法几乎和keras中的方法一致

# 方法2
# 使用动态调节，人工指定在经过多少个epoch后 进行学习率的改变
# torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
# torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
# 代表方法 使用这两种方法来进行 人工指定学习率的改变


##
# Early Stop方法
# 在训练的时候 检测某个参数，在这个参数某个patience都没有优化，就可以提前结束网络的训练
# 同样的方法在keras中也存在 存放于 tf.keras.callbacks
# 具体参看https://www.kaggle.com/sergeykalutsky/pytorch-starter 这个网站将 保存最好checkpoint和earlystop 合并在了一个类中，这个类叫EarlyStop



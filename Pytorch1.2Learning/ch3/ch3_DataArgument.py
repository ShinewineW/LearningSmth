# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch3_DataArgument
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

## 常见的翻转和旋转
# 我们可以使用  RandomHorizontalFlip()  RandomRotation()  使用 transfroms 来进行变换
# 缩放  随机裁剪 随机噪声

# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Transfer_Learning
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
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt

ResNet18Model = torchvision.models.resnet18(pretrained=False)
# print(ResNet18Model)
ResNet18Model.requires_grad_(False)  ## 这表示这个网络被冻结 不能被训练
ResNet18Model.layer2 = nn.Conv2d(in_channels=  64,out_channels= 20,kernel_size= 3,stride=1, padding= 1)

print(ResNet18Model)
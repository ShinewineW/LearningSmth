# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch1_Visdom
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

# Visdom使用方法  首先要从Visdom类中实例化对象
viz = Visdom()
# 初始化  win参数表明是这个窗口，注意这个窗口是后续调用的唯一引用参数，也就是后续如果要在同一个窗口中绘制，指定相同win即可
#        opts参数在初始化中进行，给定这个图的名字，按照字典方式进行参数传递
viz.line([0.],[0.],win = 'train_loss',opts= dict(title = 'train loss'))
# 参数传入，第一个为y轴数据，第二个为x轴数据，win指定在哪个窗口，update方式指定到底是刷新还是继续在后面添加
# viz.line([loss.item()], [glob_step], win='train_loss', update='append')

##
# 如果要绘制两条曲线，例如一般我们会把loss 和 validation_loss放在一起
# 将accu 和 val_accu放在一起 方便我们确定到底何时开始过拟合
# 初始化
viz.line([0.,0.],[0.],win = 'test',opts= dict(title = 'train loss',legend = ['loss','acc']))
# 继续添加
# viz.line([test_loss,test_accuracy],[global_step],win = 'test',update = 'append')


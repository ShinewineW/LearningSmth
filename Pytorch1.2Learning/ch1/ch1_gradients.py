# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch1_gradients
# Description:  
# Author:       Administrator
# Date:         2021/1/27
# -------------------------------------------------------------------------------
import torch
from torch import optim
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt

##
def himme(x):
    return( x[0] ** 2 + x[1] - 11)**2 + (x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
X,Y = np.meshgrid(x,y)
Z = himme([X,Y])

fig = plt.figure('himme')
ax = fig.gca(projection = '3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.show()

# 梯度下降
x = torch.randn(size = [2],requires_grad= True)
optimizer = optim.AdamW([x],lr = 1e-3) #设定优化器

for step in range(15000):

    pred = himme(x) #前向传播

    optimizer.zero_grad() #清空优化器
    pred.backward() #反向传播
    optimizer.step() #优化一步

    if step % 2000 == 0:
        print("step at {}: the pred value:{}".format(step,pred.item()))


print(x.tolist())




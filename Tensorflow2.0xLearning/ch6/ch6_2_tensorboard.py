# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:28:24 2020
@tensorboard介绍 但是visdom会更好用 因此这里就不再详细说明tensorboard
@author: Administrator
"""

#%%tensorboard工作原理 
#通过对cpu写到磁盘上的logs来进行监听，

#关键点 在正确的位置打开tensorboard
#1. 打开监听器然后在控制台输入 tensorboard --logdir logs
#2. 在代码中通过summary来写入日志
#3. 喂入数据



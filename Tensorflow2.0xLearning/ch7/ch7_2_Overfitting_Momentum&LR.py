# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:37:08 2021
@Discription: 在优化器中选取learning_rate的动量以及更新策略
使用keras相关库进行learning rate decay的学习率衰减管理
@author: Administrator
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
#带动量的优化方式，这种方式就是会考虑之前的数值带来的影响，从而约束你的优化反向
#因子β用来衡量之前的优化值占据本次更新的多少，一般设定为0.9

#Ex
optimizer_sgd = optimizers.SGD(learning_rate = 0.02,momentum = 0.9)
optimizer_rms = optimizers.RMSprop(learning_rate = 0.02,momentum = 0.9)

#%%
#学习率衰减   根据训练epoch 来慢慢放缓学习率 避免出现步长过大而导致出现问题
initial_learning_rate = 0.1

#实例化一个learning_rate的计划管理，管理中会设置多少步大概下调多少
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#在complile过程中，我们不指定具体的learning_rate，而是将上述的实例化管理引入
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])










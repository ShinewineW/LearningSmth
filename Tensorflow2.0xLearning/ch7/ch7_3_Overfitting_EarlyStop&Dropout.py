# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:01:56 2021

@author: Administrator
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%提早结束 Early Stop
#1.在验证集上选择参数
#2.检测模型在验证集上的表现
#3.在验证集表现最好的时候停止

#在callback中进行相应的定义，指定检测的loss，容忍值，以及你的loss是最小，最大，还是默认自动
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# This callback will stop the training when there is no improvement in
# the validation loss for three consecutive epochs.
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
# model.compile(tf.keras.optimizers.SGD(), loss='mse')
# history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
#                     epochs=10, batch_size=1, callbacks=[callback],
#                     verbose=0)


#%%dropout 随机放弃神经元的若干节点
#在keras中可以使用layers自带的Dropout函数 直接从父类中继承过来
layers.dropout(0.5)






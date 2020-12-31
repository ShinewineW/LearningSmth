# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:41:43 2020

@author: Administrator
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%正则化在吴恩达的相关课程中阐述的很明确
#本质就是在loss中给原有loss增加一个惩罚项，这个惩罚项会导致 W越复杂，loss越高
#因此优化器就会将w的简单 也作为优化目标之一进行优化

#经过求导之后，你会发现这个正则项本质就是在每一层的dw上增加一个系数，也就是lamda系数
#一般我们可以称之为weight decay 也就是参数衰减

#在keras中  layers的类定义中就有 你可以在实例化的过程中 从父类继承这个kernel_regularizer过来
# EX:
layers_Dense = keras.layers.Dense(16,kernel_regularizer = keras.regularizers.l2(0.001),activation = tf.nn.relu,input_shape=())

#通过如上方式来进行一个L2范数正则化的添加

#或者在自己实现 loss的过程中 通过gradienttape手动实现
# 具体代码可以看  课时87 Regularization的最后10min

# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch10_Gridient_Vanish&Explosion
# Description:  
# Author:       Administrator
# Date:         2021/1/13
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用tf中自带的 clip函数来处理loss
loss = 20
tf.clip_by_norm(loss,15)
print(loss)

# 解决梯度消失问题

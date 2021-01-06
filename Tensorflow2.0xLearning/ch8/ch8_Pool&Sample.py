# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch8_Pool&Sample
# Description:  本文件说明了使用keras来进行池化和上采样的方法
# Author:       Administrator
# Date:         2021/1/6
# -------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##最大池化
x = np.zeros(shape=(1,28,28,3))
pool = layers.MaxPool2D(2,strides=2)
out = pool(x)
print(out.shape)






##上采样
upsample = layers.UpSampling2D(size=3)
out = upsample(x)
print(out.shape)


##


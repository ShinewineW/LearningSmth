# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:27:41 2021

@author: Administrator
"""

##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


##如果直接调用kears中自带的conv2d函数，大多数情况只需要设置如下参数
#或者指定一下权重和偏置的初始化函数
x = tf.random.uniform(shape=(1,32,32,3))
layer = layers.Conv2D(5,kernel_size=(4,4),strides=1,padding='valid')
out = layer(x)
print(out.shape)
print(layer.kernel)







##如果使用tf中的conv2d函数， 你需要手动指定所有的参数细节和使用的方向维度
# filters: A `Tensor`. Must have the same type as `input`.
#       A 4-D tensor of shape
#       `[filter_height, filter_width, in_channels, out_channels]`
#手动指定卷积核的  长宽，输入通道，输出通道
#     strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
#       stride of the sliding window for each dimension of `input`. If a single
#       value is given it is replicated in the `H` and `W` dimension. By default
#       the `N` and `C` dimensions are set to 1. The dimension order is determined
#       by the value of `data_format`, see below for details.
#手动指定步长，每个步长和filters是一一对应的关系，一般我们在输入通道和batch是默认为 1的
#     padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
#       padding algorithm to use, or a list indicating the explicit paddings at
#       the start and end of each dimension. When explicit padding is used and
#       data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
#       pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
#       and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
#       [pad_top, pad_bottom], [pad_left, pad_right]]`.
x = tf.random.uniform(shape=(1,32,32,3))
filters = tf.random.normal([5,5,3,4])
out = tf.nn.conv2d(x,filters,(1,2,2,1),padding='VALID')
print(out.shape)


##关于卷积层的反向传播
#在吴恩达的C4_W1_HomeWork_Part1.py文件中已经进行了相关的计算
#具体的操作在于 你站在后续来的梯度上，从后往前看
#将对应的坐标转换为原图的变动范围，然后直接用乘法将梯度加分配到原来的矩阵中去




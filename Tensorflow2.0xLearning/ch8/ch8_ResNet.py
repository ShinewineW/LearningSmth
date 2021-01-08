# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch8_ResNet
# Description:  本文件构建了一个最简单的ResNet单元
# Author:       Administrator
# Date:         2021/1/7
# -------------------------------------------------------------------------------

# 如下代码实现一个经典的残差模块的构建

# 一般构建一个用于实现网络中的积木，我们会使用class类来进行搭建

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


##
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # 声明参数，先声明所有将会在这个层中使用的参数，具体的连接关系应该在call中调用
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')

        # 所有会使用的bn层
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        # 所有会使用的激活层
        self.activ1 = layers.Activation('relu')

    def call(self, inputs, training = None):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.activ1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = layers.add([bn2,inputs])
        out = self.activ1(add)

        return out


inputs = np.zeros(shape = (4,32,32,3),dtype=np.float32)

#实例化这个类
resdiual = BasicBlock(filter_num= 3)

#调用这个类
output = resdiual(inputs)

print(output.shape)
##


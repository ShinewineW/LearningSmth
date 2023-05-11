# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch8_ResNet_Practice
# Description:  本文件使用子类构建方法，从零构建了一个最简单的ResNet网络
# Author:       Administrator
# Date:         2021/1/7
# -------------------------------------------------------------------------------
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
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')

        # 所有会使用的bn层
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        # 所有会使用的激活层
        self.active1 = layers.Activation('relu')

        # 但是由于stride如果不为1，那么会出现size不一致的情况，因此此时层的恒等映射中还需要进行一个下采样
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride, padding='same'))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        x_identity = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.active1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = layers.add([bn2, x_identity])
        out = self.active1(add)

        return out


class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes=100):  # [2,2,2,2]
        super(ResNet, self).__init__()
        self.preprocess = Sequential(
            [
                layers.Conv2D(64, (3, 3), strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
            ]
        )
        self.layer1 = self.build_resblock(64, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.flat1 = layers.GlobalAveragePooling2D()
        self.Dense1 = layers.Dense(num_classes)

    def call(self, inputs, training=None, **kwargs):
        X = self.preprocess(inputs)
        print(X.shape)
        layer1 = self.layer1(X)
        print(layer1.shape)
        layer2 = self.layer2(layer1)
        # flat1 = self.flat1(layer2)
        out = layer2
        # out = self.Dense1(flat1)

        return out

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()

        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


# 实例化
ResModel = ResNet([2, 2], 100)
# 运行
# input_x = np.zeros(shape=(10, 64, 64, 128), dtype=np.float32)
# out = ResModel(input_x)

ResModel.build(input_shape=(None,32,32,3))

##

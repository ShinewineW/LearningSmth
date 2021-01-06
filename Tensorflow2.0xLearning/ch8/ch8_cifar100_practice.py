# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch8_cifar100_practice
# Description:  
# Author:       Administrator
# Date:         2021/1/6
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics,Input,Model
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(233)

##
def cifar100_Model():
    x = Input(shape = (32,32,3))

    # unit 1
    X =layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu')(x)
    X =layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')(X)

    # unit2
    X =layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')(X)

    # unit3
    X =layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')(X)

    # unit4
    X =layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')(X)

    # unit5
    X =layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu')(X)
    X =layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')(X)

    #Flatten
    X = tf.reshape(X,shape=(-1,512))
    # FC unit

    X  = layers.Dense(256, activation='relu')(X)
    X  = layers.Dense(128, activation='relu')(X)
    Y_logits  = layers.Dense(100)(X)

    #实例化模型
    VGG_Cifar100_Model = Model(inputs = x,outputs = Y_logits)

    return VGG_Cifar100_Model

#调用函数构建模型
Cifar100Model = cifar100_Model()


x = tf.random.normal([4,32,32,3])

out = Cifar100Model(x)

print(out.shape)

#电脑机能限制就不进行后续计算了 后续直接套用keras自带的api可以很容易实现



















##


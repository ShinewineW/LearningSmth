# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 15:07:33 2020
@Discription:本文件给定了一个自定义层 自定义模型的实例
具体该如何自定义，以及哪些参数可以自定义需要参考tensorflow.keras中的文档描述

@author: Administrator
"""

#%%
#通過 Keras.Sequential类，可以很方便的实现一个网络从顶到底部的流水线结构
#但是上述流水线结构的积木 也就是每一层的特性
#必须完全继承自 keras.layers.Layer这个类，如果不是是无法使用keras来自定义结构的
#自己的模型也必须继承子 keras.Modle这个类

import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras




class MyDense(layers.Layer):

	def __init__(self, inp_dim, outp_dim):
		super(MyDense, self).__init__()

		self.kernel = self.add_weight('w', [inp_dim, outp_dim])
		self.bias = self.add_weight('b', [outp_dim])

	def call(self, inputs, training=None):

		out = inputs @ self.kernel + self.bias

		return out 

class MyModel(keras.Model):

	def __init__(self):
		super(MyModel, self).__init__()

		self.fc1 = MyDense(28*28, 256)
		self.fc2 = MyDense(256, 128)
		self.fc3 = MyDense(128, 64)
		self.fc4 = MyDense(64, 32)
		self.fc5 = MyDense(32, 10)

	def call(self, inputs, training=None):

		x = self.fc1(inputs)
		x = tf.nn.relu(x)
		x = self.fc2(x)
		x = tf.nn.relu(x)
		x = self.fc3(x)
		x = tf.nn.relu(x)
		x = self.fc4(x)
		x = tf.nn.relu(x)
		x = self.fc5(x) 

		return x



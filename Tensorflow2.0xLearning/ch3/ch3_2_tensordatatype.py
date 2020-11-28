# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:19:22 2020
@Discription: 进行tesnor数据类型的学习
@和DrWu的deeplearning公开课中的维度使用是相反的
@author: Administrator
"""

import tensorflow as tf
import numpy as np


#%% Tensor张量的构建
#1.从 numpy list中得到
a = np.ones(36).reshape(6,6)
b = tf.convert_to_tensor(a,dtype = tf.int32)
print(b.dtype)
c = tf.convert_to_tensor([1,2],dtype = tf.float32)
print(c)

#2.人为填充某种 确定的值
#zeros 和 ones方法
#注意 这里需要填充的是维度大小！
a = tf.zeros([2,3])
b = tf.zeros(6) #这里就会给你新建一个一行6列的张量
print(a)
print (b)
#zero_like功能  这个功能和numpy中类似 就是根据传入的张量大小，新建一个等大的全为0的张量
#fill 第一个参数和zeros一样，第二个参数就是你指定的值
c = tf.fill([2,2],3)
print(c)

#3.随机初始化分布
# 正太分布  分别指定张量的维度，均值，方差
a = tf.random.normal([2,2],mean = 1,stddev = 1)
print (a)
#均匀分布
b = tf.random.uniform([2,2],minval = 0, maxval = 1)
print(b)

#4.用constant来创建 与方法1 几乎一致
a = tf.constant([1,2,3,4])
print (a)












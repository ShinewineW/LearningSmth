# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:29:40 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np


#例如对于一个4张  28*28*3的图片
a = tf.random.normal([4,28,28,3])
#如果我们要把这个变成一个 4*拍扁像素的图片 应该
b = tf.reshape(a,[4,-1])
print(b.shape)
#如果只是把图片size拍扁而保留通道
b= tf.reshape(a,[4,-1,3])
print(b.shape)

#如果我们要对数据的结构发生改变，例如从根本上改变维度
#使用tf.transpose
a = tf.random.normal([1,2,3,4])
print(a.shape)
#注 可以通过指定序列的方式来确保transpose的执行，但是一定要注意自己维护好列表的正确性
b = tf.transpose(a,perm=[0,1,3,2])
print(b.shape)

#如果我们需要对数据的维度进行改变，即增加维度或减少维度
#例如对于一个 4个班级 35个学生 每个学生8门功课的一个tensor
b = tf.zeros([4,35,8])
#现在我们有两个学校，都有这样的数据，现在我们希望把两个学校的数据合并
#形成 2个学校 4个班级 35个学生 8门功课的 tensor
#使用tf.expand_dims函数 指定一个轴 来实现维度的增加
c = tf.expand_dims(b,axis = 0)
print ( c.shape)

#挤压如果某个维度只有1个数据，对于shape = 1dim的 可以直接去掉
#squeeze会自动去除所有维度为1的维度大小
a = tf.constant([[1,2,3]])
print ( a.shape)
b = tf.squeeze(a)
print ( b.shape)

#但是如果这个维度的数不是1 那么使用squeeze就会报错
a = tf.squeeze(a,axis = 0)
print ( a.shape)





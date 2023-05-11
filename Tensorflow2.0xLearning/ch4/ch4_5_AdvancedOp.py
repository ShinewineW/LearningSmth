# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:38:10 2020
@Discription:告诫操作
tf.where:具体用法和np.where一模一样，一般用于特定条件下数据替换
tf.scatter_nd:据indices中的下标，将updates中的值一一根据坐标赋值到shape中去
@author: Administrator
"""

import tensorflow as tf 
import numpy as np

#%%1. where操作
#与np.where操作几乎一样
#例如对于一个矩阵 返回能被3整除的位置
a = tf.range(9)
a = tf.reshape(a,[3,3])
print(a)

b = tf.where(a %3 == 0, 1,-1)
print(b)

#%% scatter_nd操作
#tf.scatter_nd(indices,updates,shape)
#根据indices中的下标，将updates中的值一一根据坐标赋值到shape中去
#也就是修改指定indices坐标下，updates中的值到 shape中去
#意味着 indices的大小必须要和updates一模一样 否则会报错
indices = tf.constant([[4], [3], [1], [7]])
print(indices.shape)
updates = tf.constant([9, 10, 11, 12])

shape = tf.constant([8])
print(shape)
c = tf.scatter_nd(indices,updates,shape)
print(c)


#%%Points操作
#例如我们要获得从x到y间隔固定间隔的点坐标集群
#使用tf.mershgrid





















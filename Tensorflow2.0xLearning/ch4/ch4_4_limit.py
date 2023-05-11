# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:13:45 2020
@Discription: 本代码介绍tf中的限制幅度操作，一般为了避免出现梯度爆炸，我们会有两种
不同的限幅操作
tf.clip_by_value:对值的大小，按照max min来进行限幅
tf.clip_by_norm:对输入的张量根据范数来进行限幅
@author: Administrator
"""

import tensorflow as tf
import numpy as np


a  = tf.range(9)

#对a中的所有元素进行限幅，凡是小于指定值的，都为指定值，大于的同理
b = tf.clip_by_value(a,2,8)
print(b)

#对a中的所有数进行一个放缩操作，这种放缩是针对输入数据的二范数进行放缩的
a = tf.random.normal([2,2],mean = 10)

aa = tf.clip_by_norm(a,15)

aaa = a/tf.norm(a,2)*15

print(aa,aaa)
#可以发现aa与aaa是等效的


#%% 为了避免出现梯度爆炸和梯度消失的情况
#通过限幅的方式，保证所有维度下的反向不变，但整体的大小被缩放了

















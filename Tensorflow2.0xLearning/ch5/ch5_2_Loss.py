# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:27:54 2020
@Discription: 本课见了tf中loss的计算方法，一般我们都是用交叉熵来计算loss，可以直接使用tf.loss中的相关函数
注意要弄明白from logits的意义。
@author: Administrator
"""

import tensorflow as tf
import numpy as np

#Tensorflow中  
#%% MSEloss

y = tf.constant([1,2,3,0,2])
y = tf.one_hot(y,depth = 4)

y = tf.cast(y,dtype = tf.float32)

out = tf.random.normal([5,4])

loss1 = tf.reduce_mean(tf.square(y-out))
loss2 = tf.square(tf.norm(y-out)) / (5*4)
loss3 = tf.reduce_mean(tf.losses.MSE(y,out))
loss_temp = tf.losses.MSE(y,out)
print(loss_temp)
print(loss1,loss2,loss3)

#%% 交叉熵损失
#衡量不稳定度，越稳定的系统消息越没有价值，越不稳定的系统才有价值
#熵越小，那么说明越不稳定，惊喜度越高
a = tf.fill([4],0.25)
temp = a*tf.math.log(a) / tf.math.log(2.)
temp = -tf.reduce_sum(temp)

print(temp)
#注意tf2.0中的交叉熵损失，from logits 选项要指定为开
#同时我们直接将模型计算好的逻辑输出直接输入即可，由模型来自动实现softmax 而不是手动实现
#否则可能会出现数值不稳定的情况

























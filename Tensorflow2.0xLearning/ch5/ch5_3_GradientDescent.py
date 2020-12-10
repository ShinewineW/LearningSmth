# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:04:12 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np

#%%
#我们只需要把整个前向计算过程包在With Tf.GradientTape() as tape:
#中即可，然后在完成前向计算和损失计算之后，
#调用  [w_grad] = tape.gradient(loss,[w])即可
#其中列表的参数位置是一一对应的

w = tf.constant(1.)
x = tf.constant(2.)
y = x * w

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x*w
grad2 = tape.gradient(y2,[w])
print(grad2)

#注意所有的相关计算过程，必须全部，统统都放置在这个with中

#但是如上过程在完成使用之后，就会将调用到的参数释放掉
#如果为了保证相关调用在之后还能继续使用，我们需要在with中指定命令
with tf.GradientTape(persistent = True) as tape:
    tape.watch([w])
    y2 = x*w
grad2 = tape.gradient(y2,[w])
print(grad2)


#%% sigmoid函数求导示例
a = tf.linspace(-10.,10.,10)
a = tf.Variable(a)
print(a)

with tf.GradientTape() as tape:
    y = tf.sigmoid(a)
    
grads = tape.gradient(y,[a])
print(grads)

#%% sigmoid和relu函数合并求导示例
a = tf.linspace(-10.,10.,10)
a = tf.Variable(a)
print(a)

with tf.GradientTape() as tape:
    y = tf.where(a<0,tf.sigmoid(a),tf.nn.relu(a))

print(y)
    
grads = tape.gradient(y,[a])
print(grads)

#%% loss的梯度求导示例
x = tf.random.normal([2,4])
w = tf.random.normal([4,3])

b = tf.zeros([3])
y = tf.constant([2,0])
w = tf.Variable(w)
b = tf.Variable(b)


with tf.GradientTape() as tape:
    prob = tf.nn.softmax( (x @ w)+b,axis = 1)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth = 3) , prob))

grads = tape.gradient(loss,[w,b])
print(grads)















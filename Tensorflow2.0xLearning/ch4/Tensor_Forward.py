# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:59:55 2020
@Discription: 通过tensorflow2.0实现了一个对于Minist数据集的前向传播训练过程，
#其中还使用到了 He.etl的参数初始化方法
#softmax 函数用于对多分类问题的loss函数
#没有使用正则化
@author: Administrator
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#1.加载数据集
(x,y) , _ = datasets.mnist.load_data()

# print(x.shape,y.shape)
#说明图片是一个  28*28 60k张图片 标签为60k个标签数字
#提取几张图片看看
# plt.imshow(x[0])
# plt.title(y[0])

#2.将数据集转换为张量
x = tf.convert_to_tensor(x,dtype = tf.float32)
y = tf.convert_to_tensor(y,dtype = tf.int32)

# print(x.shape,y.shape)
# print(tf.reduce_min(x),tf.reduce_max(x)) 
# #tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(255.0, shape=(), dtype=float32)
# #说明值从0到255
# print(tf.reduce_min(y),tf.reduce_max(y))
#tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32)
#说明值从0到1

#3.归一化输入数据x 将输入数据x划归到0到1的区间
x = x/ 255.

#4.创建数据集 mini_batch方法 具体原始代码参考DrWu Deeplearning C2_W2_HomeWork

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)

#test
train_iter = iter(train_db)
sample = next(train_iter)
#print(sample)
#可以发现sample是个元组，元组由两个tensor组成
x_minibatch,y_minibatch = sample
print(x_minibatch.shape,y_minibatch.shape)
#(128, 28, 28) (128,)
#这里就发现每次的batch就为128

#5.参数初始化
#构建一个网络 从 m,28*28 -> 256 ->128->10
#注意由于这里使用了和DrWu课程中完全不一样的参数集合方法，因此是正好相反的
# W1 = tf.random.truncated_normal([28*28,256],stddev = 0.1)
W1 = tf.random.truncated_normal([28*28,256]) * (np.sqrt(2./28*28))
b1 = tf.zeros([256])
# W2 = tf.random.truncated_normal([256,128],stddev = 0.1)
W2 = tf.random.truncated_normal([256,128]) * (np.sqrt(2./256))
b2 = tf.zeros([128])
# W3 = tf.random.truncated_normal([128,10],stddev = 0.1)
W3 = tf.random.truncated_normal([128,10]) * (np.sqrt(2./128))
b3 = tf.zeros([10])
#这里只是tensor类型还不够 我们需要向tensorflow声明 这是一个可以求导的Variable
W1 = tf.Variable(W1)
b1 = tf.Variable(b1)
W2 = tf.Variable(W2)
b2 = tf.Variable(b2)
W3 = tf.Variable(W3)
b3 = tf.Variable(b3)


learning_rate = 1e-3
num_iretation = 10
costs = []
step = 0

for i in range(0,num_iretation):
    #最外围是对epoch的定义
    #对minibatch的每一个小batch进行遍历
    for (x_minibatch,y_minibatch) in train_db:
        #处理一下x_minibatch 有上可知 这里的输入是 (128,28,28)
        x_minibatch = tf.reshape(x_minibatch,[-1,28*28])
        step += 1
        
        with tf.GradientTape() as tape:
            
    #6.构建forward运算
            Z1 = x_minibatch @ W1 +b1
            #激活函数 
            A1 = tf.nn.relu(Z1)
            Z2 = A1 @ W2 + b2
            A2 = tf.nn.relu(Z2)
            Z3 = A2 @ W3 + b3
            A3 = tf.nn.softmax(Z3)
           
            
        #7.计算误差
            #先把y_minibatch变为onehot编码形式
            y_minibatch = tf.one_hot(y_minibatch,depth = 10)
            #loss = tf.square(y_minibatch - A3) #计算单个的loss函数
            loss  = -tf.reduce_sum( y_minibatch * tf.math.log(A3),axis = 1,keepdims = True)
            #print(loss.shape)
            J_cost = tf.reduce_mean(loss) #计算所有loss的平均值
            if step % 100 == 0 :
                costs.append(J_cost)
    
    #8.反向传播
        #为了能让tensorflow自动求导 我们需要把forward的过程包裹在 tf.grandint中
        grads = tape.gradient(J_cost,[W1,b1,W2,b2,W3,b3])
        #完成反向传播后 根据grads和learning_rate来进行更新
    #9.更新参数
        #强制要求所有参数在原有的形态进行更新，因此更新后会保证数据依然是Variable类型
        W1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        W2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
        W3.assign_sub(learning_rate * grads[4])
        b3.assign_sub(learning_rate * grads[5])

        
    print("The " + str(i) +" epoch: Loss is {} ".format(float(J_cost)))
    
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (x1,000)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()
    
#以上通过tensorflow完成了一次基础的深度神经网络的训练过程
        

        
        
    
    
    





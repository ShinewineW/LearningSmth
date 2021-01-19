# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:39:20 2020
Written by Netfather
This document is guided by Dr.Wu DeepLearning Course C1_W2_Homework_Part1
@Discription: 本文件实现了基本的深度学习激活函数，以及L1 L2距离的计算方法，本文件构建的函数用于Part2作业
@author: Netfather
@Last Modified data: 2021年1月19日
"""

#%% [1]
### START CODE HERE ### (≈ 1 line of code)
test = "Hello World"
### END CODE HERE ###

print ("test: " + test)

#%%[2]
#1 构建一个返回实数的sigmoid函数

import math

def basic_sigmoid(x):
    
    s = 1/(1+math.exp(-x))
    
    return s

print(basic_sigmoid(3))

#但是自己构建的simoid有一个显著的问题，那就是不支持向量操作
#这在深度学习中是无法接受的

#%% [3]
#2. 使用numpy实现sigmoid函数

import numpy as np

def sigmoid(x):
    
    s = 1/(1+np.exp(-x))
    
    return s

print(sigmoid(np.array([1,2,3])))

#%% [4]
#3. 使用numpy实现sigmoid函数的求导函数

def sigmoid_derivative(x):
    
    s = sigmoid(x)
    ds = s * (1-s)
    
    return ds

print(sigmoid_derivative(np.array([1,2,3])))

#%% [5]
#4. 重塑输入图像，将其拍扁作为列向量方便处理

def image2vector(image):
    
    height , length ,channel= image.shape()
    image_vector = image.reshape(height*length*channel,1)
    # 注意这里以及之后的一段课程，吴恩达在用numpy构建数据集时，使用的方式为(features,m),是一种特征量在前，batchsize在后的方式
    # 一般在tensorflow和keras中，使用的方式为(batchsize,features)
    
    return image_vector

#%% [6]
#5. 归一化输入数据，使用归一化之后的数据往往更加容易收敛

def normalizeRows(x):
    
    x_norm = np.linalg.norm(x,axis = 1,keepkims = True)
    x = x / x_norm
    
    return x

#%% [7]
#6. 对于单个逻辑回归，使用sigmoid函数就够了，
#但是如果对于多个物体的分类，使用Sigmoid函数就不恰当，这时候需要使用
#softmax函数来对所有待分类物体进行分类概率预测，而不是简单的逻辑回归


def softmax(x):
    
    e_x  = np.exp(x)
   # e_x_sum = e_x.sum(axis = -1).reshape(-1,1)
   #或者使用Keepdimions参数来保证参数
    e_x_sum = e_x.sum(axis = -1,keepdims = True)
    s = e_x / e_x_sum
    
    return s

print ("Test SoftMax Function")
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
    
#%% [8]
#7. 实现L1损失函数

#Graded Function L1

def L1(yhat,y):
    #yhat 为正向传播之后的值
    #y 为实际的值
    
    loss1 = np.sum(np.abs(yhat - y))
    return loss1

#8. 实现L2损失函数

#Granded Loss Function L2

def L2(yhat,y):
    
    y_temp = np.abs(yhat-y)
    loss2 = np.dot(y_temp,y_temp.T)
    
    return loss2

# Until this The Dr.Wu DeepLearning Course C1_W2_HomeWork_Part1 is finished
# See the document C1_W2_Homework_Part2.py    


    
    
    
    
    
    
    
    
    
    
    
    
    
    

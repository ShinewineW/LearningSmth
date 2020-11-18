# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:39:20 2020

@author: Administrator
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
    
    return image_vector

#%% [6]
#5. 归一化输入数据，使用归一化之后的数据往往更加容易收敛

def normalizeRows(x):
    
    x_norm = np.linalg.norm(x,axis = 1,keepkims = True)
    x = x / x_norm
    
    return x

#%% [7]
#6. 对于单个逻辑回归，使用sigmoid函数就够了，
#但是如果
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

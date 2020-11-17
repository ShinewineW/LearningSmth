# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:07:35 2020
This code is on WuDLcourse 
Code is conducted by the video on www.bilibili.com/keyword?=深度学习_Deep_Learning_Pytorch特别制作版
@author: Administrator
"""

#%%
#1.Code on C1_W2_11
#开始讲解向量化操作，如下代码例子描述了使用向量化操作的时间优势
#在深度学习中，不论怎样都要尽量避免使用for循环代码

import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

#如下是向量化操作代码
tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("The vectorized version: " + str((toc-tic)*1000) + "ms")

#如下是for循环操作代码
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("The For loop version: " + str((toc-tic)*1000) + "ms")

#%%
#2.Code on C1_W2_12
#如下代码比较对一列向量取指数的操作

import numpy as np
import time
import math

a = np.random.rand(1000000).reshape(1000000,1)

#如下是for循环代码
tic = time.time()
for i in range(1000000):
    if (i % 100000 == 0): 
        print(a[i,0])
    temp = math.exp(a[i,0])
toc = time.time()
print("The For loop version: " + str((toc-tic)*1000) + "ms")

#如下是向量化代码
tic = time.time()
a = np.exp(a)
toc = time.time()

print("The vectorized version: " + str((toc-tic)*1000) + "ms")

#%%
#3.Code on C1_W2_15
#这部分展示了python中numpy下的广播规则
import numpy as np

A = np.array([[56.0,0.0,4.4,68.0],
             [1.2,104.0,52.0,8.0],
             [1.8,135.0,99.0,0.9]])

print(A.shape)

row_Sum = A.sum(axis =0)
print (row_Sum, row_Sum.shape) #一旦指定维度，这个维度就会消失，因此下面需要reshape

percentage = A.T / row_Sum.reshape(4,1) #numpy中运算列优先对齐规则
print(percentage)
percentage2 = A / row_Sum.reshape(1,4) #numpy中运算列优先对齐规则
print(percentage2)

B = np.arange(1,7).reshape(2,3)
print(B)

#列数量一致相加
C = np.array([100,200,300]).reshape(1,3)
print(B+C)
#行数量一致相加
D = np.array([100,200]).reshape(2,1)
print(B+D)

#%%
#4.Code on C1_W2_16
#numpy向量操作说明
import numpy as np

a = np.random.randn(5)
print(a,a.shape)
#关于维度消失的那一列或者那一行，需要非常小心的维护！

print(np.dot(a,(a.T)))

#因此为了避免歧义，我们需要明确的指定reshape之后的大小到底是多少
b = np.random.randn(5).reshape(5,1)
print(np.dot(b,b.T))



















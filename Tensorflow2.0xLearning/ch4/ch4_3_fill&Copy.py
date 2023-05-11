# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:45:22 2020
@Discription：这里阐述了一些关于张量的填充与复制操作
tf.pad: 对于一个输入张量，在张量的维度周围进行指定数据的填充
        第二个参数一定是一个 n，2 的矩阵，n为输入矩阵的秩，2表示在指定维度的前后
tf.tile:对于一个输入张量，在内存中复制若干次，达到目标
        第二个参数是一个维度和输入矩阵一样的列表，列表内容表明在该指定维度复制几次，
        其中1次表明不变。
@author: Administrator
"""

import tensorflow as tf
import numpy as np

a = tf.range(9)
a = tf.reshape(a,[3,3])

print(a)

#1. tf.pad

b = tf.pad(a,[[2,1],[0,0]])
print(b)
#援引官方文档的说法，指定参数是一个 [n,2]的矩阵，其中n表示输入有多少维
#2表示是在这个维度的前面，还是后面，具体的参数大小表示加多少个padding的数值

b = tf.pad(a,[[2,1],[0,1]],constant_values = 2)
#在所有行的 前面加 2行2 
#            后面加1行2
#在所有列的 后面加 1行2
print (b)


#2.数据复制操作
#与broadcast_to的不同在于  广播并不会在内存层面给你扩容，而是使用某种方法压缩了空间
#tf.tile可以保证是完全复制

c = tf.tile(a,[1,2])
#第二个参数表明了赋值次数，对应于你的维度，因此实际上最终维度的大小就是
#input.shape * 2nd arg  
#必须保证这个维度和你的输入维度是一致的
print(c)








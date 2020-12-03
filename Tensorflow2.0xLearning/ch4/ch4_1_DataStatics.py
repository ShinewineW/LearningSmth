# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:15:08 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np

#%%范数

a = tf.ones([2,2])
tf.norm(a)#计算a矩阵的二范数

tf.norm(a,ord = 1) #计算a矩阵的一范数


#%%排序
#对张量内容进行排序  tf.sort
#如果想得到排序后的位置  tf.argsort

a = tf.random.shuffle(tf.range(8))
a = tf.reshape(a,[2,4])
print(a)
# a = tf.gather(a,indices = [1,0],axis = 0)
# print(a)

#默认是按照axis = -1来进行排序，注意到指定轴意味着这个轴会消失
#在这里默认-1 意味着列消失了，那么没有了列，这个张量就只剩下各个行，然后对行进行倒序
a = tf.sort(a,direction = 'DESCENDING')
print(a)


#%%某些时候我们不需要对所有的内容进行排序
#一般在分类过程中，我们会考察top5的错误率
#使用top_k 来返回前k个值
#该函数自动就是为最后一个维度排序的
a = np.random.randn(9) * 10
a = a.reshape(3,3)
a = tf.convert_to_tensor(a,dtype = tf.int32)
print(a)



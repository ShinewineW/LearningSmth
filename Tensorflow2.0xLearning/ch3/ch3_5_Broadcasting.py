# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:16:52 2020
@Discription: 按照numpy中一致的广播规则进行广播
注意到tensorflow和numpy一样，都是列优先原则，
因此首先按照最后一列对齐，对齐之后往前看，维度不足的地方补1，只要有维度的地方是对齐的，那么就可以进行传播
@author: Administrator
"""

import tensorflow as tf
import numpy as np

#%%1.如果a b维度不一致，那么就在需要增加维度的地方，就全部扩充
a = np.array([0,1,2]).reshape(1,-1)
b = np.array([0,10,20,30]).reshape(-1,1)

tf_a = tf.convert_to_tensor(a)
tf_b= tf.convert_to_tensor(b)

tf_c = tf_a+ tf_b
print(tf_a.shape  , tf_b.shape)
print(tf_c)


#具体原理和numpy中的广播操作一致，只要有一致的维度，就会自动在加减中进行广播
#显示的优化手段，可以节省大量内存空间
x  = tf.random.normal([4,32,32,3])
#print(x + tf.random.normal([1,4,1,1]).shape)
print((x + tf.random.normal([4,1,1,1])).shape)

#%%统合所有的张量的基本数学运算

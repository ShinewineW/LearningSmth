# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:15:08 2020
@Discription:详细说明了常用于统计方面的操作
tf.norm:等一系列用于统计的操作，例如范数，最大值最小值，最大最小位置等等
tf.sort:用于实现排序操作，其中默认axis为-1，也就是对消除了列之后的维度，进行一个排序
        在高于3的维度层面，请谨慎使用！
tf.math.top_k:用于统计前k个的indiecs，常与属性indices配合使用。返回前k个的索引，默认
        轴为最后一个维度！

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

a = tf.random.shuffle(tf.range(16))
a = tf.reshape(a,[2,2,4])
print(a)
a = tf.sort(a,direction = 'DESCENDING')
print(a)

#%%某些时候我们不需要对所有的内容进行排序
#一般在分类过程中，我们会考察top5的错误率
#使用top_k 来返回前k个值
#该函数自动就是为最后一个维度排序的 #这点非常需要注意
a = np.random.randn(9) * 10
a = a.reshape(3,3)
a = tf.convert_to_tensor(a,dtype = tf.int32)
print(a)

prob = tf.constant([[0.1,0.2,0.7],[0.2,0.7,0.1]])
prob_T = tf.transpose(prob)
target = tf.constant([2,0])

k_b = tf.math.top_k(prob_T,2).indices
print(k_b)

#%%
#如下代码实现了一个计算topk 准确率的函数
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices #获得top_k的索引
    #注意到这里索引就代表着你在这个输入预测的正确值的多少，因此可以直接与target进行比较
    pred = tf.transpose(pred, perm=[1, 0]) #所以这里为什么要进行转置？
    #将获得的索引进行转置
    print(pred.shape)
    print(target.shape)
    target_ = tf.broadcast_to(target, pred.shape)
    # [10, b]
    correct = tf.equal(pred, target_)
    print(correct)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k* (100.0 / batch_size) )
        res.append(acc)

    return res



output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob:', output.numpy())
pred = tf.argmax(output, axis=1)
print('pred:', pred.numpy())
print('label:', target.numpy())

acc = accuracy(output, target, topk=(1,2,3,4,5,6))
print('top-1-6 acc:', acc)



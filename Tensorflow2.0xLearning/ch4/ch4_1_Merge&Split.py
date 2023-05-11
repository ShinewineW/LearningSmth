# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:45:48 2020
@Discription:详细说明了合并与拆分操作
tf.concat:在指定轴上进行合并，必须保证除了指定合并的轴，其他轴大小必须完全一致
tf.stack:在指定轴上创建出一个新的维度，然后进行合并
tf.unstack:在指定轴完全拆去，使用之后指定轴会消失，维度大小-1，返回一个长度为指定轴大小的列表
tf.split:在指定轴，按照你设定的数量，或者具体的大小来进行拆分，返回一个长度受控制的列表
@author: Administrator
"""

import tensorflow as tf
import numpy as np

#%%拼接操作
#例如收集成绩单，1号同学收集1到4号班级的成绩单，2号同学收集5号到6号的成绩单
#完成收集后，需要拼接
a = tf.ones([4,35,8])
b = tf.ones([2,35,8])

c = tf.concat([a,b],axis = 0)
#指定 a,b在0号轴上进行拼接
print(c.shape)

#注意拼接操作必须保证，所有的维度都已经设置好，并且除了拼接轴以外的其他轴应该保证一致

#%%如果轴不一致，那么就需要使用stack来额外生成一个轴
#例如收集完成两个学校的所有科目数据，此时需要生成一个新轴来标记是哪个学校
#此时使用tf.stack
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])

c = tf.stack([a,b],axis = 0)
print(c.shape)

#综上 不管是哪种合并操作，都必须保证操作维度之外，其他维度全部相等
#拆分使用unstack即可
#%%拆分操作 unstack是对指定维度全部打散为1，然后返回一个包含该维度长度的一个list
a = tf.zeros([4,35,8])

c = tf.unstack(a,axis = -1)

print(len(c))

for itr in c:
    print(itr.shape)

#使用unstack之后指定维度就会消失！！

#%%如果不想彻底打散，而是希望拆分成指定的大小 使用 split
a = tf.zeros([4,35,8])

c = tf.split(a,axis= -1,num_or_size_splits = 2)

print(len(c))
#同样也可以把一个具体的数量列表传递给 num关键词，然后会自动分割
c = tf.split(a,axis= -1,num_or_size_splits = [1,4,-1])
print(len(c))
for itr in c:
    print(itr.shape)






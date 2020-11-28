# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:10:02 2020
@Discription: 切片操作，大体上与numpy一致，重点是有三个函数
tf.gather: 在某个指定轴上，按照indixes的列表中的值来进行数据的抽取
tf.gather_nd: 在某个指定的确切坐标下，按照给定的silices进行数据的抽取，但是务必保证每个抽取的值是一样的
tf.boolean_mask: 对于某个tensor按照指定的mask来进行数据抽取，注意如果指定了维度，同样务必保证mask与删除该维度之后的维度大小保持一致
@author: Administrator
"""

import tensorflow as tf
import numpy as np

#基本下标索引  通过指定下标来寻找数据
a = tf.fill([1,2,3,4],3)

print(a[0][0])

print(a[0][0][0])

#使用shape函数来先确认大小 然后通过shape来决定
#切片方式与numpy几乎完全一致
#具体参考Pythonlearning文件夹下的Pylearning.py中关于numpy的切片索引操作

#上述都是顺序采样，下面说明乱序采样的方法
#主要用于shuffle，打乱数据集使用的方法。即根据一个index表，来抽取tensor中的数据
#例如对于一个 4个班级 35个学生 每个学生8门功课的一个tensor
b = tf.zeros([4,35,8])

c = tf.gather(b,axis =0,indices = [2,3]) #意味着取出 第2班 第3班的所有学生功课数据
print ( c.shape)

print (tf.gather(b,axis = -1,indices= [3,4,23,3,24]).shape)

#按照某种指定顺序来进行收集
#如果要对两个维度的数据都要进行随机采样，那么就执行两次tf.gather即可
#例如对4个班级的5名学生的2们功课组合成一个tensor
bb = tf.gather(b,axis = 1,indices = [2,5,6,3,4])
bbb = tf.gather(bb,axis = -1,indices = [4,6])
print(bbb.shape)

#但是这样很麻烦，如何在多个维度指定所需要的值？
#例如我们想要知道 1号班级的1号学生，2号班级的2号学生的8门课成绩
#此时gather就不适合，可以考虑使用gather_nd
print(tf.gather_nd(b,[0,1,2]).shape)
print(tf.gather_nd(b,[[0,1,2]]).shape)
#这两者的区别在于最贴近里面的中括号代表了要取的东西的下标索引
#而最外面的代表你要取多少个.因此如果只有一个中括号  那么结果就直接是原来切片的大小
print(tf.gather_nd(b,[[0,1],[1,3]]).shape)
#这个结果就表明切出来的是 2位同学的  8门课的成绩数据
#由于gather_nd可以指定非常精细的维度，因此自己维护起来非常困难，建议2个中括号
#即可

#最后一种 根据mask来索引 一般常见于dropout的使用
a = np.arange(4).reshape(2,-1)
b = tf.convert_to_tensor(a)
mask = tf.convert_to_tensor(a % 2 == 0)

print(b)
print(mask)
#将 b 按照mask来进行取值
c = tf.boolean_mask(b,mask)
print(c)













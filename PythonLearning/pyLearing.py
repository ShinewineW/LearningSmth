# -*- coding: utf-8 -*-
"""
Spyder Editor
Writen by Netfather
Learned from bilibili?keyword=三小时教你入门numpy
This is a temporary script file.
"""

import numpy as np


#%%
#1.
#进行数组的初始化操作
a = np.array([[1,2,3,4,5,6],[10,11,12,13,14,15]])

#%%
#2.进行数组的切割操作 
# 冒号表示这一整行或者这一阵列我全都要
# a:b:c 表示从a开始包括a，到b为止不包括b，步进c的长度的一个切割。
b = a[1:3]
c = a[1:3,:]
d = a[:,1:-2:]

print(b,c,d,sep=('\n'))

#%%
#3.进行数组的mask操作，
#通过一些标志位我们可以很方便的实现对数组的操作

mask_input = np.array([
    -5,1,4,-45,23,-67,-43
    ])

print(mask_input[mask_input < 0])

#mask_input[mask_input < 0] = 0;

print(mask_input)

print(mask_input>3)

#选定某个范围
#例如选定mask_input的所有大于3小于45的值
#直觉告诉我们可以使用如下代码
#print(mask_input > 3 and mask_input < 45)
#但是上述代码是错误的
#这是由于and关键字引起的，让python产生了歧义，他不知道到底是对整个输入判定还是
#当中的某个数进行判定。所以根据提示使用a.any()和a.all()或者使用C++类似的
# & 代表 and 
# | 代表 or
# ~ 代表 not 
# ^ 代表 xor
print((mask_input > 3) & (mask_input < 45))

#%%
#4.花式切片操作
a = np.arange(0,36,1)
a = a.reshape(6,6)

print (a)

#切割对角线上的出来,直接传入对角坐标
b = a[[0,1,2,3,4],[1,2,3,4,5]]
print(b.flags.owndata)
print(b)

#通过mask提取所有能被3整除的数字
#注意这种方法不会保留数字在原始数组中的位置
print(a[( a % 3 == 0)])
#如果想要保持该有的位置使用如下方法
output = np.empty_like(a,dtype='float')  #生成一个类似a的矩阵，但是这其中都是垃圾数据，无法调用
output.fill(np.nan)#往其中填充nan
mask = a%3 == 0 #获得所有能被三整除数字的掩码
output[mask] = a[mask] #将mask中的位置与新建的output一一对应
print(output)
#但是这个方法相对原始，可以直接使用np.where功能
test = np.where(a%3 == 0,a,np.nan)

print(b.flags.owndata) #arg1 条件 arg2 满足条件的为多少 arg3不满足条件的为多少

#%%
#5.多维数组
#当拥有2维以上的数据时，numpy到底是如何操作的
#我们每当增加一个维度，几乎就会将现在的维度全部破坏，把新加的维度添加到第一个位置
#就像 从一维到二维，原本第一个是列，但是二维中第一个就是行了
#对于numpy最后一个元素永远是列。
#规则1：当你在数组间进行操作时，numpy做的第一件事就是检查二者的形状是否匹配（广播规则）
#规则2：元素与元素一一对应，反应在值上
#规则3：所有的统计操作，例如求和，均值，反差，平均数都是作用于整个矩阵，除非我们指定轴
#规则4：如果你的矩阵中缺失一个值，矩阵依然会运算，但是要注意nan的副作用
a = np.arange(0,6).reshape(2,3)

print(a)

print(np.sum(a))

#%%
#6.指定轴问题
#规则三所写指定轴意味着，一旦你指定了轴，那就意味着在输出中，该轴就消失了，例子如下
#对于一个 （2，3，4）维度的矩阵，指定求和维度1，输出就是一个(2,4)维度的矩阵
a = np.arange(0,24).reshape(2,3,4)

print(a.sum(axis = 1) , a.sum(axis = 1).shape)

print(a.sum(axis = -1) , a.sum(axis = -1).shape)

#注意！如下这个例子说明，那个消失的维度是不会在保留在数据中的，因此对于一个二维
#的矩阵，一旦指定一个维度之后，所有的结果形状只会是（2，）

b = np.arange(0,6).reshape(2,3)

print(b.sum(axis = -1) , b.sum(axis = -1).shape)












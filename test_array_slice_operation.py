# -*- coding: utf-8 -*-
"""
Spyder Editor

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

mask_input[mask_input < 0] = 0;

print(mask_input)



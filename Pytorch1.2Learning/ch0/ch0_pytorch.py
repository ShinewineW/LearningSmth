# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch0_pytorch
# Description:  
# Author:       Administrator
# Date:         2021/1/22
# -------------------------------------------------------------------------------
import torch
import numpy as np
import os

# print(torch.cuda.is_available())

# pytorch基本数据类型
# IntTensor FloatTensor DoubleTensor

## 1. Torch数据类型测试
a = torch.randn([2,3])
print(a.type())
print(type(a))
print(isinstance(a, torch.FloatTensor))
print(a)

## 2. 将数据搬运到GPU中
# device0 = torch.device('cuda', 0)
# b = torch.randn([2,3],device=device0)
# print(isinstance(b, torch.FloatTensor))
# print(type(b))
# print(b)
# print(isinstance(b, torch.cuda.DoubleTensor))

##
# print(a.shape)
# print(b.shape)


## 维度变换规则 squeeze unsqueeze
# 然后唯一的区别在于再tf中  我们可以使用transpose来对不同维度整理
# 但是再tensor中 使用的是 permute 而且这不是一个函数，这是一个基于对象的函数方法。

##
# pytorch中的合并与分割操作
# 1. Cat操作 Concatenates the given sequence of seq tensors in the given dimension.
# All tensors must either have the same shape (except in the concatenating dimension) or be empty.
# 除了合并维度之外的维度必须完全一致
a = torch.rand(size=[4,32,8])
b = torch.rand(size=[4,32,8])
c = torch.empty(size = [8,32,8])
torch.cat([a,b],dim = 0,out = c)
print(c.shape)

# 2. Stack操作 Concatenates sequence of tensors along a new dimension.
# 直接在指定维度上创建一个新的维度，要求所有参与stack的维度必须完全一致
d = torch.stack([a,b],dim = 1)
print(d.shape)

# 3. Split操作 Splits the tensor into chunks.
# 将张量在指定维度打散为chunks  返回的应该是一个张量列表
# 第二个参数如果是数字 表示每个chunks的大小都是这个数字，最后一个长度根据维度来
#          如果是列表，那么就直接根据列表的数字来
print("------Split INT input-------")
e = c.split(split_size= 3,dim= 0)
for element in e:
    print(element.shape)
print("------Split LIST input-------")
d = c.split(split_size=[3,5],dim= 0)
for element in d:
    print(element.shape)

# 4. Chunk操作 Splits a tensor into a specific number of chunks.
# 理论上这个chunk是被split包含的
# 这个函数只能接受int值
print("------Chunk INT input-------")
f = c.split(split_size= 5,dim= 0)
for element in f:
    print(element.shape)



##
# Pytorch中的基本数学运算
# 加减乘除 Add,Sub,Mul,Div
# 矩阵相乘 matmul  或者使用符号 @
# 幂运算 Power 或者使用符号**
# 剪裁 Clamp 对标 tf的clip操作 能够将张量的值  锁定在指定范围内

# Pytorch中的基本统计运算
# 范数 使用norm  # 还可以通过指定维度 以及是否 keep_dim来保持输出的张量大小不变
norm_test = torch.ones(size= [2,4])
print(norm_test.norm(1))
print(norm_test.norm(2))
# min max mean prod argmax argmin
# 当你制定了 dims 和 keepdims的时候 你使用 max函数 返回的是一个命名好的元组 分别标记为 (values, indices)
max_test = torch.rand(size= [2,3])
max_test_return = max_test.max(dim= 1, keepdim= True)
print(max_test_return)
print(max_test_return[0])
print(max_test_return[1])
# topk 返回指定轴上的 前k个最大值 同时和上面max的返回一样 返回的是一个命名好的元组 分别标记为 (values, indices)
print("------topk func test-------")
topk_test = torch.rand(size= [2,3])
print(topk_test)
topk_test_return = topk_test.topk(k= 2, dim= 1) # 在第0个轴上  返回前2最大的元素  # 注意 是指在这个轴上！！！
print(topk_test_return)
print(topk_test_return[0])
print(topk_test_return[1])


## Pytorch中的高阶操作
# 1. Where操作
# Return a tensor of elements selected from either input or other, depending on condition.
# 根据第一个条件 来将返回的元素按个取出来  具体用法和 np.where操作一致

# 2. Gather操作
# 在指定维度上 按照给定的index来进行收集合并

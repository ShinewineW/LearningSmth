# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch9_RNNcell
# Description:  
# Author:       Administrator
# Date:         2021/1/8
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##
RNNCell = layers.SimpleRNNCell(3)
RNNCell.build(input_shape=(None, 4))

# 使用RNNCell.trainable_variables 进行查看 得到如下输出
# [<tf.Variable 'kernel:0' shape=(4, 3) dtype=float32, numpy=
#  array([[ 0.61222756,  0.71201444, -0.8414649 ],
#         [ 0.21811128, -0.8361399 , -0.5725672 ],
#         [ 0.88111484,  0.90567493, -0.1974346 ],
#         [-0.37566787,  0.0725531 ,  0.7435373 ]], dtype=float32)>,
#  <tf.Variable 'recurrent_kernel:0' shape=(3, 3) dtype=float32, numpy=
#  array([[-0.71919477,  0.6710334 , -0.18020317],
#         [ 0.26817134,  0.5073451 ,  0.8189537 ],
#         [-0.6409704 , -0.5406619 ,  0.5448318 ]], dtype=float32)>,
#  <tf.Variable 'bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]
# 其中的 kernel:0 是用于给输入 Xi 进行提取特征的 权重矩阵 对应于 Wxa
# 其中的 recurrent_kernel:0 是用于给激活值上一层进行特征提取的 权重矩阵 对应于 Waa
# 偏置不需要多说

##
x = tf.random.normal([4,80,100]) # 4个训练句子，每个句子80个词，每个词语用100维向量表示
xt0 = x[::,0,::]

RNNCell_test2 = layers.SimpleRNNCell(64) # 构建一个隐藏64维度矩阵，用于将数据进行投影

out , ht1 = RNNCell_test2(xt0,[tf.zeros([4,64])]) # 需要注意 这里的ht1位置的返回值是一个list，这个list包含所有的隐藏状态。
# 这里区分的是 LSTM  所有的隐藏状态 都要用列表的形式体现   这里因为只有一个隐藏状态 所以要求不是很明显  但是使用LSTM时，列表的作用就体现出来了

print(out.shape)
print(ht1[0].shape)

print(id(out),id(ht1[0]))

##


# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch9_WordEmbedding
# Description:  本文件说明了如何使用自带的Embedding自己训练一个 词嵌入矩阵 一般来说是用不上
#               一般我们偏向于选择已经完全训练好的模型 Word2vec 或者 Glove 向量来实现
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


##在文本类语言中，最重要的就是embedding方法，传统的独热编码方式有非常大的缺陷
# 使用词语嵌入，可以很方便的将词汇转换成 语义关联 特征关联的特征表格

# 1.不使用已经存在的embeding方法，自己随机初始化一个embedding layers
x = tf.range(5)
x = tf.random.shuffle(x)

net = layers.Embedding(10,4)   # 这里是构建了一个embedding层，第一个向量表示要学习的总单词数量。第二个表示从这些向量中
# 抽取得到的特征。 这是一个随机初始化的Embedding  没有训练过，所有参数都是随机的
y = net(x)
print(y.shape)
##



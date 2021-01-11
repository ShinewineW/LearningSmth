# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch9_RNN_SenmantiClassification
# Description:  情感分类问题  通过一层RNN 构建了一个最简单的通过文字评论来进行输出的网络，
#               分为两种写法，一种用最底层的SimpleRNNCell来写
#                           另一种用keras自带的SimpleRNN来写
# Author:       Administrator
# Date:         2021/1/11
# -------------------------------------------------------------------------------
##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##
# 引入数据集
total_words = 10000  # 定义最常见单词上线，可调
(X_train,Y_train),(X_test,Y_test) = keras.datasets.imdb.load_data(num_words=total_words)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print(X_train[0],Y_train[0])
print(type(X_train[0]))
# 从输出我们可以看出，测试和训练都是 batch = 25000 然后输入的单词已经idx完成了，可以直接数字来进行访问

##
# 将输入数据自动pading为最长的数据，这里我们既可以使用手动方式来进行padding，也可以直接使用keras中的函数来进行
max_sentence_word_lenth = 80 # 每个句子最多只看80个单词，超过的不看，没超过的后续部分全部设置为0
bathsize = 32
X_train = keras.preprocessing.sequence.pad_sequences(X_train,maxlen=max_sentence_word_lenth)
X_test = keras.preprocessing.sequence.pad_sequences(X_test,maxlen=max_sentence_word_lenth)

# 下面按照之前的方法构建数据集
db_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
db_train = db_train.shuffle(25000).batch(bathsize,drop_remainder= True)
db_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
db_test = db_test.batch(bathsize,drop_remainder= True) #将最后一个剩下的余数 不能被batchsize的扔掉，这种问题不会出现在卷积中，但是在RNN中会出现

sample = iter(db_train)
sample = next(sample)
print(sample[0].shape,sample[1].shape)

##
# 使用RNNCell 来进行底层的循环
embedding_len = 50

class MyRNN(keras.Model):
    def __init__(self,units_layer1,units_layer2):
        super(MyRNN, self).__init__()

        self.embedding = layers.Embedding(total_words,embedding_len,input_length=max_sentence_word_lenth) #第一个参数表示你的数据集中各不相同的词汇有多少，后面表示你要把这些词汇从稀疏连接投影到什么维度的密集连接上来
        # 构建一个RNNCELL
        self.rnn_cell0 = layers.SimpleRNNCell(units_layer1,dropout=0.2)
        self.state0_layer1 = [tf.zeros([bathsize,units_layer1])] # 初始化最开始的向量
        self.rnn_cell1 = layers.SimpleRNNCell(units_layer2,dropout=0.2)
        self.state0_layer2 = [tf.zeros([bathsize,units_layer2])]
        # 构建一个将units 投影为待输出的 FC
        self.fc = layers.Dense(1)

    def call(self,inputs,training = None):
        """

        :param inputs: [b,80]
        :param training: 表明是否train 还是 test
        :return: Output of RNNModel
        """
        X = self.embedding(inputs) # 词嵌入之后，输出的X为 [batchsize,80,词嵌入矩阵维度 这里为50]
        print(X.shape)
        hidden_layer1 = self.state0_layer1
        hidden_layer2 = self.state0_layer2
        for word in tf.unstack(X,axis = 1):
            out_layer1,hidden_layer1 = self.rnn_cell0(word,hidden_layer1,training)
            out,hidden_layer2 = self.rnn_cell1(out_layer1,hidden_layer2,training)
        print(out.shape)
        X = self.fc(out)
        prob = tf.sigmoid(X)
        return prob

# X = tf.random.normal(shape=(32,80))
# Test_Y = RNNModel(X)
# print(Test_Y.shape)
# print(RNNModel.summary())
# 一定要分清楚 这里面的tricky point 。 batchsize  句子长度 不同词语个数 词嵌入维度！ 这些一定要分清楚

##
RNNModel = MyRNN(64,32)
RNNModel.compile(optimizer= keras.optimizers.Adam(learning_rate= 1e-3),
                 loss= keras.losses.BinaryCrossentropy(from_logits= True),
                 metrics=['accuracy'])

RNNModel.fit(db_train,epochs= 2,validation_data=db_test)


##
# 下面直接使用自带的SimpleRNN来进行构建
def RNNSimpleModel(units_layer1,units_layer2):
    inputs = keras.Input(shape=[max_sentence_word_lenth])
    # 经过embeding 从[batchsize,80] 到 [batchsize,80,50]
    X = keras.layers.Embedding(total_words,50,input_length= max_sentence_word_lenth)(inputs)
    print(X.shape)
    # 经过一层RNN
    X,test = layers.SimpleRNN(units_layer1,return_sequences=True,return_state=True,dropout= 0.2)(X)
    print(X.shape)
    print(test.shape)
    X = layers.SimpleRNN(units_layer2,dropout= 0.2)(X)
    # 注意 这里 如果 两个return都设定True 那么两个返回的都是正常的张量
    # 如果这里只设定return_state = True 那么这里返回的是一个列表
    # 如果什么都不设定，那么就直接返回最后一个隐藏状态的张量
    # print(len(X))
    # print(X[0].shape,id(X[0]))
    # print(X[1].shape,id(X[1]))
    X = layers.Dense(1,activation='sigmoid')(X)

    MyModel = keras.Model(inputs = inputs,outputs = X)
    return MyModel

# MyModel = RNNSimpleModel(64,32)
# X = tf.random.normal(shape=(32,80))
# Y = MyModel(X)
# print(Y.shape)

MyModel = RNNSimpleModel(64,32)
MyModel.compile(optimizer= keras.optimizers.Adam(learning_rate= 1e-3),
                 loss= keras.losses.BinaryCrossentropy(from_logits= True),
                 metrics=['accuracy'])

MyModel.fit(db_train,epochs= 2,validation_data=db_test)


##


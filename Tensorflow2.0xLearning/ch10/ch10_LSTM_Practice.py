# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch10_LSTM_Practice
# Description:  本文文件完成了LSTM的实例化
#               但是可能是tf的版本问题，使用LSTMCell 在fit时候会报错，具体原因未知
#               直接使用LSTM很方便 而且关于 return_sequence 和 return_state的不同需要注意
#               关于这两个参数的辨析 请参看 https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/ 的解释
# Author:       Administrator
# Date:         2021/1/12
# -------------------------------------------------------------------------------
# 导入必要包
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def Generate_Dataset_imdb (BatchSize,MAX_DICTIONARY = 10000,MAX_SENTENCE_WORDS = 80,drop_remainder= False):
    '''
    函数用于自动生成满足tf.keras格式的imdb 电影评价数据集的 处理完成数据集
    :param MAX_DICTIONARY: 一共有10000个 各不相同的词
    :param MAX_SENTENCE_WORDS: 每个句子 最多看80个单词，超过的不看，没超过的padding为0
    :param BatchSize: 生成的数据集有多少个batchsize
    :param drop_remainder: 是否在batchsize中保留尾数，False表示保留尾数， True表示不保留尾数
    :return: 返回两个 tf.datasets格式的数据集 (train_db,test_db)
    '''
    # 1. 从datasets中引入数据数据
    (X_train,Y_train),(X_test,Y_test) = keras.datasets.imdb.load_data(num_words=MAX_DICTIONARY)
    num_train = Y_train.shape[0] #用于打乱
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    # 输出一个样例
    print(X_train[0],Y_train[0])
    print(type(X_train[0]))
    print(Y_train.max(),Y_train.min())

    # 2. 将导入的数据集进行处理，
    # 指定格式化大小
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,maxlen= MAX_SENTENCE_WORDS)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,maxlen= MAX_SENTENCE_WORDS)

    # 构建数据集
    db_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
    db_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test))

    # 进行打乱和生成batchsixze
    db_train = db_train.shuffle(num_train).batch(BatchSize,drop_remainder= drop_remainder)
    db_test = db_test.batch(BatchSize,drop_remainder=drop_remainder)

    # 3. 测试结果
    sample = iter(db_train)
    sample = next(sample)
    print(sample[0].shape,sample[1].shape)
    sample = iter(db_test)
    sample = next(sample)
    print(sample[0].shape,sample[1].shape)

    return db_train,db_test


# Test
# db_train , db_test = Generate_Dataset_imdb(BatchSize= 32,drop_remainder= True)
# OK！ (32, 80) (32,)

# 构建LSTMCell 搭建的基础循环神经网络类
class MyRNN(keras.Model):
    def __init__(self,units_layer1,units_layer2,embedding_len,BatchSize,MAX_DICTIONARY = 10000,MAX_SENTENCE_WORDS = 80):
        '''
        从父类继承过来的初始化函数，用于初始化所有模型所需要的层的参数
        :param units_layer1: 第一层RNN网络的隐层神经元个数
        :param units_layer2: 第二层RNN网络的隐层神经元个数
        :param embedding_len:  词嵌入层的维度
        :param BatchSize:  数据集中所使用的Batchsize批次，用于初始化隐藏状态 c0 和 a0
        :param MAX_DICTIONARY: 词典大小，用于初始化embedding层
        :param MAX_SENTENCE_WORDS: 句子最大长度，用于初始化embedding层
        '''
        super(MyRNN, self).__init__()

        self.embedding = layers.Embedding(MAX_DICTIONARY,embedding_len,input_length=MAX_SENTENCE_WORDS) #第一个参数表示你的数据集中各不相同的词汇有多少，后面表示你要把这些词汇从稀疏连接投影到什么维度的密集连接上来
        # 构建一个RNNCELL 使用LSTM来构建
        self.cnn_cell0 = layers.LSTMCell(units_layer1,dropout= 0.2)
        self.state0_layer1 = [tf.zeros([BatchSize,units_layer1]),tf.zeros([BatchSize,units_layer1])] # [c0,a0]
        self.cnn_cell1 = layers.LSTMCell(units_layer2,dropout= 0.2)
        self.state0_layer2 = [tf.zeros([BatchSize,units_layer2]),tf.zeros([BatchSize,units_layer2])] # [c0,a0]
        # self.rnn_cell0 = layers.SimpleRNNCell(units_layer1,dropout=0.2)
        # self.state0_layer1 = [tf.zeros([BatchSize,units_layer1])] # 初始化最开始的向量
        # self.rnn_cell1 = layers.SimpleRNNCell(units_layer2,dropout=0.2)
        # self.state0_layer2 = [tf.zeros([BatchSize,units_layer2])]
        # 构建一个将units 投影为待输出的 FC
        self.fc = layers.Dense(1)

    def call(self,inputs,training = None):
        """

        :param inputs: [b,80]
        :param training: 表明是否train 还是 test
        :return: Output of RNNModel
        """
        X = self.embedding(inputs) # 词嵌入之后，输出的X为 [batchsize,80,词嵌入矩阵维度 这里为50]
        # print(X.shape)
        hidden_layer1 = self.state0_layer1
        hidden_layer2 = self.state0_layer2
        for word in tf.unstack(X,axis = 1):
            out_layer1,hidden_layer1 = self.cnn_cell0(word,hidden_layer1,training)
            out,hidden_layer2 = self.cnn_cell1(out_layer1,hidden_layer2,training)
        # print(out.shape)
        X = self.fc(out)
        prob = tf.sigmoid(X)
        return prob


db_train , db_test = Generate_Dataset_imdb(BatchSize= 32,drop_remainder= True)
# RNNModel = MyRNN(32,32,50,32)  不使用子类方式 有bug 问题不知
def RNNSimpleModel(units_layer1,units_layer2,embedding_units,MAX_DICTIONARY = 10000,MAX_SENTENCE_WORDS = 80):
    inputs = keras.Input(shape=[MAX_SENTENCE_WORDS])
    # 经过embeding 从[batchsize,80] 到 [batchsize,80,50]
    X = keras.layers.Embedding(MAX_DICTIONARY,embedding_units,input_length= MAX_SENTENCE_WORDS)(inputs)
    # print(X.shape)
    # 经过一层RNN
    X,test_memory,test_carry = layers.LSTM(units_layer1,return_sequences=True,return_state=True,dropout= 0.2)(X)
    # print(X)
    # print(test_memory)
    # assert(tf.equal())
    #具体解释 请看 https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/ 的解释
    #这里  X返回的sequence中的最后一个元素 应该 等于 test_memory
    # print(X.shape)
    # print(test.shape)
    # print(test_memory.shape)
    X,test_memory,test_carry = layers.LSTM(units_layer2,dropout= 0.2,return_state = True)(X)
    # print(X)
    # print(test_memory)
    #这里 X 的值应该和test_memory相等
    # print(X[0].shape,id(X[0]))
    # print(X[1].shape,id(X[1]))
    X = layers.Dense(1,activation='sigmoid')(X)

    MyModel = keras.Model(inputs = inputs,outputs = X)
    return MyModel
# X = tf.ones(shape=(32,80))
# Y = RNNModel(X)
# print(Y.shape)
RNNModel = RNNSimpleModel(64,48,50)
# X = tf.ones(shape=(3,1))
# Y = RNNModel(X)
RNNModel.compile(optimizer= keras.optimizers.Adam(learning_rate= 1e-3),
                 loss= keras.losses.BinaryCrossentropy(),
                 metrics= ['accuracy'])

RNNModel.fit(db_train,validation_data= db_test,epochs= 4)
#RNNModel.build(input_shape= (None,80))
# print(RNNModel.summary())


##


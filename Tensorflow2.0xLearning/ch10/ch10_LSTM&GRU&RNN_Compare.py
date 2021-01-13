# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch10_LSTM&GRU&RNN_Compare
# Description:  详细比较了return_state 和 return_sequence两个关键参数在不同RNN单元中带来的结果不同的问题
#               一言以蔽之： 只要 return_sequence开启，那么结果输出就会被替换成sequence输出
#                           只要return_state开启，那么不管激活值是否和结果一致，这个激活值都会被加入到原有的输出后
#                           如果二者同时开启，那么就会变成 返回sequence和state
#                           实际中 除非想要对LSTM中的隐藏层carry值做操作，否则都可以不开启return_state
#               详细比较了 biodirectional 层的 merge_mode 参数对输出造成的影响 要特别注意concat模式带来的维度翻倍！！

# Author:       Administrator
# Date:         2021/1/13
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics,Input,Model
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% 实现RNN网络，并辨析其中  return_sequences=False, return_state=False 这两个参数的异同点
# 由于RNN网络是一个激活值和隐藏状态carry完全一致的网络模型，那么理论上
# return_sequences=False, return_state=False: 只返回一个值，这个值的输出大小为(batchsize,hidden_units) 这个就是RNN最后一个位置cell的结果输出
# return_sequences=True, return_state=False: 只返回一个值，大小为(batchsize,Sequence_Len,hidden_units) 这个就是RNN每一个Cell的激活输出
# return_sequences=False, return_state=True: 返回两个值，第一个值大小为 (batchsize,hidden_units) 为最后一个位置cell的结果输出
#                                                      第一个值大小也为 (batchsize,hidden_units) 为最后一个位置cell的激活
#                                            由于SimpleRNN特性，我们这里可以看到 这两个值的大小是完全一致的！虽然id不同，但是值完全一致
# return_sequences=True, return_state=True: 返回两个值， 第一个为大小为(batchsize,Sequence_Len,hidden_units) 这个就是RNN每一个Cell的激活输出
#                                                       第二个值大小为(batchsize,hidden_units) 为最后一个cell的激活输出
#                                           由于SimpleRNN特性，我们这里可以看到 RNN的最后一个元素是完全和第二个返回值一致的！因为Senquence的最后一个激活值就是等于最后一个激活输出

# 同样对于双向SimpleRNN网络，merge_mode 会显著的影响各个返回值的大小，注意使用时的大小错误问题。
# 同时双向的使用，并不会对上面使用不同return返回值的大小产生影响。


def SimpleRNN_Compare(X,MAX_SENTENCE_LENTH = 3,Embedding_dims = 1):
    '''
    用于辨析keras中SimpleRNN单元 return_sequences 和 return_state 的不同
    :param X: 测试数据 输入大小固定为 (1,3,1) 1为batchsize 3为最大句子单词长度 1为通过词嵌入矩阵之后 映射到的特征维度
    :param MAX_SENTENCE_LENTH: 最大句子长度 与测试保持一致
    :param Embedding_dims: 词嵌入矩阵维度，与测试保持一致
    :return: 无返回
    '''

    inputs = Input(shape=(MAX_SENTENCE_LENTH,Embedding_dims))
    SimpleRNN_None = layers.SimpleRNN(4)(inputs)
    ModelRNN_None = Model(inputs = inputs,outputs = SimpleRNN_None)
    Y_None = ModelRNN_None.predict(X)
    print("return_sequences=False, return_state=False: the results is {}".format(Y_None))
    print("The shape of outputs is {}".format(np.array(Y_None).shape))
    print("---------------------------------------------------------------------------")

    SimpleRNN_Returnsequence = layers.SimpleRNN(4,return_sequences= True)(inputs)
    Model_Returnsequence = Model(inputs = inputs,outputs = SimpleRNN_Returnsequence)
    Y_Returnsequence = Model_Returnsequence.predict(X)
    print("return_sequences=True, return_state=False: the results is {}".format(Y_Returnsequence))
    print("The shape of outputs is {}".format(np.array(Y_Returnsequence).shape))
    print("---------------------------------------------------------------------------")

    SimpleRNN_Returnstates = layers.SimpleRNN(4,return_state=True)(inputs)
    Model_Returnstates = Model(inputs = inputs,outputs = SimpleRNN_Returnstates)
    Y_Returnstates = Model_Returnstates.predict(X)
    print("return_sequences=False, return_state=True: the results is {}".format(Y_Returnstates))
    print("The shape of outputs is {}".format(np.array(Y_Returnstates).shape))
    print("---------------------------------------------------------------------------")

    SimpleRNN_All = layers.SimpleRNN(4,return_state= True,return_sequences= True)(inputs)
    Model_All = Model(inputs = inputs, outputs = SimpleRNN_All)
    Y_All = Model_All.predict(X)
    for step,elements in enumerate(Y_All):
        print("return_sequences=True, return_state=True: the results on location {} is {}".format(step,elements))
    print("---------------------------------------------------------------------------")

def SimpleRNN_Bidirecation(X,MAX_SENTENCE_LENTH = 3,Embedding_dims = 1):
    '''

    :param X:
    :param MAX_SENTENCE_LENTH:
    :param Embedding_dims:
    :return:
    '''
    print("--------------------------------Auto Mode----------------------------------")
    inputs = Input(shape=(MAX_SENTENCE_LENTH, Embedding_dims))
    SimpleRNN_layers = layers.SimpleRNN(4)
    # SimpleRNN_layers = layers.SimpleRNN(4,return_sequences= True)
    Bidirectoional_SimpleRNN_ave = layers.Bidirectional(SimpleRNN_layers,merge_mode= 'ave')(inputs)
    Model_BiSimpleRNN_ave = Model( inputs = inputs, outputs = Bidirectoional_SimpleRNN_ave)
    Y_BiSimpleRNN_ave = Model_BiSimpleRNN_ave.predict((X))
    print("return_sequences=False, return_state=False: the ave mode results is {}".format(Y_BiSimpleRNN_ave))
    print("The shape of ave mode outputs is {}".format(np.array(Y_BiSimpleRNN_ave).shape))

    Bidirectoional_SimpleRNN_concat = layers.Bidirectional(SimpleRNN_layers, merge_mode='concat')(inputs)
    Model_BiSimpleRNN_concat = Model(inputs=inputs, outputs=Bidirectoional_SimpleRNN_concat)
    Y_BiSimpleRNN_concat = Model_BiSimpleRNN_concat.predict((X))
    print("return_sequences=False, return_state=False: the concat mode results is {}".format(Y_BiSimpleRNN_concat))
    print("The shape of concat mode outputs is {}".format(np.array(Y_BiSimpleRNN_concat).shape))

    print("----------------------------Return_Sequence True-----------------------------------")
    inputs = Input(shape=(MAX_SENTENCE_LENTH, Embedding_dims))
    # SimpleRNN_layers = layers.SimpleRNN(4)
    SimpleRNN_layers = layers.SimpleRNN(4,return_sequences= True)
    Bidirectoional_SimpleRNN_ave = layers.Bidirectional(SimpleRNN_layers,merge_mode= 'ave')(inputs)
    Model_BiSimpleRNN_ave = Model( inputs = inputs, outputs = Bidirectoional_SimpleRNN_ave)
    Y_BiSimpleRNN_ave = Model_BiSimpleRNN_ave.predict((X))
    print("return_sequences=False, return_state=False: the ave mode results is {}".format(Y_BiSimpleRNN_ave))
    print("The shape of ave mode outputs is {}".format(np.array(Y_BiSimpleRNN_ave).shape))

    Bidirectoional_SimpleRNN_concat = layers.Bidirectional(SimpleRNN_layers, merge_mode='concat')(inputs)
    Model_BiSimpleRNN_concat = Model(inputs=inputs, outputs=Bidirectoional_SimpleRNN_concat)
    Y_BiSimpleRNN_concat = Model_BiSimpleRNN_concat.predict((X))
    print("return_sequences=False, return_state=False: the concat mode results is {}".format(Y_BiSimpleRNN_concat))
    print("The shape of concat mode outputs is {}".format(np.array(Y_BiSimpleRNN_concat).shape))

#TEST OK
# X = tf.ones(shape=[1,3,1])
# SimpleRNN_Compare(X)
# SimpleRNN_Bidirecation(X)

#%% 实现GRU网络，并辨析其中  return_sequences=False, return_state=False 这两个参数的异同点
# 由于GRU网络是一个激活值和隐藏状态carry完全一致的网络模型，那么理论上
# return_sequences=False, return_state=False: 只返回一个值，这个值的输出大小为(batchsize,hidden_units) 这个就是GRU最后一个位置cell的结果输出
# return_sequences=True, return_state=False: 只返回一个值，大小为(batchsize,Sequence_Len,hidden_units) 这个就是GRU每一个Cell的激活输出
# return_sequences=False, return_state=True: 返回两个值，第一个值大小为 (batchsize,hidden_units) 为最后一个位置cell的结果输出
#                                                      第一个值大小也为 (batchsize,hidden_units) 为最后一个位置cell的激活
#                                            由于GRU特性，我们这里可以看到 这两个值的大小是完全一致的！虽然id不同，但是值完全一致
# return_sequences=True, return_state=True: 返回两个值， 第一个为大小为(batchsize,Sequence_Len,hidden_units) 这个就是RNN每一个Cell的激活输出
#                                                       第二个值大小为(batchsize,hidden_units) 为最后一个cell的激活输出
#                                           由于GRU特性，我们这里可以看到 RNN的最后一个元素是完全和第二个返回值一致的！因为Senquence的最后一个激活值就是等于最后一个激活输出

# 同样对于双向GRU网络，merge_mode 会显著的影响各个返回值的大小，注意使用时的大小错误问题。
# 同时双向的使用，并不会对上面使用不同return返回值的大小产生影响。
def GRU_Compare(X,MAX_SENTENCE_LENTH = 3,Embedding_dims = 1):
    '''
    用于辨析keras中GRU单元 return_sequences 和 return_state 的不同
    :param X: 测试数据 输入大小固定为 (1,3,1) 1为batchsize 3为最大句子单词长度 1为通过词嵌入矩阵之后 映射到的特征维度
    :param MAX_SENTENCE_LENTH: 最大句子长度 与测试保持一致
    :param Embedding_dims: 词嵌入矩阵维度，与测试保持一致
    :return: 无返回
    '''

    inputs = Input(shape=(MAX_SENTENCE_LENTH,Embedding_dims))
    GRU_None = layers.GRU(4)(inputs)
    ModelRNN_None = Model(inputs = inputs,outputs = GRU_None)
    Y_None = ModelRNN_None.predict(X)
    print("return_sequences=False, return_state=False: the results is {}".format(Y_None))
    print("The shape of outputs is {}".format(np.array(Y_None).shape))
    print("---------------------------------------------------------------------------")

    GRU_Returnsequence = layers.GRU(4,return_sequences= True)(inputs)
    Model_Returnsequence = Model(inputs = inputs,outputs = GRU_Returnsequence)
    Y_Returnsequence = Model_Returnsequence.predict(X)
    print("return_sequences=True, return_state=False: the results is {}".format(Y_Returnsequence))
    print("The shape of outputs is {}".format(np.array(Y_Returnsequence).shape))
    print("---------------------------------------------------------------------------")

    GRU_Returnstates = layers.GRU(4,return_state=True)(inputs)
    Model_Returnstates = Model(inputs = inputs,outputs = GRU_Returnstates)
    Y_Returnstates = Model_Returnstates.predict(X)
    print("return_sequences=False, return_state=True: the results is {}".format(Y_Returnstates))
    print("The shape of outputs is {}".format(np.array(Y_Returnstates).shape))
    print("---------------------------------------------------------------------------")

    GRU_All = layers.GRU(4,return_state= True,return_sequences= True)(inputs)
    Model_All = Model(inputs = inputs, outputs = GRU_All)
    Y_All = Model_All.predict(X)
    for step,elements in enumerate(Y_All):
        print("return_sequences=True, return_state=True: the results on location {} is {}".format(step,elements))
    print("---------------------------------------------------------------------------")


# Test OK!
# X = tf.ones(shape=[1,3,1])
# GRU_Compare(X)

#%% 实现LSTM网络，并辨析其中  return_sequences=False, return_state=False 这两个参数的异同点
# 由于GRU网络是一个激活值和隐藏状态carry完全一致的网络模型，那么理论上
# return_sequences=False, return_state=False: 只返回一个值，这个值的输出大小为(batchsize,hidden_units) 这个就是LSTM最后一个位置cell的结果输出这个结果和激活值是一致的
# return_sequences=True, return_state=False: 只返回一个值，大小为(batchsize,Sequence_Len,hidden_units) 这个就是LSTM每一个Cell的激活输出
# return_sequences=False, return_state=True: 返回含有三个值的列表，第一个值大小为 (batchsize,hidden_units) 为最后一个位置cell的结果输出
#                                                      第二个值大小也为 (batchsize,hidden_units) 为最后一个位置cell的激活输出
#                                                      第三个值大小也为 (batchsize,hidden_units) 为最后一个位置cell的隐藏层Carry输出
#                                            由于LSTM特性，我们这里可以看到 这里前两个值的大小是完全一致的！虽然id不同，但是值完全一致
#                                                       但是! 隐藏层是和另外两个完全不同的输出
# return_sequences=True, return_state=True: 返回含有三个值的列表， 第一个为大小为(batchsize,Sequence_Len,hidden_units) 这个就是LSTM每一个Cell的激活输出
#                                                       第二个值大小为(batchsize,hidden_units) 为最后一个cell的激活输出
#                                                      第三个值大小也为 (batchsize,hidden_units) 为最后一个位置cell的隐藏层Carry输出
#                                           由于LSTM特性，我们这里可以看到 LSTM的最后一个激活值是完全和第一个sequence的最后一个值相等的！
#                                           因为Senquence的最后一个激活值就是等于最后一个激活输出

# 同样对于双向LSTM网络，merge_mode 会显著的影响各个返回值的大小，注意使用时的大小错误问题。
# 同时双向的使用，并不会对上面使用不同return返回值的大小产生影响。

def LSTM_Compare(X,MAX_SENTENCE_LENTH = 3,Embedding_dims = 1):
    '''
    用于辨析keras中LSTM单元 return_sequences 和 return_state 的不同
    :param X: 测试数据 输入大小固定为 (1,3,1) 1为batchsize 3为最大句子单词长度 1为通过词嵌入矩阵之后 映射到的特征维度
    :param MAX_SENTENCE_LENTH: 最大句子长度 与测试保持一致
    :param Embedding_dims: 词嵌入矩阵维度，与测试保持一致
    :return: 无返回
    '''

    inputs = Input(shape=(MAX_SENTENCE_LENTH,Embedding_dims))
    LSTM_None = layers.LSTM(4)(inputs)
    ModelRNN_None = Model(inputs = inputs,outputs = LSTM_None)
    Y_None = ModelRNN_None.predict(X)
    print("return_sequences=False, return_state=False: the results is {}".format(Y_None))
    print("The shape of outputs is {}".format(np.array(Y_None).shape))
    print("---------------------------------------------------------------------------")

    LSTM_Returnsequence = layers.LSTM(4,return_sequences= True)(inputs)
    Model_Returnsequence = Model(inputs = inputs,outputs = LSTM_Returnsequence)
    Y_Returnsequence = Model_Returnsequence.predict(X)
    print("return_sequences=True, return_state=False: the results is {}".format(Y_Returnsequence))
    print("The shape of outputs is {}".format(np.array(Y_Returnsequence).shape))
    print("---------------------------------------------------------------------------")

    LSTM_Returnstates = layers.LSTM(4,return_state=True)(inputs)
    Model_Returnstates = Model(inputs = inputs,outputs = LSTM_Returnstates)
    Y_Returnstates = Model_Returnstates.predict(X)
    print("return_sequences=False, return_state=True: the results is {}".format(Y_Returnstates))
    print("The shape of outputs is {}".format(np.array(Y_Returnstates).shape))
    print("---------------------------------------------------------------------------")

    LSTM_All = layers.LSTM(4,return_state= True,return_sequences= True)(inputs)
    Model_All = Model(inputs = inputs, outputs = LSTM_All)
    Y_All = Model_All.predict(X)
    for step,elements in enumerate(Y_All):
        print("return_sequences=True, return_state=True: the results on location {} is {}".format(step,elements))
    print("---------------------------------------------------------------------------")

# Test OK!
# X = tf.ones(shape=[1,3,1])
# LSTM_Compare(X)









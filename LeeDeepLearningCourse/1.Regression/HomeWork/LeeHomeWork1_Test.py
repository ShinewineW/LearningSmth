# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         LeeHomeWork1_Test
# Description:  
# Author:       Administrator
# Date:         2021/1/27
# -------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def Mymodel():
    inputs = layers.Input(shape=(9,17))
    flatten = layers.Flatten()(inputs)
    x1 = layers.Dense(512, activation="relu")(flatten)
    x1 = layers.Dense(256,activation="relu")(x1)
    x2 = layers.Dense(128)(x1)
    x3 = layers.BatchNormalization()(x2)
    x4 = layers.Activation('relu')(x3)
    _ = layers.Dense(64)(x4)
    _ = layers.BatchNormalization()(_)
    _ = layers.Activation('relu')(_)
    _ = layers.Dense(32)(_)
    _ = layers.BatchNormalization()(_)
    _ = layers.Activation('relu')(_)
    outputs = layers.Dense(1)(_)  #不是分类问题 因此不使用 sigmoid
    print(outputs.shape)
    model = keras.Model(inputs = inputs , outputs = outputs)
    return model


network = Mymodel()
network.load_weights(r'tmp/')

def Testfiles():
    testfiles = open("test.csv",mode= 'r',encoding="big5")
    rows = 0 #用于记录读出的行数
    features = 0 #用于记录每一天的数据
    list_perday_test = [] #记录每一天的所有列表
    list_allyear_test = [] #按照每一天收集所有的列表
    while True:
        onerow = testfiles.readline()
        if (onerow == ''):
            break
        #处理意外情况
        # Test 用
        # if (rows >= 20):
        #     break
        #对每一个相同的日期
        onelist = onerow.split(sep=',')
        if (rows < 4319):
            onelist[-1] = onelist[-1][0:-1] #最后一行去换行符
        if(features == 18): #到了第18个表示一天记录完成，清空标志，并将一天的放入总list中
            features = 0
            list_allyear_test.append(list_perday_test.copy())
            list_perday_test.clear() #清空之前的数据
        if(features == 10): # 第10个特征元素由于是NR，因此不记录
            features += 1
            rows = rows + 1
            continue
        templist = list(map(lambda x:float(x),onelist[2:])) #抽出feature
        list_perday_test.append(templist)
        features += 1
        rows = rows + 1
    # 最后一天数据压入
    list_allyear_test.append(list_perday_test.copy())
    list_perday_test.clear()  # 清空之前的数据
    # print(list_perday)
    # print(list_allyear)
    Test_X_Year = np.array(list_allyear_test) #转为numpy数组
    # print(Test_X_Year.shape)
    testfiles.close()
    Test_X_Year = np.transpose(Test_X_Year,axes=(0,2,1))
    Test_Y = network.predict(Test_X_Year)
    np.savetxt('output.csv',Test_Y,delimiter=',')

Testfiles()
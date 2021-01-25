# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         LeeHomeWork1_Regression
# Description:  作业kaggle提交
#               比较反人类的训练数据XD 结果尚可
# Author:       Administrator
# Date:         2021/1/25
# -------------------------------------------------------------------------------
## 导入必要包
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## 加载训练数据
# 打开文件
trainfiles = open("train.csv",mode= 'r',encoding="big5")
# 尝试读一行
# onerow = trainfiles.readline()
# print(onerow)
# onelist = onerow.split(sep=',')
# onelist[-1] = "".join(list(filter(lambda char:char.isdigit(),onelist[-1])))
# print(onelist)
# onerow = trainfiles.readline()
# print(onerow)
# onelist = onerow.split(sep=',')
# print(onelist)
# 开始进行处理，由于每一次都是18行的数据，我们要做的：
# 1. 首先毫无疑问18个测试元素除了一个NR的 其他的都是特征
# 2. 然后我们要根据日期，将原本竖列表示的列表，完全转换为横向表示
# 3. 将横向表示的数据通过滑动窗口每10个读出来，那么此时的数据矩阵大小应该为(batchsize,10,17) 再同时生成y
# 4. 最后通过神经网络进行训练
rows = 0 #用于记录读出的行数
features = 0 #用于记录每一天的数据
list_perday = [] #记录每一天的所有列表
list_allyear = [] #按照每一天收集所有的列表
while True:
    onerow = trainfiles.readline()
    if (onerow == ''):
        break
    #处理意外情况
    if (rows == 0):
        rows +=1
        continue
    # Test 用
    # if (rows >= 20):
    #     break
    #对每一个相同的日期
    onelist = onerow.split(sep=',')
    if (rows < 4320):
        onelist[-1] = onelist[-1][0:-1] #最后一行去换行符
    if(features == 18): #到了第18个表示一天记录完成，清空标志，并将一天的放入总list中
        features = 0
        list_allyear.append(list_perday.copy())
        list_perday.clear() #清空之前的数据
    if(features == 10): # 第10个特征元素由于是NR，因此不记录
        features += 1
        rows = rows + 1
        continue
    templist = list(map(lambda x:float(x),onelist[3:])) #抽出feature
    list_perday.append(templist)
    features += 1
    rows = rows + 1
# 最后一天数据压入
list_allyear.append(list_perday.copy())
list_perday.clear()  # 清空之前的数据
# print(list_perday)
# print(list_allyear)
Train_X_Year = np.array(list_allyear) #转为numpy数组
print(Train_X_Year.shape)
trainfiles.close()

##
# 将所有的string元素转为int
print(Train_X_Year)
## 2. 将数据以10为窗口滑动  抽取出Train 和 test
# 由于给定的数据只有当月的前20天 所以数据并不能一味的连续滑动，我们需要将数据再做个细分  分成每个月
Train_X_Year2 = np.zeros(shape= (20,17,24))  #12个月 每个月20天 每天24h 每小时17个参数
Train_X = []
Train_Y = []
for i in range(12):
    Train_X_Year2 = Train_X_Year[20*i : 20*(i+1),::,::]
    Train_X_Year2 = np.transpose(Train_X_Year2, axes=(0, 2, 1))
    # Train_X_Year2[::,::,::] = Train_X_Year[20*i : 20*(i+1),::,::]
    # Train_X_Year2 = np.transpose(Train_X_Year2,axes=(0,2,1))
    #将每个月前20天每10h的内容抽出来作为一个batch
    Train_X_Year2 = np.reshape(Train_X_Year2,newshape=(20*24,-1)) #(480,17)
    print(Train_X_Year2.shape)
    # print(Train_X_Year2[-3:,::])
    for i in range(471):
        temp_train = Train_X_Year2[i:i+9,::]
        Train_X.append(temp_train)
        Train_Y.append(np.array(Train_X_Year2[i+9,9]))

TrainX = np.array(Train_X).astype(np.float32)
TrainY = np.array(Train_Y).astype(np.float32)
print(TrainX.shape)
print(TrainY.shape)
# 最终得到 5628个batch

# print(Train_X_Year2[-1,-1,-3::,::])
## 测试数据是否正确
# print(TrainX[-2:,::,::])
# print(TrainY[-2:])

# TestOK!
## 使用keras开始训练
def Mymodel():
    inputs = layers.Input(shape=(9,17))
    flatten = layers.Flatten()(inputs)
    x1 = layers.Dense(256,activation="relu")(flatten)
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

# test_x = tf.ones(shape=(30,10,17))
# test_y = network.predict(test_x)
# print(test_y.shape)

Early_Stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20,mode='min')
# callbacks=[Early_Stop_callback]
network.compile(optimizer = optimizers.Adam(lr=1e-4),
                loss = tf.keras.losses.MSE,
                metrics = [metrics.mean_squared_error]
                )
network.fit(x = TrainX, y = TrainY ,epochs = 150,batch_size= 64)

##
# network = keras.models.load_model(r'./MyDense4_weights.h5')


## 读入 test.csv
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



##
print(Test_X_Year.shape)
Test_X_Year = np.transpose(Test_X_Year,axes=(0,2,1))
Test_Y = network.predict(Test_X_Year)
print(Test_Y.shape)

##
np.savetxt('output.csv',Test_Y,delimiter=',')

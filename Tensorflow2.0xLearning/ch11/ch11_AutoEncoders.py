# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch11_AutoEncoders
# Description:  完成了一个最简单的自动编码器，使用 fashionmnist 数据集 使用自己还原自己
# Author:       Administrator
# Date:         2021/1/13
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics,Input,Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Auto Encoders 实战
# 定义全局变量
h_dims = 20 # 将原始图片的高纬度特征投影到20维的特征向量上来
BatchSize = 512 # 每个batch
learning_rate = 1e-3 # 学习率设定

def Generate_Dataset_fashionmnist (BatchSize,drop_remainder= False):
    '''
    函数用于自动生成满足tf.keras格式的fashionmnist 衣物图片数据集的 处理完成数据集
    :param BatchSize: 生成的数据集有多少个batchsize
    :param drop_remainder: 是否在batchsize中保留尾数，False表示保留尾数， True表示不保留尾数
    :return: 返回两个 tf.datasets格式的数据集 (train_db,test_db)
    '''
    # 1. 从datasets中引入数据数据
    (X_train,Y_train),(X_test,Y_test) = datasets.fashion_mnist.load_data()
    num_train = Y_train.shape[0] #用于打乱
    # print(X_train[0],Y_train[0])
    # print(type(X_train[0]))
    print(Y_train.max(),Y_train.min())
    # 0,1 区间归一化
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.
    # 空间打散 将28，28 变换为 28*28
    X_train = np.reshape(X_train,newshape= [60000,28*28])
    X_test = np.reshape(X_test,newshape= [10000,28*28])
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    # 2. 将导入的数据集进行处理，
    # 构建数据集 由于这里是进行自编码器的实现，训练集和测试机都是只有X
    db_train = tf.data.Dataset.from_tensor_slices((X_train,X_train))
    db_test = tf.data.Dataset.from_tensor_slices((X_test,X_test))

    # 进行打乱和生成batchsixze
    db_train = db_train.shuffle(num_train).batch(BatchSize)
    db_test = db_test.batch(BatchSize)

    # 3. 测试结果
    sample = iter(db_train)
    sample = next(sample)
    print(sample[0].shape)
    sample = iter(db_test)
    sample = next(sample)
    print(sample[0].shape)

    return db_train,db_test

def AE_Model():

    # Encoder 部分
    inputs = Input(shape = (28*28))
    Encoder_1 = layers.Dense(256,activation= 'relu')(inputs)
    Encoder_2 = layers.Dense(128,activation= 'relu')(Encoder_1)
    Middle = layers.Dense(h_dims)(Encoder_2)

    # Decoder 部分
    Decoder_1 = layers.Dense(128,activation= 'relu')(Middle)
    Decoder_2 = layers.Dense(256,activation= 'relu')(Decoder_1)
    Final = layers.Dense(28*28)(Decoder_2)
    # For test
    # print(Final.shape)

    model = Model(inputs = inputs,outputs = Final)

    return model

# For shape check
# x = tf.ones(shape=[4,28*28])
# model = AE_Model()
# y = model.predict(x)
# Test OK

def evaluation(model):
    x = next(iter(db_test))
    x = x[0] # shape = (512,784)
    x = x[20] # # shape = (1,784)
    x = np.expand_dims(x,axis= 0)
    print(x.shape)
    logits = model.predict(x)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat,[28,28]).numpy()
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(x,[28,28]))
    plt.subplot(1,2,2)
    plt.imshow(x_hat)
    plt.show()
    # plt.savefig(r'C:\Users\Administrator\Desktop\MyGit\test.png')


# Main
db_train,db_test = Generate_Dataset_fashionmnist(BatchSize)
model = AE_Model()
# model.summary()

model.compile(
    optimizer= optimizers.Adam(learning_rate= learning_rate),
    loss= tf.keras.losses.BinaryCrossentropy(from_logits= False),
    metrics = ['accuracy']
)
model.fit(db_train,epochs= 20,validation_data= db_test,validation_freq= 1)

evaluation(model)
























# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W1_HomeWork_Part3
# Description:  本次作业成功使用keras定义了一个lstmrnn循环网络
#               用法和普通的Dense没什么太大区别，但是关键在于如何定义个共享权重的LSTM单元
#               这个单元只有一种权重，每个权重都由多个输入来进行调试
#               最后在采样过程中，也是通过这个训练好的权重进行模型的采样与实现
# Author:       Administrator
# Date:         2021/1/6
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------
import IPython
import sys
from music21 import *
import numpy as np

from C5_W1_HomeWork_Part3_DataSet.grammar import *
from C5_W1_HomeWork_Part3_DataSet.qa import *
from C5_W1_HomeWork_Part3_DataSet.preprocess import *
from C5_W1_HomeWork_Part3_DataSet.music_utils import *
from C5_W1_HomeWork_Part3_DataSet.data_utils import *

from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


##
# 试听训练集中的音频片段
IPython.display.Audio(r'C5_W1_HomeWork_Part3_DataSet/data/30s_seq.mp3')


## 加载数据集
# 我们的音乐系统使用唯一的78个值
X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)
#从输出我们可以得出  和上一课的单词表类似，上一课使用唯一的27位单词表表示当前输入
#音乐使用唯一的78位音符表示当前输入
#每一个音符合集为30个
#总的训练数据为60个




##在这一部分中，你将构建和训练一个音乐学习模型。为此，你将需要构建一个模型，
#这个模型的训练数据为 (m,Tx,78)张量的输入
#这个模型的对应Y 为 (Ty,m,78)的输出
#我们将使用具有64维隐藏状态的LSTM。
n_a = 64

#定义三个层级对象，这个三个层级对象可以以layers的形式进行调用
reshapor = Reshape((1,78)) #兼容拍成向量
LSTM_cell = LSTM(n_a,return_state=True)
densor = Dense(n_values,activation='softmax')

#实现djmodel()，你需要执行2个步骤：

# 创建一个空列表“输出”在每个时间步保存的LSTM单元的输出。
# 循环所有的t时刻

def djmodel(Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """
    #实例化输入X
    X = Input(shape=(Tx,n_values)) #输入为Tx长度的，n_values

    #为最开始的输入 a0 c0 做准备
    a0 = Input(shape=(n_a,),name = 'a0')
    c0 = Input(shape=(n_a,),name = 'c0')
    a = a0
    c = c0

    #创建一个输出 output
    output = []

    for i in range(Tx):
        # 1.选择定第t时刻的音符集x,用于输入LSTM
        x = Lambda(lambda x:X[:,i,:])(X)
        #2.确保输入x的维度 我们使用Reshape操作进行充值维度
        x = reshapor(x)
        #3.运行LSTM的一个步骤，我们得到激活输出和隐藏状态输出
        a,_,c = LSTM_cell(X,initial_state=[a,c])
        #4.将输出的激活值放入全连接层+softmax层进行多分类概率统计
        y_pred= densor(a)
        output.append(y_pred)

    #将以上步骤组合起来
    model = Model(inputs=[X,a0,c0],outputs=output)

    return model

#实例化模型
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
#定义优化器
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

#编译模型
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#将初始化状态定义为0
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

#喂入数据
model.fit([X, a0, c0], list(Y), epochs=100)
#


##
#如上就用keras实现了一个lstmRNN网络的构建
#下面用模型进行预测和采样
#由于推断模型是一个和训练完全不一样的模型
#推断模型需要你将上一时刻的输入作为下一时刻的输出
#因此需要重新构建一个能够为你进行模型预测的全新keras模型

def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """
    x0 = Input(shape=(1,n_values))  #输入仅为一个音符，因此是一个78深度的独热向量

    a0 = Input(shape = (n_a,),name = 'a0')
    c0 = Input(shape=(n_a,),name = 'c0')
    a = a0
    c = c0
    outputs = []
    x = x0

    for i in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        y_sample = densor(a)
        outputs.append(y_sample)
        x = Lambda(one_hot)(y_sample)

    sample_model = Model(inputs = [x0,a0,c0], outputs = outputs)

    return sample_model


#实例化模型
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

#为模型输入初始化向量
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

##实现最后的预测和采样函数
def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """

    #从上面构建的模型中预测得到pred
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])

    #从输出的softmax张量中 根据每一个维度 得到每个位置应该对应的音符
    indices = np.argmax(pred,axis=-1)

    #将对应音符转为输出音乐的对应数字
    results = to_categorical(indices,num_classes=78)

    return results,indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))
##
out_stream = generate_music(inference_model)









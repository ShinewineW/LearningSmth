# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:28:28 2020
@Discription: 本次作业通过keras快速搭建了一个检测笑脸的卷积神经网络模型，免去使用tensorflow
来进行训练，而是用keras封装的更高层次的函数来进行
在训练超参数过程中发现，batchsize调节过低过高都会显著影响结果的准确性

@author: Netfather
@Last Modified data: 2021年1月19日
"""

import numpy as np
import pydot
import keras
#import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
from C4_W2_HomeWork_Part2_DataSet.kt_utils import *
import matplotlib.pyplot as plt
from IPython.display import SVG
from matplotlib.pyplot import imshow

K.set_image_data_format('channels_last')


#%%
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


#观察数据形状可知 是由
#600个  64*64*3的图片  以及对应的标签位 1 0 的数据
#Test 4 images to test
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(X_train[i])
#     if Y_train[i] == [1]:
#         plt.title('happy')
#     elif Y_train[i] == [0]:
#         plt.title('unhappy')
#Test OK!

#%%使用keras快速构建一个简单的卷积网络模型

#如下是一个最简单基础的卷积神经网络模型定义
#一般遵循如下步骤
#1.设定整个模型的入口，使用keras.layers.Input函数来指定
#2.设定模型细节，包括但不限于，卷积，池化，批量归一化等等
#3.全连接层，使用keras.layers.Dense函数来进行全连接层的指定
#4.模型的实例化，为整个模型第一好出口和入口，同时模型名字

# def model(input_shape):
#     #设定张量入口
#     X_input = Input(input_shape)
#     #设定0填充的大小
#     X = ZeroPadding2D((3,3))(X_input)   
    
#     #CONV->BN->RELU 一般这一套我哦们称之为一个标准化流程
#     X = Conv2D(32,(7,7),stride = (1,1),name = 'conv0')(X)
#     X = BatchNormalization(axis = 3, name = 'bn0')(X)
#     X = Activation('relu')
    
#     #卷积一定次数之后 我们使用最大池化
#     X = MaxPooling2D(pool_size=(2,2), strides = (2,2),name = 'max_pool')(X)
    
#     #连接全连接层
#     X = Flatten()(X)
#     X = Dense(1,activation='sigmoid',name = 'fc')(X)
    
#     #完成结构定义之后 定义整个模型
#     model = Model(inputs = X_input,outputs = X,name = 'Happy_Model')
#     return model

#下面使用如上代码来完成笑脸检测
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(1, 1))(X_input)
    X = Conv2D(8, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    
    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(16, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    
    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(32, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    
    # FC
    X = Flatten()(X)
    Y = Dense(1, activation='sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = Y, name='HappyModel')
    ### END CODE HERE ###
    
    return model

#%%完成上述模型函数后，下面开始实例化模型
# 1.通过调用上面的函数创建模型
# 2.通过调用model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])编译模型
# 3.通过调用model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)训练模型
# 4.通过调用model.evaluate(x = ..., y = ...)测试模型

#第一步 创建模型，为模型指定输入张量大小
happyModel = HappyModel((64,64,3))

#第二步 编译模型，为模型指定优化器，loss函数，学习率，以及计算准确率的方法
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

#第三步：喂入数据
happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)

#%%
#第四步 评估模型
preds = happyModel.evaluate(x=X_test, y=Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


#%%使用自己的图像来进行测试
import cv2

img = cv2.imread(r'C4_W2_HomeWork_Part2_DataSet/my_image.jpg') 
#如果不加如下两行 数据会发蓝
b,g,r = cv2.split(img) 
img = cv2.merge([r,g,b])
img = cv2.resize(img,(64,64), interpolation=cv2.INTER_AREA)
plt.imshow(img)
img = img/255.
img = np.expand_dims(img ,axis = 0)

print(happyModel.predict(img))


#完成笑脸检测，这个模型可以很快速的检测你的面部表情是笑脸还是哭脸





    





























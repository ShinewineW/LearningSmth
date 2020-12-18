# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:55:20 2020
@Discription:
    为了使用这份文件，你必须手动执行如下
    pip3 install keras==2.1.0
    过高的keras版本不支持tensorflow1.x
    如下代码实现了一个ResNet50的步骤，比较简化，能够在sign上实现很高的表现
    但是似乎sign数据集的手都是手心照片 并不是手背 所以自我拍摄的图片效果很差？
    
@author: Administrator
"""

import numpy as np
import tensorflow as tf
import pydot
import scipy.misc
import cv2
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

from C4_W2_HomeWork_Part1_DataSet.resnets_utils import *
from IPython.display import SVG
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


#%%
#开始构建残差模块
#分为如下4个步骤
# 主路径的第一部分：

# 第一个CONV2D具有形状为（1,1）和步幅为（1,1）的F1个滤波器。其填充为“valid”，其名称应为conv_name_base + '2a'。使用0作为随机初始化的种子。
# 第一个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2a'。
# 然后应用ReLU激活函数。
# 主路径的第二部分：

# 第二个CONV2D具有形状为(f,f) 的步幅为（1,1）的F2个滤波器。其填充为“same”，其名称应为conv_name_base + '2b'。使用0作为随机初始化的种子。
# 第二个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2b'。
# 然后应用ReLU激活函数。
# 主路径的第三部分：

# 第三个CONV2D具有形状为（1,1）和步幅为（1,1）的F3个滤波器。其填充为“valid”，其名称应为conv_name_base + '2c'。使用0作为随机初始化的种子。
# 第三个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2c'。请注意，此组件中没有ReLU激活函数。
# 最后一步：

# 将shortcut和输入添加在一起。
# 然后应用ReLU激活函数。

#如下定义残差网络中的识别块，这个块要求输入与输出尺寸完全相同
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    #定义每个节点的名字基准
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #从filters字典中重新获得滤波器
    F1,F2,F3 = filters
    
    #转存输入X，将X存储到X_shortcut中，方便在最后和输出相加
    X_shortcut = X
    
    # 实现主要路径的第一步
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #实现主要路径的第二部
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #实现主要路径的第三步
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    #最后一步 合并两个层  然后使用激活
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


# tf.reset_default_graph()

# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float", [3, 4, 4, 6])
#     X = np.random.randn(3, 4, 4, 6)
#     A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#     print("out = " + str(out[0][1][1][0]))
#Test OK!


#下面是卷积块，不同于恒等映射，由于输入输出尺寸不同，这个网络要求
#在跳跃连接中进行维度的匹配

# 主路径的第一部分：

# 第一个CONV2D具有形状为（1,1）和步幅为（s，s）的F1个滤波器。其填充为"valid"，其名称应为conv_name_base + '2a'。
# 第一个BatchNorm标准化通道轴。其名字是bn_name_base + '2a'。
# 然后应用ReLU激活函数。
# 主路径的第二部分：

# 第二个CONV2D具有（f，f）的F2滤波器和（1,1）的步幅。其填充为"same"，并且名称应为conv_name_base + '2b'。
# 第二个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2b'。
# 然后应用ReLU激活函数。
# 主路径的第三部分：

# 第三个CONV2D的F3滤波器为（1,1），步幅为（1,1）。其填充为"valid"，其名称应为conv_name_base + '2c'。
# 第三个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2c'。请注意，此组件中没有ReLU激活函数。
# Shortcut path：

# CONV2D具有形状为（1,1）和步幅为（s，s）的F3个滤波器。其填充为"valid"，其名称应为conv_name_base + '1'。
# BatchNorm标准化通道轴。它的名字应该是bn_name_base + '1'。
# 最后一步：

# 将Shortcut路径和主路径添加在一起。
# 然后应用ReLU激活函数。

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    #定义名字基准
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   
    
    #从滤波器字典中重新得到滤波器
    F1, F2, F3 = filters
    
    #保存X的输入为X_shortcut
    X_shortcut = X
    
    #主路径第一层
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #主路径第二部
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #主路径第三步
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    #跳跃层中的链接
    X_shortcut =  Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut =  BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    
    #合并跳跃层和主路径
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# tf.reset_default_graph()

# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float", [3, 4, 4, 6])
#     X = np.random.randn(3, 4, 4, 6)
#     A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#     print("out = " + str(out[0][1][1][0]))
#Test OK!

#%%接下来使用上面搭建的两个块 来构建第一个ResNet模型 50层
# 零填充填充（3,3）的输入
# 阶段1：
#     - 2D卷积具有64个形状为（7,7）的滤波器，并使用（2,2）步幅，名称是“conv1”。
#     - BatchNorm应用于输入的通道轴。
#     - MaxPooling使用（3,3）窗口和（2,2）步幅。
# 阶段2：
#     - 卷积块使用三组大小为[64,64,256]的滤波器，“f”为3，“s”为1，块为“a”。
#     - 2个标识块使用三组大小为[64,64,256]的滤波器，“f”为3，块为“b”和“c”。
# 阶段3：
#     - 卷积块使用三组大小为[128,128,512]的滤波器，“f”为3，“s”为2，块为“a”。
#     - 3个标识块使用三组大小为[128,128,512]的滤波器，“f”为3，块为“b”，“c”和“d”。
# 阶段4：
#     - 卷积块使用三组大小为[256、256、1024]的滤波器，“f”为3，“s”为2，块为“a”。
#     - 5个标识块使用三组大小为[256、256、1024]的滤波器，“f”为3，块为“b”，“c”，“d”，“e”和“f”。
# 阶段5：
#     - 卷积块使用三组大小为[512、512、2048]的滤波器，“f”为3，“s”为2，块为“a”。
#     - 2个标识块使用三组大小为[256、256、2048]的滤波器，“f”为3，块为“b”和“c”。
# 2D平均池使用形状为（2,2）的窗口，其名称为“avg_pool”。
# Flatten层没有任何超参数或名称。
# 全连接（密集）层使用softmax激活将其输入减少为类数。名字是'fc' + str(classes)。


# GRADED FUNCTION: ResNet50

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    #将输入通过keras API修改为 tensor张量    
    X_input = Input(input_shape)
    
    #处理输入数据
    X = ZeroPadding2D((3,3))(X_input)
    
    #第一步
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    #第二步
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
    #第三步
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    
    #第四步
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    #第五步
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 2048], stage=5, block='b')
    X = identity_block(X, 3, [256, 256, 2048], stage=5, block='c')
    
    #平均池化
    X = AveragePooling2D(pool_size=(2,2))(X)
    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

#%%

model = ResNet50(input_shape = (64, 64, 3), classes = 6)   
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))      

model.fit(X_train, Y_train, epochs = 20, batch_size = 32)  


preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#%%
#保存模型到文件

# print("Saving model to disk \n")
# save_path = r"C4_W2_HomeWork_DataSet/signs_resnet50.h5"
# model.save(save_path)

#%%
#测试自己的图片


image = cv2.imread(r'C4_W2_HomeWork_Part1_DataSet/test_images/one.jpg') 
#如果不加如下两行 数据会发蓝
b,g,r = cv2.split(image) 
image = cv2.merge([r,g,b])
image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
plt.imshow(image)
image = image/255.
image = np.expand_dims(image ,axis = 0)

print(np.argmax(model.predict(image,verbose = 1)))

#实际上在实际拍摄的图片上运行的并不好





    
    


    
    























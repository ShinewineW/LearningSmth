# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:42:46 2020
@Disrcription: 本作业实现了一个很简单的脸部识别代码。通过加载已经训练好的Xception模型
#来返回一个人脸的准确信息。
@author: Administrator
"""

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

K.set_image_data_format('channels_first')

import cv2
import os
import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from C4_W4_HomeWork_Part1_DataSet.fr_utils import *
from C4_W4_HomeWork_Part1_DataSet.inception_blocks_v2 import *

np.set_printoptions(threshold=sys.maxsize)


#%%1.编码人脸图像到128维的特征空间

#这里使用 网络结构遵循Szegedy et al.中的Inception模型。  并使用预训练权重
#该网络使用96x96尺寸的RGB图像作为输入。具体来说，输入一张人脸图像（或一批m人脸图像）
# 作为维度为 (m,3,96,96)的张量。注意到这里的张量是通道优先的
# 输出维度为（m，128）的矩阵，该矩阵将每个输入的面部图像编码为128维向量

#创建已经预训练好的人脸模型
FRmodel = faceRecoModel(input_shape = (3,96,96))


# print("Total Params:", FRmodel.count_params())
#Test OK!
#%%2.编码实现三元损失函数
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    #y_true为keras中一定要的参数，这里不使用
    #y_pred为一个三维的列表，包含A,P,N三张照片
    
    #从输入中复原图片
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    #For Dedug
    # print(anchor.shape,positive.shape,negative.shape)
    
    #计算A与P之间的距离
    pos_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    #计算A与N之间的距离
    neg_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    #得到基本损失
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)#其中alpha表示正向示例应该大于负向示例多少
    #将这个损失加入max函数得到最终损失
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    #注意 以上的代码都没有指定轴，实际上，每一个操作都是在最后一个轴进行的
    #也就是 pos_dist,neg_dist,basic_loss的计算结果是一个[m,]的结果
    #最后的对loss求和会将其变为sclar
    return loss

# with tf.Session() as test:
#     tf.set_random_seed(1)
#     y_true = (None, None, None)
#     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
#               tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
#               tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
#     loss = triplet_loss(y_true, y_pred)
    
#     print("loss = " + str(loss.eval()))
#Test OK!

#%%加载预训练好的模型
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

#%%
#如下代码使用上述模型进行检测
#将每个人的coding录入数据库
database = {}
database["danielle"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/danielle.png", FRmodel)
#%%
print(database["danielle"].shape)


#%%
database["younes"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding(r"C4_W4_HomeWork_Part1_DataSet/images/arnaud.jpg", FRmodel)

#%%
def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path,model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist<0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    ### END CODE HERE ###
        
    return dist, door_open
#%%
verify(r"C4_W4_HomeWork_Part1_DataSet/images/camera_0.jpg", "younes", database, FRmodel)


#%%

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

#%%如上代码实现了一个根据三元损失函数，进行一次前向传播得到loss的代码
#根据loss来判断是否为指定人
























# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch12_Test_flow_from_directory_Funciton
# Description:  This file test the function of flow_from_directory
#                 通过分析输出 我们可以得出结论：
#                 对于这个功能函数，他会流水线不断读入数据 同时next函数并不会告诉你这个epoch是否走完
#                 正确的使用方法应该为
#                 1. 定义好 超参数_batchsize
#                 2. 定义好 超参数 训练图片数量
#                 3. 手动实现1个epoch循环：
#                     如果是要求每个batchsize全部均等的场合 for batch_count in range(int(total_num / _Batchsize) ) 可以很好的满足
#                     如果不要求每个batchsize全部均等  for batch_count in range(int(total_num / _Batchsize) + 1) 可以很好的满足
#                 4. 在实现一个epoch之后， 手动调用一下 .reset 和 .on_epoch_end() 方法 这时候第二轮的照片就会和原来的不一样
#                 5. 按照如上流程即可实现
# Author:       Administrator
# Date:         2021/1/22
# -------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics,Input,Model
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import cv2


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义超参数
_Gen_Dimension = 100
_Image_Shape_InputModel = (64,64)
_Batchsize = 512  # 图片batchsize大小
_Path = r"C:\Users\Administrator\Desktop\AnimeDataset"
_learning_rate = 0.002

def Rescale(*args):
    image = args[0] #从args中fetch每张图片 size = (with,height,3)
    image = image / 255.
    return image


Train_image = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function= Rescale)

#构建一个图片生成器
db_train = Train_image.flow_from_directory(
    directory= _Path,target_size= _Image_Shape_InputModel,color_mode='rgb',classes = None,
    class_mode= None,batch_size= _Batchsize, shuffle=True, seed = 233
)

plt.clf() # 清空图例

for epoch in range(2):
    db_train.reset() # 清空batch_index参数
    db_train.on_epoch_end() # 重新随机乱序
    for batch in range(int(63565 / _Batchsize) + 1):
        batch_inner = db_train.batch_index

        sample = db_train.next()
        print("at batch inner{}   at batch count{}".format(batch_inner,batch))
        # Test the remainder batchsize
        # print(sample.shape)
        if (batch_inner == 0):
            image = sample[0,::,::,::]
            image = np.squeeze(image)
            plt.subplot(1,2,epoch+1)
            plt.imshow(image)
plt.show()


# 通过分析输出 我们可以得出结论：
# 对于这个功能函数，他会流水线不断读入数据 同时next函数并不会告诉你这个epoch是否走完
# 正确的使用方法应该为
# 1. 定义好 超参数_batchsize
# 2. 定义好 超参数 训练图片数量
# 3. 手动实现1个epoch循环：
#     如果是要求每个batchsize全部均等的场合 for batch_count in range(int(total_num / _Batchsize) ) 可以很好的满足
#     如果不要求每个batchsize全部均等  for batch_count in range(int(total_num / _Batchsize) + 1) 可以很好的满足
# 4. 在实现一个epoch之后， 手动调用一下 .reset 和 .on_epoch_end() 方法 这时候第二轮的照片就会和原来的不一样
# 5. 按照如上流程即可实现
"""
(512, 64, 64, 3)
at batch inner124   at batch count124
(77, 64, 64, 3)
at batch inner0   at batch count125

"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:09:53 2020
@Discription: 正确使用tf.data.datasets类别可以帮助你很快搭建一个符合规则的数据类实例化
              正确使用model.fit中的validation_split 可以帮你快速实现k_fold的交叉验证
@author: Administrator
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#%%
(x,y),(x_test,y_test) = datasets.mnist.load_data()

print(x.shape,y.shape)
print(x_test.shape,y_test.shape)

#%%这种时候我们就会考虑将x，y人为再划分两个部分，一部分作为traiin，一部分作为val
#具体操作如下

x_train,x_val = tf.split(x,num_or_size_splits = [50000,10000])
y_train_y_val = tf.split(x,num_or_size_splits = [50000,10000])

#后续操作就和之前cifar10的操作一致  使用from_tensor_slice来构建成成对的datasets

#%%但是有些时候我们可能希望使用k-fold validation 也就是一个动态的k个分割。
#在每次训练前，我们手动分割出不同的train和val 然后放入进行训练和验证

#同样 keras提供相关函数的支持，在network.fit中使用(db_train_val, 然后指定 validation_split = 0.1)
#就可以自动使用动态valdation来进行训练。 注意此时提供的数据为 db_train_val



























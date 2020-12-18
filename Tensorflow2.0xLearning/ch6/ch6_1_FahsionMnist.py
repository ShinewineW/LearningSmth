# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:13:42 2020

@author: Administrator
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt

#%%
(x,y),(x_test,y_test) = datasets.fashion_mnist.load_data()
print(x.shape,y.shape)
print(x_test.shape,y_test.shape)

#%%显示几张照片
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(x[i])
#     plt.title(y[i])
    
x = tf.convert_to_tensor(x,dtype = tf.float32) / 255.
y = tf.convert_to_tensor(y,dtype = tf.int32)

batchsize = 128
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.batch(batchsize).shuffle(10000)

x_test = tf.convert_to_tensor(x,dtype = tf.float32) / 255.
y_test = tf.convert_to_tensor(y,dtype = tf.int32)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.batch(batchsize)
    
db_iter = iter(db)
sample = next(db_iter)

model = Sequential([
    layers.Dense(256,activation = tf.nn.relu),
    layers.Dense(128,activation = tf.nn.relu),
    layers.Dense(64,activation = tf.nn.relu),
    layers.Dense(32,activation = tf.nn.relu),
    layers.Dense(10,activation = tf.nn.relu)
    ])








































#%%
def main():
    
    pass
    
    
    

if __name__=='__main__':
    main()
    
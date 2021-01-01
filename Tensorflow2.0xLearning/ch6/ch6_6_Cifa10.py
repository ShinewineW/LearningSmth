# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:36:35 2020
@Discription:本代码通过深度神经网络实现了一个简单的cifar10数据集的分类预测任务
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
BatchSize = 128

(x,y),(x_val,y_val) = datasets.cifar10.load_data()

y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y,depth = 10)
y_val = tf.one_hot(y_val,depth = 10)

print(x.shape)
print(y.shape)
print(x_val.shape)
print(y_val.shape)

# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(x[i])
#     plt.title(np.squeeze(y[i]))

#%%数据预处理
def preprocess(x,y):
    x = tf.cast(x,dtype = tf.float32) / 255.
    y = tf.cast(y,dtype = tf.int32)
    return (x,y)

train_db = tf.data.Dataset.from_tensor_slices((x,y))  #将x,y一一对应
train_db = train_db.map(preprocess).shuffle(50000).batch(BatchSize)
val_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))  #将x,y一一对应
val_db = val_db.map(preprocess).batch(BatchSize)

# Test OK!
sample = next(iter(val_db))
print(sample[0].shape,sample[1].shape)

#%%构建网络
#在初始化函数中 初始化所有参数
#在所有call函数中，实现连接细节或者操作细节



#1.新建层，你需要从父类Layers中继承初始化和call函数来自定义一个新的层
class MyDense(layers.Layer):
    #自定义的层可操作性更大
    def __init__(self,input_dim,output_dim):
        super(MyDense,self).__init__()
        self.W = tf.Variable(
            initial_value = tf.random.normal([input_dim,output_dim]) * tf.math.sqrt(2./input_dim),
            trainable=True, 
            name = 'w') #类数据W，标志该层的权重。
        #为了区别这个自定义层和传统层的区别，我们这里不再定义bias
        
    def call(self,inputs,training=None,activation='relu'):
        """
        Parameters
        ----------
        inputs : tensor张量
            你需要维护好每个输入这个层的大小形状
        training : TYPE, optional
            DESCRIPTION. The default is None.
        activation : TYPE, optional
            DESCRIPTION. The default is 'relu'.
            'relu':使用relu激活
            'none':不使用任何激活

        Returns
        -------
        x : TYPE
            DESCRIPTION.

        """
        
        x = inputs @ self.W
        if activation == 'relu':
            x = tf.nn.relu(x)
        return x
    

#2.自定义网络,这个网络又层积木构建，这个自定义网络需要阐明这个网络结构的细节
#你需要从父类Model中继承所有需要的函数，并重写初始化和call
class MyNetwork(keras.Model):
    
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.fc1 = MyDense(32*32*3,256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)
    
    def call(self,inputs,training=None):
        """
        Parameters
        ----------
        inputs : [batchsize,32,32,3]
            自指定，记得手动拍扁.
        training : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        x = tf.reshape(inputs,[-1,32*32*3])
        #[batch,32*32*3] -> [batch,256]
        x = self.fc1(x)
        #[batch,256] -> [batch,128]
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x,activation = 'none')#最后一层不需要激活，因此指定为none
        #最终得到 [batch,10]的输出
        return x
    

#%%实例化
#实例化
network = MyNetwork()   
#编译
network.compile(optimizer = optimizers.Adam(lr = 1e-3),
                loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy']
                )
network.fit(train_db,epochs = 20,validation_data = val_db,validation_freq = 1)

#%%模型保存
#一般优先保存网络的权值
network.evaluate(val_db)
# network.save_weights('ckpt_cifar10/MyDense5_weights.ckpt')

#%%
#新建一个一模一样结构的网络，从保存的ckpt中导入权重

network_restore = MyNetwork()
network_restore.compile(optimizer = optimizers.RMSprop(learning_rate = 1e-3),
                        loss = tf.losses.CategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy']
                        )
network_restore.load_weights('ckpt_cifar10/MyDense5_weights.ckpt')
network_restore.evaluate(val_db)#这一步至关重要 你一定要让数据进入
#这个load进来的权重才会被真正使用。


print(network_restore.summary())
   
        
        
    
    
    


















        
        
        
        

    





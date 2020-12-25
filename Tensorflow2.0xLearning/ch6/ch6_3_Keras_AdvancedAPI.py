# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:55:14 2020
@Discription: 本代码讲解了keras高级api接口，使用接口拼接出来的模型model，自带model特性，实例化这些类就可以
非常方便的实现整个网络的训练过程
@author: Administrator
"""
import tensorflow.keras as K
#%%介绍Keras下的 Metrics
#最典型的使用 就是 'accuracy' 这个参数，可以i自动实现分类问题的准确率判定

#第一步 新建一个meter  新建一把标尺
acc_meter = K.Metrics.Accuracy()
loss_meter = K.Metrics.Mean()
#第二步 标尺有了之后 我们向其中添加数据

# loss_meter.update_state(loss)
# acc_meter.update_state(y,pred)

#第三步将数据取出来
#使用  变量.result()方法得到tensor类 然后再跟上 .numpy()得到numpy类

#如果标尺使用结束，需要把缓存中的所有数据清空
#使用 变量.reset_states()方法

#具体利用请看同目录下 ch6_1_FashionMnist.py中的使用

#%%介绍Keras下的 compile编译功能和 fit喂养功能
#Compile 一般指定学习率，优化器，损失和评估指标，也就是上一个cell提到的标尺
#Fit   将数据通过batchsize喂入模型 同时还可以指定traning中间穿插的验证集，来一边训练一边验证模型质量
#Evaluate 通过测试功能来进行测试 这个代码是在模型全部训练完成后，进行一次测试
#Predict 通过预测功能来进行最终模型的预测


























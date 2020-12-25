# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:13:42 2020
@Discription: 本代码实现了一个简单的深度神经网络，对fashion_mnist数据集进行训练和测试，
如果选择batchsize为128  20个epoch后最终结果为88%左右
如果选择batchsize为32   20个epoch后最终结果为88.52%  说明对于深度神经网络，这个结果只能到这里了
@author: Administrator
"""

import tensorflow as tf
# from tensorflow import keras
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

batchsize = 32
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.batch(batchsize).shuffle(60000)

x_test = tf.convert_to_tensor(x_test,dtype = tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test,dtype = tf.int32)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.batch(batchsize)
    
db_iter = iter(db)
sample = next(db_iter)

model = Sequential([
    layers.Dense(256,activation = tf.nn.relu),
    layers.Dense(128,activation = tf.nn.relu),
    layers.Dense(64,activation = tf.nn.relu),
    layers.Dense(32,activation = tf.nn.relu),
    layers.Dense(10)
    ])

model.build(input_shape = [None,28*28])
optimizer = optimizers.Adam(lr = 1e-4)

#实例化两个标尺
acc_meter = metrics.Accuracy()  #准确率标尺
loss_meter = metrics.Mean() #平均损失标尺
#model.summary()
#%%

for epoch in range(5):
    
    for step, (x,y) in enumerate(db):
        x = tf.reshape(x,[-1,28*28])
        
        with tf.GradientTape() as tape:
            #[num,784] -> [num,10]
            logits = model(x)
            y_onehot = tf.one_hot(y,depth = 10)
            # loss = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
            loss2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits = True))
            
            loss_meter.update_state(loss2)
        grads = tape.gradient(loss2,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))      

        if step % 100 == 0:
            print(epoch,step,'loss:',loss_meter.result().numpy())
            #然后清空buffer，这样每次打印出来的就是标准的上100个step的loss平均值
            loss_meter.reset_states()

    #Test
    #法1 使用手动计算accuracy的方法
    # total_correct =0
    # total_num = 0
    # for x,y in db_test:
    #     x = tf.reshape(x,[-1,28*28])
    #     logits = model(x)
        
    #     prob = tf.nn.softmax(logits, axis = 1)
    #     pred = tf.argmax(prob,axis = 1)
    #     pred = tf.cast(pred,dtype = tf.int32)
    #     correct = tf.equal(pred,y)
    #     correct = tf.reduce_sum(tf.cast(correct,dtype = tf.int32))
    #     total_correct += int(correct)
    #     total_num += x.shape[0]
    # print(total_num)
    # acc = total_correct / total_num
    # print('acc:' , acc)
    
    #法2 使用accuracy标尺
        
    for step,(x,y) in enumerate(db_test):
        x = tf.reshape(x,[-1,28*28])
        logits = model(x)
        
        pred = tf.argmax(logits,axis = 1)
        pred = tf.cast(pred,dtype = tf.int32)
        
        acc_meter.update_state(y, pred)
    
    print(step,'Evaluate ACC:', acc_meter.result().numpy())
    acc_meter.reset_states()
    
    
#%%
            






























#%%
def main():
    
    pass
    
    
    

if __name__=='__main__':
    main()
    
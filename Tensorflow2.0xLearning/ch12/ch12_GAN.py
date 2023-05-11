# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch12_GAN
# Description:  传统的GAN loss 所使用公式是从 KL散度中演变过来，KL散度是一个衡量两个分布差别多大的指标，通过最小化或最大化这个指标，我们可以完成对鉴别器和生成器的训练
#                   但是这种GAN有一个严重问题就是训练不稳定，JS散度无法正确衡量 两个完全不重叠的分布 他们的差别到底有多大，分别不出就给不出优化的反向，导致训练非常的困难
#               WGAN ：核心要点  Wasserstein Distance 可以解决传统通过JS散度导致无法正确衡量 两分布完全不重叠的区别 也就是JS散度不能很好的指导两个完全不重叠的分布 进行重合
#               同时使用权重裁剪技术 保证鉴别器网络是一个平缓变化的过程，输出不会剧烈变化
#               注意 这里是权重裁剪！ 不是梯度裁剪，梯度裁剪是为了防止梯度消失或者爆炸，保证每次梯度在既定范围内。

# Author:       Administrator
# Date:         2021/1/20
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics,Input,Model
import matplotlib.pyplot as plt
import os
import numpy as np


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义超参数
_Gen_Dimension = 100
_Image_Shape_InputModel = (64,64)
_Batchsize = 512  # 图片batchsize大小
_Path = r"C:\Users\Administrator\Desktop\AnimeDataset"
_Generator_Save_Path = 'Mygen1.h5'
_learning_rate = 0.002


# tf.keras.applications.Xception(
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
# )

# include_top设定为False标志着不加入全连接层，此时就对输入图像的大小没有要求，但是如果设定为加入全连接层，那么输入大小就固定为(299,299,3) 否则会大小不一致而报错
# 但是设定的大小不能比71还小，否则会出问题
# Xception_model = Xception(include_top = False, weights = "imagenet")
# # test_x = tf.ones(shape = [1,96,96,3])
# # test_y = Xception_model.predict(test_x)
# # print(test_y.shape)
# Xception_model.trainable = False
# # Xception_model.summary()
# Base_model = Model(inputs = Xception_model.input,outputs = Xception_model.get_layer("add_11").output)
# # Base_model.summary()
# test_x = tf.ones(shape = [1,96,96,3])
# test_y = Base_model.predict(test_x)
# print(test_y.shape)

## 1. 构建 Generator网络 用于生成图片，用于欺骗 鉴别器网络

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        # 从 [b,_Gen_Dimension]维度向量  扩展到  [b,64,64,3]
        # 这里 [b,_Gen_Dimension] 可以从已经训练好的Xception网络中获得
        self.fc1 = layers.Dense(3*3*512)
        # 然后reshape到 (batchsize,3,3,512)

        # 表示padding不同时输出lenght的判断
        # if output_padding is None:
        #     if padding == 'valid':
        #         # note the call to `max` below!
        #         length = input_length * stride + max(filter_size - stride, 0)
        #     elif padding == 'full':
        #         length = input_length * stride - (stride + filter_size - 2)
        #     elif padding == 'same':
        #         length = input_length * stride

        self.deconv1 = layers.Conv2DTranspose(256,3,3,'valid')
        # 反卷积到 (batchsize,3*3 + max(3-3) = 9,9,256)
        self.bn1 = layers.BatchNormalization()
        self.deconv2 = layers.Conv2DTranspose(128,5,2,'valid')
        # 反卷积到 (batchsize,9*2 + max(5-2) = 21,21,256)
        self.bn2 = layers.BatchNormalization()
        self.deconv3 = layers.Conv2DTranspose(3,4,3,'valid')
        # 反卷积到 (batchsize,21*3 + max(4-3) = 64,64,3)

    def call(self,inputs,training = None,mask=None):
        x = self.fc1(inputs)
        x = tf.nn.leaky_relu(tf.reshape(x,shape=[-1,3,3,512]))

        x = tf.nn.leaky_relu(self.bn1(self.deconv1(x),training = training))
        x = tf.nn.leaky_relu(self.bn2(self.deconv2(x),training = training))
        x = self.deconv3(x)
        x = tf.nn.tanh(x)
        # shape = [batchsize,64,64,3]
        return x

## 2. 构建 鉴别器网络 用于鉴别 Generator中的图片是否和训练图片很相似
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64,5,3,'valid')
        self.conv2 = layers.Conv2D(128,5,3,'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256,5,3,'valid')
        self.bn3 = layers.BatchNormalization()

        # [b,w,h,3] = [b,-1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)  #对鉴别器网络来说这就是一个1分类问题，因此只输出1 表明是否相似

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x),training = training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        x = self.flatten(x)
        logits = self.fc(x)
        # 输出维度  (batchsize,1)

        return logits


# Test OK!
# Gen1 = Generator()
# Dis1 = Discriminator()
# x = tf.random.normal(shape= [2,64,64,3])
# z = tf.random.normal(shape = [2,1024])
#
# y = Dis1.predict(Gen1.predict(z))
# print(y.shape)

## 加载数据集 这里由于使用的是Gan网络 对于梯度非常敏感，因此我们这里在不适用relu的前体现，输入也设定的更加均衡，不适用[0,1] 而是输入在[-1,1]

# flow_from_directory(
#     directory, target_size=(256, 256), color_mode='rgb', classes=None,
#     class_mode='categorical', batch_size=32, shuffle=True, seed=None,
#     save_to_dir=None, save_prefix='', save_format='png', follow_links=False,
#     subset=None, interpolation='nearest'
# )

# preprocessing_function: function that will be applied on each input.
# The function will run after the image is resized and augmented. The function should take one argument: one image (Numpy tensor with rank 3),
# and should output a Numpy tensor with the same shape.
def Rescale(*args):
    image = args[0] #从args中fetch每张图片 size = (with,height,3)
    image = image / 127.5 - 1
    return image

Train_image = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function= Rescale)

#构建一个图片生成器
db_train = Train_image.flow_from_directory(
    directory= _Path,target_size= _Image_Shape_InputModel,color_mode='rgb',classes = None,
    class_mode= None,batch_size= _Batchsize, shuffle=True, seed = 233
)

# print(db_train)
# 迭代器是一个db_train，其中保存所有加载的图片，由于设定class_mode为none 因此不存在标签类
#返回一个  迭代器，迭代得到满足题意的图片

#Test OK!
# sample = next(iter(db_train))
# # (_Batchsize,64,64,3)
# # 取出来的sample是一个元组对   第一个位置放置每一个batch  第二个位置放置每一个标签
# print(sample.shape)
# test = sample[0,::,::,::]
# # test = np.squeeze(test).astype(np.uint8)
# # print(test.shape)
# print(np.max(test),np.min(test))

# plt.clf()
# plt.imshow(test)
# plt.show()

##
# 鉴别器网络 loss实现

# 计算真的图片的loss
def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf.ones_like(logits)) #给定标签1 因为所有图片都是真的，因此labels给定
    return tf.reduce_mean(loss)
# 计算假的图片的loss
def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf.zeros_like(logits)) #给定标签1 因为所有图片都是真的，因此labels给定
    return tf.reduce_mean(loss)

def d_loss_fn(gen1,dis1,batch_z,batch_x,is_training):
    # 1.将真实图片设置为real
    real_image = batch_x
    # 2.将虚假图片设定为fake
    fake_image = gen1(batch_z,is_training)  # 从gen1中得到一个fake_image
    # 3. 将两种图片放入鉴别器
    fake_logits = dis1(fake_image,is_training)
    real_logits = dis1(real_image,is_training)
    # 4. 计算二者的区别  现在根据图片真假输出loss
    dis1_loss_real = celoss_ones(real_logits)  # 真图片的loss,并进行均值
    dis1_loss_fake = celoss_zeros(fake_logits) # 假图片的loss，并进行均值
    # 5. 合并loss
    dis1_loss = dis1_loss_real + dis1_loss_fake

    return dis1_loss

## 生成器网络loss实现
def g_loss_fn(gen1,dis1,batch_z,is_training):
    #1. 通过batch_z得到一个fake_Image
    fake_image = gen1(batch_z,is_training)
    #2. 将fake_image送入鉴别器网络
    fake_image_logits = dis1(fake_image,is_training)
    #3. 为了能骗过鉴别器，我们是希望这个loss是和1比较接近，这意味着鉴别器人为假图片和真的几乎一样
    gen1_loss = celoss_ones(fake_image_logits)

    return gen1_loss

##
gen1 = Generator()
gen1.build(input_shape=(None,_Gen_Dimension))

dis1 = Discriminator()
dis1.build(input_shape=(None,64,64,3))

gen1_optimizer = tf.optimizers.Adam(learning_rate= _learning_rate,beta_1= 0.5)
dis1_optimizer = tf.optimizers.Adam(learning_rate= _learning_rate,beta_1= 0.5)

for epoch in range(100):
    # plt.clf() # 每 次epoch清空一下图例
    db_train.reset() # 清空batch_index参数
    db_train.on_epoch_end() # 重新随机乱序
    for batch in range(int(63565 / _Batchsize) + 1):

        batch_num = db_train.batch_index
        batch_z = tf.random.uniform([_Batchsize,_Gen_Dimension],minval= -1.,maxval= 1.)
        batch_x = db_train.next() # 真实图片 (_batchsize,64,64,3) # 注意 这里的 db_train 是一个

        # 先训练区分器
        with tf.GradientTape() as tape:
            dis1_loss = d_loss_fn(gen1,dis1,batch_z,batch_x,True)
        grads = tape.gradient(dis1_loss,dis1.trainable_variables)
        dis1_optimizer.apply_gradients(zip(grads,dis1.trainable_variables))

        # 然后训练生成器
        with tf.GradientTape() as tape:
            gen1_loss = g_loss_fn(gen1,dis1,batch_z,True)
        grads = tape.gradient(gen1_loss,gen1.trainable_variables)
        gen1_optimizer.apply_gradients(zip(grads,gen1.trainable_variables))

        if batch_num % 10 == 0:
            print("On {} epoch:".format(epoch),"After {} batch_num:".format(batch_num) , "dis1_loss:", float(dis1_loss), "gen1_loss:", float(gen1_loss))

    # 完成一个epoch后 保存一下图片
    z = tf.random.uniform([20,_Gen_Dimension]) #一个epoch后采样20张图片进行保存
    fake_image = gen1.predict(z) # 输出范围为 -1,1
    fake_image = ((fake_image + 1.0) * 127.5).astype(np.uint8)
    img_path = os.path.join('images','gan-{}.png'.format(epoch))
    plt.figure(figsize=(40, 40), dpi=32)
    for i in range(20):
        plt.subplot(4,5,i+1)
        fake_image_show = np.squeeze(fake_image[i,::,::,::])
        plt.imshow(fake_image_show)
    plt.savefig(img_path)






















































##


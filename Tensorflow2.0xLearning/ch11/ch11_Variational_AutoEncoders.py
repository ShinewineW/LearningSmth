# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch11_Variational_AutoEncoders
# Description:  使用多种方法不同的方法实现了变分自编码器。
#               经验： 对于loss一定要小心处理，超小的loss往往意味着模型哪里出了问题
#                   1. 继承父类构建子类的方法 往往适用于非常复杂的结构，和非常规loss
#                   2. 在keras中，如果遭遇非常规loss，往往手动对模型添加 add_loss来添加loss 对模型添加add_metric来添加输出监视
#                   3. 在keras搭建中，如果某个维度和输入的batchsize有关，而你又不想在Input层中指定batchsize，那么就考虑使用tf.shape()[] 这种形式来完全指定
#                   4. 如果网络模型中有某一层是需要 某种分布，某种固定化但是非常规的操作，优先推荐使用 Lambda隐层方法
# Author:       Administrator
# Date:         2021/1/18
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics,Input,Model
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import cv2

# 和本课时的另一份作业一样，使用fashionmnist数据集来实现一个半监督学习，但是这里使用变分自动编码器
tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义全局变量超参数 batchsize = 32
_BatchSize = 512

# 得到待训练的参数 一共返回4种数据集
def Generate_Dataset_fashionmnist (BatchSize,drop_remainder= False):
    '''
    函数用于自动生成满足tf.keras格式的fashionmnist 衣物图片数据集的 处理完成数据集
    :param BatchSize: 生成的数据集有多少个batchsize
    :param drop_remainder: 是否在batchsize中保留尾数，False表示保留尾数， True表示不保留尾数
    :return: 返回前两个是不使用tf.datasets格式的数据 大小为(m,28*28) 其中m为数据总量，这两个是给 more flexible 的情况下，不需要在Input层指定batchsize的时候使用
                后两个 tf.datasets格式的数据集，返回的是一个列表，列表中每个元素是(_BatchSize,28*28)， 在非常严格，需要你在Input层就指定batchsize，全局都使用batchsize的时候使用
    '''
    # 1. 从datasets中引入数据数据
    (X_train,Y_train),(X_test,Y_test) = datasets.fashion_mnist.load_data()
    num_train = Y_train.shape[0] #用于打乱
    # 由于是半监督学习，这里的Y_train 和 Y_test都是无用的
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.
    # 空间打散 将28，28 变换为 28*28
    X_train = np.reshape(X_train,newshape= [60000,28*28])
    X_test = np.reshape(X_test,newshape= [10000,28*28])

    # 2. 将导入的数据集进行处理，
    # 构建数据集 由于这里是进行自编码器的实现，训练集和测试机都是只有X
    db_train = tf.data.Dataset.from_tensor_slices(X_train)
    db_test = tf.data.Dataset.from_tensor_slices(X_test)

    # 进行打乱和生成batchsixze
    db_train = db_train.shuffle(num_train).batch(BatchSize,drop_remainder = drop_remainder)
    db_test = db_test.batch(BatchSize,drop_remainder = drop_remainder)

    return X_train,X_test,db_train,db_test

X_train,X_test,db_train,db_test = Generate_Dataset_fashionmnist(_BatchSize,drop_remainder=True)

## 下面分两种方法搭建VAE变分自编码器
# 使用最朴素的tf.GradientTape，构建一个继承自Model的子类，通过子类的实例化来完成模型的搭建
# 适用于非常复杂的模型结构和非常规的loss构造（例如此题中的情况
# ！！！！！ 如下非常重要！！！！！！！！！
# 如果你在训练过程中，发现输出的某一项loss 低到非常低的值，往往说明网络的loss定义出了问题
# 在一开始的过程中，我由于发现 kl_divergence非常小，就直接使用 tf.reduce_sum函数，我的设想是： 由于这个loss非常小，就使用sum来对这个loss进行一个更大权重的调整
# 但是这么做导致了非常严重的问题，就是模型在kl散度上用力过猛，导致无论什么输入，最终都是一个正太化的完全一致的输出，具体后果查看 /way2_test/epoch20Sample.jpg 与 epoch20predict.jpg
# 所以这里的对于kl散度的 loss 一定要使用 tf.reduce_mean!!!!!

# 第一种朴素方法的结果 存储于 way1_test中
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

def evaluation_way1_predict(vae_model_1,epoch):
    path_work = os.getcwd()
    output_dir = os.path.join(path_work, "way1_test")
    file_name = "Test_evaluation_method_epoch"+str(epoch)+"predict.jpg"
    final_output_dir = os.path.join(output_dir,file_name)
    x_batch = next(iter(db_test))  # (_BatchSize,28*28)
    y_predict,_,_ = vae_model_1.predict(x_batch,batch_size = _BatchSize)
    x_show = np.reshape(x_batch,(_BatchSize,28,28))
    y_show = np.reshape(y_predict,(_BatchSize,28,28))
    plt.clf()  # 清空当前图
    plt.figure(figsize=(40,40),dpi=32)
    for count,index in enumerate(range(40)):
        if(count % 2 == 0):
            plt.subplot(10,4,index+1)
            plt.imshow(np.squeeze(x_show[index,::,::]),'gray')
        else:
            plt.subplot(10, 4, index + 1)
            plt.imshow(np.squeeze(y_show[index-1, ::, ::]),'gray')
    plt.savefig(final_output_dir)

# 法1： 使用最基础的tf语法来完成训练，只需要使用一个函数，从头到尾执行就可以
# 这里为了便于训练，将网络参数调整小
class VAE_1(Model):
    def __init__(self):
        super(VAE_1, self).__init__()
        self.encoder1 = layers.Dense(128,activation='relu')
        self.middle_mean = layers.Dense(10)
        self.middle_log_var = layers.Dense(10)
        self.decoder1 = layers.Dense(128,activation='relu')
        self.finaloutput = layers.Dense(28*28,activation='sigmoid')

    def Encoder(self,inputs):
        x = self.encoder1(inputs)
        middle_mean = self.middle_mean(x)
        middle_log_var = self.middle_log_var(x)

        return middle_mean,middle_log_var

    def Sample(self,z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def Decoder(self,Samle_z):
        decoder1 = self.decoder1(Samle_z)
        outputs = self.finaloutput(decoder1)

        return outputs


    def call(self, inputs, training=None):
        mu, log_var = self.Encoder(inputs)
        # reparameterization trick
        z = self.Sample(mu, log_var)

        x_hat = self.Decoder(z)

        return x_hat, mu, log_var

# 实例化模型

# model = VAE_1()
# model.build(input_shape=(None,784))
# # 定义优化器
# optimizer = tf.optimizers.Adam(1e-3)
# # vae_model_1.summary()
# for epoch in range(10):
#
#     for step, x in enumerate(db_train):
#
#         with tf.GradientTape() as tape:
#             # 完成前向传播
#             x_hat,mean,log_var = model(x)
#             # 完成loss定义
#             rec_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum',name='binary_crossentropy')(x,x_hat)
#             # 计算mean ， log_var的Kl散度，用于表示这二者和标准0，1正太分布间究竟差了多少
#             # https: // stats.stackexchange.com / questions / 7440 / kl - divergence - between - two - univariate - gaussians
#             kl_div = -0.5 * (log_var + 1 - mean**2 - tf.exp(log_var))
#             # kl_div_loss = tf.reduce_sum(kl_div)# 找到问题 reduce_sum会导致完全错误的结果
#             kl_div_loss = tf.reduce_mean(kl_div)
#             total_loss = rec_loss + kl_div_loss
#
#         grads = tape.gradient(total_loss,model.trainable_variables)
#         optimizer.apply_gradients(zip(grads,model.trainable_variables))
#
#
#         if step % 1000 == 0:
#             print("On {} epoch, At {} step, kl_div_loss:{}, binary_loss:{}".format(epoch,step,kl_div_loss,rec_loss))
    #每完成一轮训练，我们输出几张示例图用于检查效果
    # evaluation_way1_predict(model, epoch)

## 法2 使用keras 实现自变分编码器，使用add_loss方法，对网络进行维护。 这种方法简单易操作，而且由keras自动进行维护
# 但是存在一个问题，在编写程序中发现，中间的 eps = tf.random.normal(z_log_var.shape) 代码，如果不在开始的inputs中指定batch_size会报错，原因在于不能使用未指定的tensor.shape作为shape的实例化对象
# 因此在这里，包括整个代码 都是使用 tf.shape 然后固定维度 来直接对shape进行使用，这样这个网络的参数就只会在数据进入的时候进行实例化
# 我的解决方法  第一种： 使用嵌入网络中的结构来对eps层进行初始化 前提使用 eps = tf.random.normal((tf.shape(z_mean)[0],tf.shape(z_mean)[1])) 这种方式来指定维度
#             第二种： 使用Lambda隐藏层，来对层中的每一个参数进行遍历，查看结果 这种方法也可以。 而且这种方法是非常推荐的，如果某个层需要多个其他维度的张量数据进行操作，往往使用Lamda隐层来实现功能
#             第三种： 自己重新定义一个新层，这个层继承自layers 这种方法也可以 但是限制很严重  必须指定batchsize

# 总结： 自定义新层的方法 必须使用标准的tf.dataset生成数据集才可以
# 如果你使用 .shape参数来进行赋值 例如方法2中 使用shape=tensor.shape, initializer="random_normal", trainable=False 是会报错的，因为第一维度不确定
# Cannot convert a partially known TensorShape to a Tensor: (None, 2)

#

class Sample(layers.Layer):
    def __init__(self,tensor):
        super(Sample, self).__init__()
        # 初始化过程我们需要按照 输入的tensor大小来初始化一个normal
        self.eps = self.add_weight(
            shape=tensor.shape, initializer="random_normal", trainable=False
        )
    def __call__(self, *args, **kwargs):
        z_mean,z_log_var = args
        return z_mean + tf.exp(0.5 * z_log_var) * self.eps

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def VAE_2():
    #,batch_size= _BatchSize
    # inputs = Input(shape=(28*28), name='encoder_input',batch_size= _BatchSize) # 使用方法3时 指定batch_size
    inputs = Input(shape=(28 * 28), name='encoder_input')
    x = layers.Dense(128, activation='relu')(inputs)
    z_mean = layers.Dense(2, name='z_mean')(x)
    z_log_var = layers.Dense(2, name='z_log_var')(x)

    # 方法1： 直接把采样嵌入到模型中！
        # 1. 设定一个正太分布
    eps = tf.random.normal((tf.shape(z_mean)[0],tf.shape(z_mean)[1]))
        # 2. 获得标准方差
    std = tf.exp(z_log_var)
        # 3. 通过元素乘法进行采样
    Sample_Z = layers.Add()([z_mean, layers.Multiply()([eps, std])])

    # 方法2： 使用匿名函数Lambda配合sampling函数对层中每一个元素都进行操作
    # Sample_Z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # 方法3： 自定义子类： 抽样层，但是此法和嵌入模型中没有区别，注意 使用此法 需要在两个Input函数中指定 batchsize = _BatchSize
    # Sample_Z = Sample(z_log_var)(z_mean,z_log_var)


    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, Sample_Z], name='encoder')
    # encoder.summary()

    # build decoder model
    # latent_inputs = Input(shape=(2), name='z_sampling',batch_size= _BatchSize) # 使用方法3时指定batch_size
    latent_inputs = Input(shape=(2), name='z_sampling')
    x = layers.Dense(128, activation='relu')(latent_inputs)
    outputs = layers.Dense(28*28, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    # decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs = inputs, outputs = outputs, name='vae_mlp')

    # 加入loss
    reconstruction_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum',name='binary_crossentropy')(inputs, outputs)

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss =-0.5 *  tf.reduce_mean(kl_loss)
    # 如果这里的 kl_loss =-0.5 *  tf.reduce_sum(kl_loss) 那么就会发生和第一个里面一样的错误
    vae.add_loss(kl_loss)
    vae.add_metric(kl_loss, name='kl_loss',aggregation='mean')
    vae.add_loss(reconstruction_loss)
    vae.add_metric(reconstruction_loss, name='mse_loss',aggregation='mean')

    return vae,encoder,decoder

def evaluation_way2_predict(vae_model_2,epoch):
    path_work = os.getcwd()
    output_dir = os.path.join(path_work, "way2_test")
    file_name = "epoch"+str(epoch)+"predict.jpg"
    final_output_dir = os.path.join(output_dir,file_name)
    x_batch = next(iter(db_test))  # (_BatchSize,28*28)
    y_predict = vae_model_2.predict(x_batch,batch_size = _BatchSize)
    x_show = np.reshape(x_batch,(_BatchSize,28,28))
    y_show = np.reshape(y_predict,(_BatchSize,28,28))
    plt.clf()  # 清空当前图
    plt.figure(figsize=(40,40),dpi=32)
    for count,index in enumerate(range(40)):
        if(count % 2 == 0):
            plt.subplot(10,4,index+1)
            plt.imshow(np.squeeze(x_show[index,::,::]),'gray')
        else:
            plt.subplot(10, 4, index + 1)
            plt.imshow(np.squeeze(y_show[index-1, ::, ::]),'gray')
    plt.savefig(final_output_dir)

def evaluation_way2_Sample(decoder,epoch):
    path_work = os.getcwd()
    output_dir = os.path.join(path_work, "way2_test")
    file_name = "epoch"+str(epoch)+"Sample.jpg"
    final_output_dir = os.path.join(output_dir,file_name)
    sample_z = tf.random.normal(shape=(_BatchSize,2))
    y_predict = decoder.predict(sample_z,batch_size = _BatchSize)
    y_show = tf.reshape(y_predict,(-1,28,28))
    plt.clf()  # 清空当前图
    plt.figure(figsize=(40,40),dpi=32)
    for index in range(40):
        plt.subplot(10, 4, index + 1)
        plt.imshow(np.squeeze(y_show[index, ::, ::]),'gray')
    plt.savefig(final_output_dir)

vae_model_2,encoder,decoder = VAE_2()
#
# # 测试模型输出 TEST OK!
# x = tf.ones(shape=[_BatchSize,28*28])
# y = vae_model_2.predict(x)
# print(y.shape)

vae_model_2.compile(optimizer='adam')
# vae_model_2.summary()

# train the autoencoder
# vae_model_2.fit(x= db_train,epochs=50) # 使用方法3时，必须使用db_train 来保证所有待训练全部处于同一个batchsize
vae_model_2.fit(x= X_train,epochs=50,batch_size= 128)
# ,batch_size= 512

evaluation_way2_predict(vae_model_2,6)
evaluation_way2_Sample(decoder,6)





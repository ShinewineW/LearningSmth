# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:10:25 2020
@Discription: 本视频构建了一个神经风格迁移的网络，具体的loss定义已经完全弄明白，但是关于计算图的构建
还是需要大量的编程练习才能弄清楚如果使用tf1.x来按需构建计算图
@author: Netfather
@Last Modified data: 2021年1月19日
"""

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from cv2 import imread
import numpy as np
import tensorflow as tf

from C4_W4_HomeWork_Part2_DataSet.nst_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%预加载已经训练好的VGG-19模型
#此模型已经在imgnet上进行了训练，并取得了良好效果

# model = load_vgg_model(r"C4_W4_HomeWork_Part2_DataSet\pretrained-model\imagenet-vgg-verydeep-19.mat")
# print(model)

#查看nst_utils中的源代码可知，这模型存储与python字典中，方便我们访问随机深度


# 该模型存储在python字典中，其中每个变量名称都是键，而对应的值是包含该变量值的张量。
# 要通过此网络测试图像，只需要将图像提供给模型。在TensorFlow中，你可以使用 
# tf.assign函数执行此操作。特别地，你将使用如下的assign函数：
# model["input"].assign(image)
# 这会将图像分配为模型的输入。此后，如果要访问特定层的激活函数，例如当网络在此图
# 像上运行时说4_2 层，则可以在正确的张量conv4_2上运行TensorFlow会话，如下所示：
# sess.run(model["conv4_2"])

#%% 1.损失函数的构建
#我们先查看一下内容图像C

# content_image = imread(r"C4_W4_HomeWork_Part2_DataSet/images/louvre.jpg",-1)
# print(content_image.shape)
# # B,G,R = content_image[::,::,0],content_image[::,::,1],content_image[::,::,2]
# content_image =content_image[::,::,[2,1,0]]
# imshow(content_image)


#下面按照课程中所述，进行J_Content的构建
#具体定义请看课程或者我写的笔记

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    #将输入数据的必要参数进行还原
    
    _,n_H,n_W,n_C = a_C.shape
    
    #计算激活图之间的距离
    #注意使用squeeze将第0个维度的1给消除
    temp = tf.squeeze(tf.square(tf.subtract(a_C, a_G)),axis = 0)
    print(temp.shape)
    sum_temp = tf.reduce_sum(temp)/ tf.cast(4*n_H*n_W*n_C,dtype = tf.float32)
      
    return sum_temp


#Test OK!
# tf.reset_default_graph()

# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_content = compute_content_cost(a_C, a_G)
#     print("J_content = " + str(J_content.eval()))

#%%2.如下步骤进行J_Style的构建
#我们先常看一下风格图像
# style_image = imread(r"C4_W4_HomeWork_Part2_DataSet/images/monet_800600.jpg",-1)
# print(style_image.shape)
# style_image = style_image[::,::,[2,1,0]]
# imshow(style_image)

#%%
#具体相关 请查看网页或者笔记。这里的Gram矩阵并不是线性代数中的
#而是将原图像的nw，nh挤压成一维，然后和自己的转置做点积成的一个矩阵
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A,tf.transpose(A))
    ### END CODE HERE ###

    
    return GA

# tf.reset_default_graph()

# with tf.Session() as test:
#     tf.set_random_seed(1)
#     A = tf.random_normal([3, 2*1], mean=1, stddev=4)
#     GA = gram_matrix(A)
    
#     print("GA = " + str(GA.eval()))
#Test OK！

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    #1.从输入图像中得到必要数据
    _,n_H,n_W,n_C = a_S.shape
    
    #先消除数量为1的维度，然后transpose进行维度交换，然后reshape后送入matrix
    a_S_For_Gram = tf.reshape(tf.transpose(tf.squeeze(a_S),perm = [2,1,0]),(n_C,n_H*n_W))
    a_G_For_Gram = tf.reshape(tf.transpose(tf.squeeze(a_G),perm = [2,1,0]),(n_C,n_H*n_W))
    
    # print(a_S_For_Gram.eval())
    # print(a_G_For_Gram.eval())
    
    a_S_Gram = gram_matrix(a_S_For_Gram)
    a_G_Gram = gram_matrix(a_G_For_Gram)

    
    J_style = tf.reduce_sum(tf.square(tf.subtract(a_S_Gram, a_G_Gram))/tf.cast((4*n_H*n_W*n_H*n_W*n_C*n_C),dtype = tf.float32))
    
    return J_style

# def compute_layer_style_cost1(a_S, a_G):
#     """
#     Arguments:
#     a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
#     a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
#     Returns: 
#     J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
#     """
    
#      ### START CODE HERE ###
#     # Retrieve dimensions from a_G (≈1 line)
#     m, n_H, n_W, n_C = a_G.get_shape().as_list()

#     # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
#     a_S = tf.reshape(a_S,shape=(n_H* n_W,n_C))
#     a_G = tf.reshape(a_G,shape=(n_H* n_W,n_C))
#     # print(tf.transpose(a_S).eval())
#     # print(tf.transpose(a_G).eval())

#     # Computing gram_matrices for both images S and G (≈2 lines)
#     GS = gram_matrix(tf.transpose(a_S))
#     GG = gram_matrix(tf.transpose(a_G))

#     # Computing the loss (≈1 line)
#     J_style_layer =tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*(n_C*n_C)*(n_W * n_H) * (n_W * n_H))

#     ### END CODE HERE ###
    
#     return J_style_layer


#Test OK!
# tf.reset_default_graph()

# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_style_layer = compute_layer_style_cost(a_S, a_G)
    
#     print("J_style_layer = " + str(J_style_layer.eval()))    
    

#当前定义 的是从一层的激活图中得到风格矩阵并计算损失，但是往往一层并不能代表一张图的风格
#在实际操作中，我们一般对所有层都使用这个计算，然后通过权重分配得到一个总的loss
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


#%%融合上述两种损失

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    ### START CODE HERE ### (≈1 line)
    J = alpha*J_content+beta*J_style
    ### END CODE HERE ###

    
    return J

#%%最后创建组合一起实现风格迁移
# 创建一个交互式会话
# 加载内容图像
# 加载风格图像
# 随机初始化要生成的图像
# 加载VGG16模型
# 构建TensorFlow计算图：
#      通过VGG16模型运行内容图像并计算内容损失
#      通过VGG16模型运行风格图像并计算风格损失
#      计算总损失
#      定义优化器和学习率
# 初始化TensorFlow图，并运行大量迭代，然后在每个步骤更新生成的图像。

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()
#交互式会话  交互式会话将启动自身作为默认会话以构建计算图。这使你可以运行变量而无需经常引用会话对象，从而简化了代码。

content_image = imread(r"C4_W4_HomeWork_Part2_DataSet/images/louvre_small.jpg")
content_image =content_image[::,::,[2,1,0]]
content_image = reshape_and_normalize_image(content_image)

style_image = imread(r"C4_W4_HomeWork_Part2_DataSet/images/monet.jpg")
style_image = style_image[::,::,[2,1,0]]
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model(r"C4_W4_HomeWork_Part2_DataSet\pretrained-model\imagenet-vgg-verydeep-19.mat")

#计算得到J_Content

#将内容图片喂入VGGnET
sess.run(model['input'].assign(content_image))

#我们指定将conv4_2的输出作为a_C
out = model['conv4_2']

#通过计算图运行得到a_C
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
#这里的out是为了后续，后续我们会把G图像送入，当G图像送入时，a_G就是out的输出
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, alpha = 10, beta = 40)

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    generated_image=sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image(r"C4_W4_HomeWork_Part2_DataSet/output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image(r'C4_W4_HomeWork_Part2_DataSet/output/generated_image.jpg', generated_image)
    
    return generated_image

#%%
model_nn(sess, generated_image)



    
    

















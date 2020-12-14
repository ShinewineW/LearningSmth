# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:38:11 2020
@Discription: 本次作业完成了一个完全的卷积神经网络，使用tf1.x框架
整体流程已经掌握，具体问题在于最后使用已经训练好的参数来可视化一张真实
图片发生了问题。具体如何解决看看通过后续学习能否解决。
@author: Administrator
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.python.framework import ops
from C4_W1_HomeWork_DataSet.cnn_utils import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1)

#加载数据集
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print(X_train_orig.shape)
print(Y_train_orig.shape)
print(X_test_orig.shape)
print(classes)

#根据shape输出可以得到训练集是1000个 64*64*3的图片，对应标签是数字
#显示几张图片查看
# index = 4
# for i in range(index):
#     plt.subplot(2, 2, i+1)
#     plt.axis(False)
#     plt.imshow(X_train_orig[i])
#     plt.title(np.squeeze(Y_train_orig[:,i]))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

#%%下面开始用tf1.x构建卷积神经网络
#1.针对网络分别初始化可训练参数和喂养参数
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    #用None来预先不给定参数，方便之后使用mini_batch
    #给训练参数构建占位符
    X = tf.placeholder(tf.float32,(None,n_H0,n_W0,n_C0))
    #给输出Y构建占位符
    Y = tf.placeholder(tf.float32,(None,n_y))
    
    return X,Y

#Test OK
# X, Y = create_placeholders(64, 64, 3, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))


#你无需担心偏差变量，因为TensorFlow函数可以处理偏差。还要注意你只会为conv2d函数初始化权重/滤波器，
#TensorFlow将自动初始化全连接部分的层。
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    tf.set_random_seed(1) 
    #初始化W1 是一个4*4*3 的8个卷积核
    W1 = tf.get_variable("W1",shape = (4,4,3,8),initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2",shape = (2,2,8,16),initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {
        'W1' : W1,
        'W2': W2
        }
    
    return parameters

# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
#Test OK!

#2.完成参数初始化之后，开始正向传播
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    #从输入中还原必要参数
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)    
    MP1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(MP1, W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    MP2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    FC1 = tf.contrib.layers.flatten(MP2)
    Z3 = tf.contrib.layers.fully_connected(FC1, num_outputs = 6, activation_fn=None)
    #注意到这里没有使用激活函数，因此这里Z3是一个logits，在后续cost计算中需要注意这点
    
    return Z3


# tf.reset_default_graph()
# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print("Z3 = " + str(a))
#Test OK!

#3. 正向传播完成 开始计算损失
def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    J_cost = tf.reduce_mean(cost)
    
    return J_cost


# tf.reset_default_graph()

# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
#     print("cost = " + str(a))
#Test OK！

#4.由于后续的反向传播和参数更新，tf会自动帮我们实现，因此这里直接开始模型构建
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    
    #实现反向传播和梯度更新
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init =  tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        for epoch in range(num_epochs):
            
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:  
                (minibatch_X, minibatch_Y) = minibatch
                
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
            
             # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
     
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
    return train_accuracy, test_accuracy, parameters


_, _, parameters = model(X_train, Y_train, X_test, Y_test)    


# #%%
# print(parameters)


# #%%使用上述模型返回的参数来进行提供图片识别
# image = cv2.imread(r'C4_W1_HomeWork_DataSet/thumbs_up.jpg')
# image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
# image = image/255.
# image = np.expand_dims(image ,axis = 0)

# x = tf.placeholder(dtype = tf.float32,shape = (1,64,64,3))
# print(image.shape)

# W1 = tf.convert_to_tensor(parameters["W1"])
# W2 = tf.convert_to_tensor(parameters["W2"])
# print(W1)
# print(W2)

# #%%

# params = {"W1": W1,
#           "W2": W2}

# Z3  = forward_propagation(x, params)
# a = tf.argmax(Z3)

# with tf.Session() as sess:
#     prediction = sess.run(a, feed_dict = {x: image})
    
# print(prediction)

#Test failed! 可能需要后续的学习才能明白如何在tf1.x中测试自己的图像

    
    

























    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:51:57 2020
@Discription: 这里使用Part1作业中构建的L层深度神经网络来对 猫分类器 进行训练
@author: Netfather
@Last Modified data: 2021年1月19日
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2

from C1_W4_HomeWork_DataSet.dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#1.数据集 使用深度神经网络来对C1_W2_Homework中的逻辑回归图像分类问题进行作业
#数据集内容如下
# - 标记为cat（1）和非cat（0）图像的训练集m_train
# - 标记为cat或non-cat图像的测试集m_test
# - 每个图像的维度都为（num_px，num_px，3），其中3表示3个通道（RGB）。

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#Test 数据集是否被读入
# index = 7
# plt.imshow(train_x_orig[index])
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
#Test OK!

print(train_x_orig.shape)
print(train_y.shape)

train_number = train_x_orig.shape[0]
num_px  = train_x_orig.shape[1]
num_py = train_x_orig.shape[2]
channel = train_x_orig.shape[-1]

test_number = test_x_orig.shape[0]

#如下代码拍平X向量为 numpx*numpy*channel,train_number的向量
#注意！ 这里只能用.T 否则数据结构被破坏 报错！
train_x = train_x_orig.reshape(train_number,-1).T
test_x = test_x_orig.reshape(test_number,-1).T

#归一化输入矩阵
train_x = train_x / 255
test_x = test_x / 255

#Test
# print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))

#%%下面开始模型搭建
#一般构建深度神经网络遵循如下步骤
#     1.初始化参数/定义超参数
#     2.循环num_iterations次：
#         a. 正向传播
#         b. 计算损失函数
#         C. 反向传播
#         d. 更新参数（使用参数和反向传播的梯度）
#     4.使用训练好的参数来预测标签

#1.两层神经网络模型构建
#定义好参数的输入，hidden，output
n_x = num_px*num_py*channel
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    #1.初始化参数
    costs = [] #
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    #parameters = initialize_parameters_deep(layers_dims)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    #2.迭代计算前向传播
    for i in range(num_iterations):
        #1.计算Lineaar->Relu
        A_h,cache_h = linear_activation_forward(X, W1, b1, activation = "relu")
        #2.计算Linear->sigmoid
        A_Out,cache_Out = linear_activation_forward(A_h, W2, b2, activation = "sigmoid")
        #3.计算损失
        J_Cost = compute_cost(A_Out, Y)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(J_Cost)))
        if print_cost and i % 100 == 0:
            costs.append(J_Cost)
        #4.反向传播计算
        dAL = - (np.divide(Y, A_Out) - np.divide(1 - Y, 1 - A_Out))
        dA_prev,dw,db = linear_activation_backward(dAL, cache_Out, activation = "sigmoid")
        grads = {
            "dW2" : dw,
            "db2" : db
            }
        dA_prev,dw,db = linear_activation_backward(dA_prev, cache_h, activation = "relu")
        grads.setdefault("dW1", dw)
        grads.setdefault("db1", db)
        # print(dw.shape,db.shape)
        # print(grads["dW1"].shape)
        # grads = {
        #     "dW1" : dw,
        #     "db1" : db
        #     }
        #5.更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
    
    #如下代码展示loss下降曲线
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()    
    
    return parameters


parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

#%%
# predictions_train = predict(train_x, train_y, parameters)

# predictions_test = predict(test_x, test_y, parameters)

#%%

#2. 搭建L层的神经网络
#设置常量 一个5层的深度神经网络
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    #1.初始化参数
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        
        #1.正向传播
        AL,caches = L_model_forward(X, parameters)
        #2.计算损失
        J_Cost = compute_cost(AL, Y)
        #3.逆向传播
        grads = L_model_backward(AL, Y, caches)
        #更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
         # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, J_Cost))
        if print_cost and i % 100 == 0:
            costs.append(J_Cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters  

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
        
 #%%
pred_train = predict(train_x, train_y, parameters)   
pred_test = predict(test_x, test_y, parameters)

#%%
print_mislabeled_images(classes, test_x, test_y, pred_test)
plt.savefig(r'C1_W4_HomeWork_DataSet/test_Wrong.png', dpi=80)

#%%
## START CODE HERE ##
my_image = r"C1_W4_HomeWork_DataSet/my_cat_test1.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = my_image
image = np.array(cv2.imread(fname))
my_image = np.array(cv2.resize(image,(num_px,num_px))).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

####至此  C1_W4_HomeWork全部完成 已经完成构建一个深度神经网络的所有函数






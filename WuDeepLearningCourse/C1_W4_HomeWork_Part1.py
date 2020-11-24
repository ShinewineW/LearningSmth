# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:41:09 2020
#构建一个任意深度的深度神经网络
@author: Administrator
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from C1_W4_HomeWork_DataSet.testCases_v2 import *
from C1_W4_HomeWork_DataSet.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#%% 1构建初始化参数函数
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    #初始化随机数种子
    np.random.seed(1)
    #所有的W参数只由输入特征数和下一层的输出特征数有关系
    W1 = np.random.randn(n_h,n_x)*0.01
    W2 = np.random.randn(n_y,n_h)*0.01
    #所有的b参数只有一个维度，然后数目和特征数保持一致
    b1 = np.zeros(n_h).reshape(-1,1)
    b2 = np.zeros(n_y).reshape(-1,1)
    #组合输出词典 parameters
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {
        "W1" : W1,
        "W2" : W2,
        "b1" : b1,
        "b2" : b2  
        }
    return parameters

#Test 初始化函数
# parameters = initialize_parameters(2,2,1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#Test OK!

#2构建L层的深度神经网络初始化函数
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    #这里的layer_dims是一个列表，导入的是从输入个数开始一路到最后输出特征数的所有信息
    np.random.seed(3)
    parameters = {}
    
    L = len(layer_dims)
    
    for i in range(1,L):
        parameters["W"+str(i)]= np.random.randn(layer_dims[i],layer_dims[i-1])
        parameters["b"+str(i)]  = np.zeros(layer_dims[i]).reshape(-1,1)
    
    assert(parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
    assert(parameters['b' + str(i)].shape == (layer_dims[i], 1))
        
    return parameters

#Test 初始化L层的深度神经网络
# parameters = initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#Test OK!

#3.构建对单一网络层执行的正向传播模块
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A) + b
    
    assert(W.shape[-1] == A.shape[0])
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    cache = (A, W, b)
    return Z,cache

#Test 前向线性模块
# A, W, b = linear_forward_test_case()

# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))
#Test OK!

#4. 构建正向激活函数
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z_liner,linear_cache = linear_forward(A_prev,W,b)
    
    if activation == "sigmoid":
        A,activation_cache = sigmoid(Z_liner)
    elif activation == "relu":
        A,activation_cache = relu(Z_liner)
    
    cache = (linear_cache, activation_cache)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    return A,cache

#Test 线性激活模块
# A_prev, W, b = linear_activation_forward_test_case()

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))
#Test OK!

#5. 通过如上两个模块搭建一个 
# [Linear->Relu] (重复 L-1 次) -> [Linear->sigmoid] 模型

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    #这里的caches中保存的是元组！！！
    #所以返回值是一个包含元组的列表
    caches = []
    L = len(parameters)//2
    temp_prev = X
    for i in range(1,L):
        W = parameters["W" + str(i)]
        b = parameters["b" + str(i)]
        temp_prev,temp_cache = linear_activation_forward(temp_prev,W,b,activation = "relu")
        caches.append(temp_cache)
    
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL,temp_cache = linear_activation_forward(temp_prev, W, b, activation = "sigmoid")
    caches.append(temp_cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL,caches

#Test L层的relu 模型函数
# X, parameters = L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))
# print(caches)
#Test OK!

#6. 计算这个神经网络的损失函数
def compute_cost(AL,Y):
    
    J_Cost = -np.mean(Y*np.log(AL) + (1-Y) * np.log(1-AL))
    return J_Cost

#Test 计算L层神经网络的损失率
# Y, AL = compute_cost_test_case()

# print("cost = " + str(compute_cost(AL, Y))) 
#Test OK!

#%%最难部分 获得一个反向传播模块函数
#1 计算线性部分的反向导数
#有三个要计算 dw dx db
#其中dx是返回上一层的导数， dwdb存于dic中用于学习率更新使用
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """  
    #看不懂如下公式，可以仔细查看C1_W3_Homework中的backpropogate函数进行检查
    #根据求导公式
    ex_num = dZ.shape[-1]
    A,W,b = cache
    #W的维度为(4,n_pre),dZ的维度为(4,ex_num)
    dA_prev  = np.dot(W.T,dZ) #保证维度为上一层的特征量大小
    assert(dA_prev.shape == A.shape)
    #A的维度为(n_pre,ex_num)
    dW = np.dot(dZ,A.T) / ex_num
    assert(dW.shape == W.shape)
    #
    db = np.mean(dZ,axis = -1,keepdims= True)
    assert(db.shape == b.shape)
    
    return dA_prev,dW,db

#Test 线性模块反向函数
# dZ, linear_cache = linear_backward_test_case()

# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
#Test OK!

#2计算激活函数的反向梯度函数
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    
    return dA_prev, dW, db

#测试激活函数的反向梯度函数
# AL, linear_activation_cache = linear_activation_backward_test_case()

# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")

# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))    
#Test OK!

#3.计算完成整个函数的反向梯度函数
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    L = len(caches)
    grads = {}
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    cache_temp = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, cache_temp, activation = "sigmoid")
    
    for i in reversed(range(0,L-1)):
        cache_temp = caches[i]
        grads["dA"+str(i+1)],grads["dW" +str(i+1)],grads["db" + str(i+1)] = linear_activation_backward(grads["dA" + str(i+2)], cache_temp, activation = "relu")
    
    return grads

# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))   
#Test OK!

#%%梯度更新函数
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
    ### END CODE HERE ###
        
    return parameters

#至此L层的神经网络已经构建完成，下一个作业将使用这个构建完成的神经网络来进行图片处理

  
        
        
    
    

        
        
    

        
    

    
    
    
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
    

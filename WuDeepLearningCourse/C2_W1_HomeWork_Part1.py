# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:14:02 2020
@Discription:本作业详细阐述了不同初始化方法带来的训练结果巨大差异，使用Xavier方法改进而来的He方法拥有最好的效果
@author: Netfather
@Last Modified data: 2021年1月19日
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from C2_W1_HomeWork_DataSet.init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from C2_W1_HomeWork_DataSet.init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()

print(train_X.shape)
print(train_Y.shape)
#300个点，train_X存放的是数据的点位置，train_Y存放的是label标签
# plt.scatter(train_X[0],train_X[1])

#%%初始化方法
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
    
    grads = {}
    costs = []
    train_number = X.shape[-1]
    layer_dims = [X.shape[0],10,5,1] #3层网络，输入特征数2，扩展到10，5后收缩到1，标记为分类
    
    #1.初始化W，b
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layer_dims)
    
    #2.迭代每一步
    for i in range(0,num_iterations):
        
        #直接使用dataset中提供的前向计算函数
        Aout,cache = forward_propagation(X, parameters)
        
        #使用Aout计算损失率
        J_cost = compute_loss(Aout, Y)
        
        #反向传播
        grads = backward_propagation(X, Y, cache)
        
        #根据grads更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost & (i %1000 == 0):
            print("Cost after iteration {}: {}".format(i, J_cost))
            costs.append(J_cost)
            
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


#%%.如上通过通用方法构建了一个标准的深度神经网络模型，但是没定义初始化函数，如下开始
#定义初始化函数

#1.零初始化，即对所有的权重和偏置都是用0
    
def initialize_parameters_zeros(layers_dims):
    L = len(layers_dims)
    parameters = {}
    
    for i in range(1,L):
        parameters.setdefault("W"+str(i),np.zeros([layers_dims[i],layers_dims[i-1]]))
        parameters.setdefault("b"+str(i),np.zeros([layers_dims[i]]).reshape(-1,1))
    
    return parameters

#Test 0初始化函数是否正确
# parameters = initialize_parameters_zeros([3,2,1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#Test OK！

# parameters = model(train_X, train_Y, initialization = "zeros")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

#之所以结果为50%是因为二分类初始化得到的数据集就是 0 1 五五开，所以答案全为0 结果就是50%

# plt.title("Model with Zeros initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


#2.随机初始化函数
def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for i in range(1,L):
        #注意 np.random.randn输入的参数分别行和列，而不是像zeros一样输入一个shape
        parameters.setdefault("W"+str(i),np.random.randn(layers_dims[i],layers_dims[i-1])*10)
        parameters.setdefault("b"+str(i),np.zeros([layers_dims[i]]).reshape(-1,1))
    
    return parameters

#Test 随机初始化参数函数
# parameters = initialize_parameters_random([3, 2, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# parameters = model(train_X, train_Y, initialization = "random")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#Test OK!

#3.使用脱胎于Xavier初始化而来的参数初始化方法
#这个初始化方法针对的是W，不使用之前的*10，而是根据前一层的特征数量来确定这一层的参数
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for i in range(1,L):
        #注意 np.random.randn输入的参数分别行和列，而不是像zeros一样输入一个shape
        parameters.setdefault("W"+str(i),np.random.randn(layers_dims[i],layers_dims[i-1])*np.sqrt(2./layers_dims[i-1]))
        parameters.setdefault("b"+str(i),np.zeros([layers_dims[i]]).reshape(-1,1))
    
    return parameters     

# parameters = initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))    

# parameters = model(train_X, train_Y, initialization = "he")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)


# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#Test OK! 这种优化方法几乎有着完美的准确率

#Part1 结束

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
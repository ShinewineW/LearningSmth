# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:06:31 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from C2_W1_HomeWork_DataSet.reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from C2_W1_HomeWork_DataSet.reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from C2_W1_HomeWork_DataSet.testCases import *


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()

print(train_X.shape)
print(train_Y.shape)
#与Part1作业数据类型非常类似，都是一个代表点位置，一个代表标记

#%% Step1.构建一个model用于验证dropout对结果的提升
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
    
    grads = {}
    costs = []
    train_number = X.shape[-1]
    layer_dims = [X.shape[0],20,3,1] #3层网络，输入特征数2，扩展到10，5后收缩到1，标记为分类
    
    #初始化 W，b 使用Xavier方法
    parameters = initialize_parameters(layer_dims)
    
    for i in range(0,num_iterations):
        #前向计算,此处需要考虑是否使用了dropout正则化
        if keep_prob == 1:
            Aout,cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            Aout,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
            
        #计算Cost这里需要考虑到如果使用正则化手段，那么误差函数的计算
        if lambd == 0:
            J_cost = compute_cost(Aout, Y)
        else:
            J_cost = compute_cost_with_regularization(Aout,Y,parameters,lambd)
        
        #反向传播
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        #更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, J_cost))
        if print_cost and i % 1000 == 0:
            costs.append(J_cost)
        
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

#在不进行任何正则化的情况下 验证正确性
# parameters = model(train_X, train_Y)
# print ("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)


# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#Test OK！


#%%下面加入正则化函数
#1. 首先是L2正则化，即在代价函数中，为W增加一个惩罚项，这个惩罚项会
#抑制W出现过于巨大的数值

#首先完成带了L2正则化的Jcost运算
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    
    train_number = Y.shape[-1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y)
    
    L2_regularization_cost = (1./train_number*lambd/2)*(np.sum(np.square(W1)) + np.sum(W2 * W2) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

#Test
# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()

# print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))

#Test OK!

#然后完成L2正则化的反向传播运算
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    #应当注意到，这个加法项目是独立在原梯度公式以外，只有当需要
    #计算对w1，w2各层导数的时候，才会起作用的一项因此
    
    train_number = X.shape[-1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    #1.按原方计算
    gradients = backward_propagation(X, Y, cache)
    #2.单独抽出dW进行修改
    gradients["dW1"] += lambd / train_number * W1
    gradients["dW2"] += lambd / train_number * W2
    gradients["dW3"] += lambd / train_number * W3
    
    return gradients


#Test 测试带L2正则化的反向求导函数
# X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

# grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("dW3 = "+ str(grads["dW3"]))


# parameters = model(train_X, train_Y, lambd = 0.7)
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
#Test OK！

#2. 完成DroupOut正则化代码

#首先完成关于droupout的正向传播代码
#对于droupout我们应该清楚对于每一个隐藏层，我们都有一个keepprob参数，这个参数
#指导着这一层某个神经元的关闭或开启，在每次迭代中都有随机神经元关闭或开启



    
    
    
     
    
    





















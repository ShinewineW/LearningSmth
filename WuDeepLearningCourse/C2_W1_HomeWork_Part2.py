# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:06:31 2020
@Discription: 本作业详细阐述了使用不同的正则化方法来使得结果更具有非线性，使得结果在最终Test模型上取得最好效果
                1. 不加入任何正则项
                2. 在loss中使用W的二范数进行正则化
                3. 使用Dropout正则化
@author: Netfather
@Last Modified data: 2021年1月19日
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

##



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

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)
    #第一层神经元的结果
    #此时使用droupout技术关闭部分神经元
    Mask1 = np.random.rand(A1.shape[0],A1.shape[1])
    Mask1 = Mask1 < keep_prob #得到矩阵A1对应的掩码
    A1 = A1 * Mask1 #直接进行一个矩阵的相乘
    A1 = A1 / keep_prob #反向droupout中非常关键的一步，这里就可以保证输出的期望和原来相比是不变的
    
    #第二层神经元
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    #此时使用dropout技术关闭部分神经元
    Mask2 = np.random.rand(A2.shape[0],A2.shape[1])
    Mask2 = Mask2 < keep_prob
    A2 = A2 * Mask2
    A2 = A2 / keep_prob
    
    #最终输出
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, Mask1, A1, W1, b1, Z2, Mask2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

# X_assess, parameters = forward_propagation_with_dropout_test_case()

# A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
# print ("A3 = " + str(A3))
#Test OK!前向传播的dropout技术已经完成

#如下代码实现dropout技术的反向传播
#通过cache中的掩码mask可以很方便的将不需要的项归0，但是为了保证二者期望一致，同样需要在输出上/keep_prob

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:

    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    #由于整个函数在一开始就变了 因此需要重新写反向传播的梯度
    gradients = {}
    train_number = X.shape[-1]
    Z1, Mask1, A1, W1, b1, Z2, Mask2, A2, W2, b2, Z3, A3, W3, b3 = cache
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,A2.T) / train_number
    db3 = np.mean(dZ3,axis = -1,keepdims= True)
    
    #第二层
    dA2 = np.dot(W3.T,dZ3) * Mask2/ keep_prob
    temp_mask = A2 > 0
    dZ2 = temp_mask * dA2
    # dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    #考虑keep_prob的问题
    dW2 = np.dot(dZ2,A1.T)  / train_number 
    db2 = np.mean(dZ2,axis = -1,keepdims= True)
    
    #第一层
    dA1 = np.dot(W2.T,dZ2) * Mask1 / keep_prob
    # dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    temp_mask = A1 > 0
    dZ1 = temp_mask * dA1
    #考虑keep_prob的问题
    dW1 =  np.dot(dZ1,X.T)  / train_number 
    db1 = np.mean(dZ1,axis = -1,keepdims= True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


# X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

# gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)

# print ("dA1 = " + str(gradients["dA1"]))
# print ("dA2 = " + str(gradients["dA2"]))
#Test OK!

# 现在让我们使用dropout（keep_prob = 0.86）运行模型。 这意味着在每次迭代中，你都以24％的概率关闭第1层和第2层的每个神经元。 

parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


#至此Part2完结 Dropout在这其中发挥了非常惊艳的效果，可以看到最终的结果决策边界非常的完美
#tricky的点在于使用droupout我们只会在训练中使用，然后对于每一个mask之后 都 马上/keep_prob来保证结果期望的不变

    

    
    



    
    
    
     
    
    





















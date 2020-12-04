# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:57:42 2020
@Discription:这里代码实现了Minibatch的训练方法，以及不同种类的优化函数 有最基础的随机梯度下降，Momentum梯度，Adam梯度
@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from C2_W2_HomeWork_DataSet.opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from C2_W2_HomeWork_DataSet.opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from C2_W2_HomeWork_DataSet.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%%基础版梯度下降，梯度下降的最基础公式
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    L = len(parameters) // 2
    
    for i in range(1,L+1):
        temp_dW  = grads["dW"  + str(i)]
        temp_db = grads["db" + str(i)]
        parameters["W" + str(i)] -= learning_rate*temp_dW
        parameters["b" + str(i)] -= learning_rate*temp_db
    
    return parameters


#测试基础版梯度下降是否正确
# parameters, grads, learning_rate = update_parameters_with_gd_test_case()

# parameters = update_parameters_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#Test OK!

#%%实现Mini_Batch的梯度下降方法
#比较通用的方法为 首先打乱 shuffle 将原有数组按照一定序列的list进行打乱
#然后是划分 Partition 将打乱好的输入数据按照mini_batch_size的大小进行划分
#其中需要考虑余数的问题
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m)) #对m个大小的数字进行随机乱序排列
    shuffled_X = X[:, permutation]  #按照permutation的乱序抽出X重新组织为 shuffled_X
    shuffled_Y = Y[:, permutation].reshape((1,m))
    
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
     
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    #如果数量总大小并不能整除，那么就需要手动处理最后一项
    Yu_Num = m % mini_batch_size
    if Yu_Num  != 0:
        mini_batch_X = shuffled_X[:,-Yu_Num::]
        mini_batch_Y= shuffled_Y[:,-Yu_Num::]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

#Test OK!
# X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

# print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
# print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
# print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
# print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
# print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
# print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
# print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

#%%下面是有效的优化算法
#1. 动量优化算法
def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    #初始化一个最开始存储速度的词典，后续就由上一次的速度进行维护
    L = len(parameters) // 2
    v = {}
    
    #这里是借用parameters中的各层大小来初始化v，v的维护在更新过程中由自己负责
    for i in range(1, L + 1):
        v.setdefault("dW" + str(i),np.zeros(parameters["W" + str(i)].shape))
        v.setdefault("db" + str(i),np.zeros(parameters["b" + str(i)].shape))
    
    return v

# parameters = initialize_velocity_test_case()

# v = initialize_velocity(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
#Test OK!

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """
    
    L = len(parameters) // 2
    
    for i in range(1,L+1):
        #首先根据上一次速度更新这一次速度
        v["dW" + str(i)] = beta*v["dW" + str(i)] + (1-beta)*grads["dW" + str(i)]
        v["db" + str(i)] = beta*v["db" + str(i)] + (1-beta)*grads["db" + str(i)]        
        #然后根据速度更新各参数
        parameters["W" + str(i)]  -= learning_rate * v["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * v["db" + str(i)]
        
    return parameters,v


# parameters, grads, v = update_parameters_with_momentum_test_case()

# parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
#TEST OK!

#%%下面是Adam优化器的实现过程
#Adam优化器统筹了RMSprop和momentum优化的优点，同时为了解决初始起步优化效果差的问题
#还使用了偏差校准

#1.根据parameters中的w和b 初始化v与s
def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    L = len(parameters) // 2
    v = {}
    s = {}
    
    #这里是借用parameters中的各层大小来初始化v，v的维护在更新过程中由自己负责
    for i in range(1, L + 1):
        v.setdefault("dW" + str(i),np.zeros(parameters["W" + str(i)].shape))
        v.setdefault("db" + str(i),np.zeros(parameters["b" + str(i)].shape))
        s.setdefault("dW" + str(i),np.zeros(parameters["W" + str(i)].shape))
        s.setdefault("db" + str(i),np.zeros(parameters["b" + str(i)].shape))
    
    return v,s

# parameters = initialize_adam_test_case()

# v, s = initialize_adam(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))
   #Test OK!
   
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    
    for i in range(1,L+1):
        
        #1.获得v_W 和 s_W 然后更新w
        v["dW" + str(i)]  = beta1 * v["dW" + str(i)]  + (1-beta1) * grads["dW"  + str(i)]
        v_corrected["dW" + str(i)] = v["dW" + str(i)]  / (1-pow(beta1,t))
        s["dW" + str(i)]  = beta2 * s["dW" + str(i)]  + (1-beta2) * (grads["dW"  + str(i)]**2)
        s_corrected["dW" + str(i)] = s["dW" + str(i)]  / (1-pow(beta2,t))
        parameters["W" + str(i)] -= learning_rate * v_corrected["dW" + str(i)] / (np.sqrt(s_corrected["dW" + str(i)]) + epsilon)
        
        #同样方法更新db
        v["db" + str(i)]  = beta1 * v["db" + str(i)]  + (1-beta1) * grads["db"  + str(i)]
        v_corrected["db" + str(i)] = v["db" + str(i)]  / (1-pow(beta1,t))
        s["db" + str(i)]  = beta2 * s["db" + str(i)]  + (1-beta2) * (grads["db"  + str(i)]**2)
        s_corrected["db" + str(i)] = s["db" + str(i)]  / (1-pow(beta2,t))
        parameters["b" + str(i)] -= learning_rate * v_corrected["db" + str(i)] / (np.sqrt(s_corrected["db" + str(i)]) + epsilon)
        
    return parameters,v,s

# parameters, grads, v, s = update_parameters_with_adam_test_case()
# parameters, v, s  = update_parameters_with_adam(parameters, grads, v, s, t = 2)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))
#Test OK!

#%%下面使用三种不同优化模型来考察效果
train_X, train_Y = load_dataset()

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters        
#%%
#1.小批量梯度下降算法
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
        
#%%
#2.带冲量的小批量梯度下降
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#%%
#3.使用Adam优化器的小批量梯度下降
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#至此已经完成对于小批量梯度下降算法的描述 以及最原始的梯度下降，带动量的梯度下降以及Adam优化器梯度下降的描述


        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:20:32 2020
@Discription 本文件实现了一个含有单个隐藏层，隐藏神经元数为4的神经网络的构建
                1.使用模型对一个花瓣图形进行了简单分类。

@author: Netfather
@Last Modified data: 2021年1月19日
"""

import numpy as np
import matplotlib.pyplot as plt
from C1_W3_HomeWork_DataSet.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from C1_W3_HomeWork_DataSet.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1)

#%%引入数据集
X,Y = load_planar_dataset()

#可视化数据集
plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0,:].shape), s=40, cmap=plt.cm.Spectral)

#这个数据集是由
# 包含特征（x1，x2）的numpy数组（矩阵）X
# 包含标签（红色：0，蓝色：1）的numpy数组（向量）Y。

#%%考察数据集的形状结构特征
print(X.shape)
print(Y.shape)

#可以知道这个数据是400个大小，每个数据含有两个特征 x1，x2，然后正确的标签信息存储在Y中

train_number = X.shape[-1]
print(train_number)

#%%使用简单逻辑回归来观察这个分类问题
#直接使用sklearn中的线性模型来快速得到

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

#看得出来由于数据集并不是线性可分，所以分割效果其实很糟糕

#%%使用含有单个隐藏层的神经网络进行分割

# 建立神经网络的一般方法是：
# 1.定义神经网络结构（输入单元数，隐藏单元数等）。
# 2.初始化模型的参数
# 3.循环：

# 实现前向传播
# 计算损失
# 后向传播以获得梯度
# 更新参数（梯度下降）
# 我们通常会构建辅助函数来计算第1-3步，然后将它们合并为nn_model()函数。一旦构建了nn_model()并学习了正确的参数，就可以对新数据进行预测。

#%%
#1.定义神经网络结构
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    #由于这次数据是流入，不需要把所有样本全部合一起来，因此输入层的大小就为2
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x,n_h,n_y)

#2. 初始化模型参数
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """  
    #参数W的大小非常重要，可以理解为是把2个特征通过参数w投射为4个特征
    
    W1 = np.random.randn(n_h,n_x)*0.01
    #参数W的大小，仅由下一层的大小和输入的特征数决定
    b1 = np.zeros(n_h).reshape(n_h,1)
    
    #W2为最后sigmoid函数所在位置的参数
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(n_y).reshape(n_y,1)
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# n_x, n_h, n_y = initialize_parameters_test_case()

# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
    
#3.实现模型的前向传播
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    #得到参数
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    #前向传播
    Z_Hidden = np.dot(W1,X) + b1
    A_Hidden = np.tanh(Z_Hidden)
    #(4*400)
    #这样每个样本就从原来的2特征映射为4特征
    
    Z_Output = np.dot(W2,A_Hidden) + b2
    A_Output = sigmoid(Z_Output)
    #(1*400)
    
    assert(A_Output.shape == (1, X.shape[-1]))
    
    cache = {"Z_Hidden": Z_Hidden,
             "A_Hidden": A_Hidden,
             "Z_Output": Z_Output,
             "A_Output": A_Output}
    
    return A_Output, cache 

# #Test
# X_assess, parameters = forward_propagation_test_case()

# A2, cache = forward_propagation(X_assess, parameters)

# # Note: we use the mean here just to make sure that your output matches ours. 
# print(np.mean(cache['Z_Hidden']) ,np.mean(cache['A_Hidden']),np.mean(cache['Z_Output']),np.mean(cache['A_Output']))

#4. 实现代价函数的计算

def compute_cost(A_Output, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """    
    
    yhat = A_Output
    #yi = np.where(yhat <= 0.5,0,1)
    #计算代价函数 注意这里用的是元素乘法，因为维度是同等大小的
    J_Cost = -np.mean(Y*np.log(yhat)+ (1-Y)*np.log(1-yhat))

    J_Cost = np.squeeze(J_Cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(J_Cost, float))
    
    return J_Cost

# A2, Y_assess, parameters = compute_cost_test_case()

# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
    
#5.反向传播计算
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    train_number = X.shape[-1]
    
    Z_Hidden = cache["Z_Hidden"]
    A_Hidden = cache["A_Hidden"]
    Z_Output = cache["Z_Output"]
    A_Output = cache["A_Output"]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    
    #计算反向传播的通理论，反向Z的结果是直接在矩阵上运算
    #然后具体到参数上来，再来计算1/m应该放在哪里
    dZ_Output = A_Output - Y
    
    dW2 = np.dot(dZ_Output,A_Hidden.T)/train_number
    db2 = np.mean(dZ_Output,axis = -1,keepdims= True)
    
    #下面注释的两行使用的是Relu激活函数
    # dZ_Temp =  np.dot(W2.T,dZ_Output)
    # dZ_Hidden = np.where(Z_Hidden > 0,dZ_Temp,0)
    dZ_Hidden = np.dot(W2.T,dZ_Output) * (1-np.power(A_Hidden,2))   #和Z_Hidden的矩阵维度一致 (4*m)
    dW1 = np.dot(dZ_Hidden,X.T)/train_number
    db1 = np.mean(dZ_Hidden,axis = -1,keepdims= True)
    
    grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}
    
    return grads
 
#Test   
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))   

#6. 参数更新
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
       
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
     
    return parameters

#Test
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#%% 将上述建立的前向传播，损失计算，反向传播，梯度更新整合在一起
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    
    for i in range(num_iterations):
        
        A_Output,cache = forward_propagation(X,parameters)
        
        
        J_Cost = compute_cost(A_Output,Y,parameters)
        
        grads = backward_propagation(parameters, cache, X, Y)
        
        parameters = update_parameters(parameters, grads)
        
        if i % 1000:
            if print_cost :
                print ("Cost after iteration %i: %f" %(i, J_Cost))
        
    return parameters

# X_assess, Y_assess = nn_model_test_case()

# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))  

#%%预测函数 使用训练完成的参数进行预测
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
  ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    ### END CODE HERE ###
    
    return predictions    

#%%在搭建好的模型上运行最终的结果
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))


#%%可选练习
#1. 调整隐藏层大小
plt.figure(figsize = (16,32),dpi = 80)
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    

    

        























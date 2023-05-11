# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:36:58 2020
Written by Netfather
Learnned from C1_W2_HomeWork
为了完整使用这份作业
你需要完成如下步骤
1.安装numpy matplotlib opencv h5py一共四个包
2.将这个py文件所在目录加入到当前python的环境路径中
3.挨个运行一边查看效果
@Discription: 本文件实现了一个最简单的单层神经网络的实现。即直接将图片拍扁之后，将特征映射到输出为1的空间上
                1.构建前向传播
                2.进行反向传播
                3.更新所有权重
                4.进行训练和预测
@author: Netfather
@Last Modified data: 2021年1月19日
"""

#%% 导入所有的包 运行一遍检查是否有错误

import numpy as np
import matplotlib.pyplot as plt
from C1_W2_HomeWork_DataSet.lr_utils import load_dataset
import C1_W2_HomeWork_Part1 as C1_W2_Work
import cv2  #如果pycharm中报错，请在pycharm中安装 opencv_python

#%% 导入数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#%% 从数据集中抽出一些照片进行考察
index = 5
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

print(train_set_x_orig.shape)
#说明训练数据有 209张 64*64*3的图片
print(train_set_y.shape)
#说明标签数据是一个和图片一一对应的行向量
print(test_set_x_orig.shape)
print(test_set_y.shape)

#%% Step1. 图片处理 现在我们把图片处理成拍扁的矩阵
#处理方法为对第一张图变成一个列向量，第二张也变成列向量然后叠放这样
#所以最后的大小应该为 64*64*3，209大小

train_number,train_height,train_weight,train_channel = train_set_x_orig.shape
test_number,test_height,test_weight,test_channel = test_set_x_orig.shape

train_matrix = train_set_x_orig.reshape(train_number,-1).T
print(train_matrix.shape)
test_matrix = test_set_x_orig.reshape(test_number,-1).T
print(test_matrix.shape)

#%%标准化数据,一般要减去numpy数组的均值，然后除以标准差
#但是图片数据集只需要除以255即可

train_matrix = train_matrix / 255.
test_matrix = test_matrix / 255.

#%% Sigmoid函数从之前文件import
#下一步初始化权重矩阵

def initialized_with_zeros(dim):
    
    w = np.zeros(dim).reshape(dim,1)
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w,b

#%% 前向传播和后向传播

def propagate(w,b,X,Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """
    train_number = X.shape[-1]
    
    yhat = C1_W2_Work.sigmoid(np.dot(w.T,X) + b)
    
    cost = -np.mean(Y*np.log(yhat) + (1-Y)*np.log(1-yhat))
    
    dw = np.dot(X, (yhat - Y).T) / train_number
    
    db = np.mean(yhat - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

##TEST
# w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))

#%% 优化函数
#使用梯度下降算法来对参数进行优化
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        grads,cost = propagate(w, b, X, Y)
        
        #赋值 dw db
        dw = grads["dw"]
        db = grads["db"]
        
        #更新规则
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #每100次迭代生成一次日志
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))
        
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
    
#Test 
# params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print(costs)    

#%%编写单个图片的测试函数

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    image_number = X.shape[-1]
    Y_Prediction = np.zeros(image_number).reshape(1,-1)
    w = w.reshape(X.shape[0],1)
    
    yhat = C1_W2_Work.sigmoid(np.dot(w.T, X) + b)
    
    for i in range(yhat.shape[-1]):
        
        if yhat[:,i] <= 0.5:
            Y_Prediction[:,i] = 0
        if yhat[:,i] > 0.5:
            Y_Prediction[:,i] = 1
        
    assert(Y_Prediction.shape == (1, image_number))
    
    return Y_Prediction

#print ("predictions = " + str(predict(w, b, X)))    
        
#%%将所有上述构建全部整合到model函数中
     # Y_prediction对测试集的预测
     # Y_prediction_train对训练集的预测
     # w，损失，optimize（）输出的梯度 
     
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """     
    w,b = initialized_with_zeros(X_train.shape[0])
    #w ,b 的大小和一副图像拍扁后的大小有关系
    
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,True)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)   
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d  

    
d = model(train_matrix, train_set_y, test_matrix, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


#%%
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#%% 展示不同学习率下的收敛曲线
# learning_rates = [0.01, 0.001, 0.0001]
# models = {}
# for i in learning_rates:
#     print ("learning rate is: " + str(i))
#     models[str(i)] = model(train_matrix, train_set_y, test_matrix, test_set_y, num_iterations = 1500, learning_rate = i,  print_cost = True)
#     print ('\n' + "-------------------------------------------------------" + '\n')

# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

# plt.ylabel('cost')
# plt.xlabel('iterations')

# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()

#%% 预测自己的图片
#由于最新版的scipy中已经不包含原本的函数模块，这里直接使用opencv来对输入的图片进行处理


fname = r'C1_W2_HomeWork_DataSet/cat_in_iran.jpg'
image = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
my_image = np.array(cv2.resize(image, (train_height,train_weight))).reshape((1, train_height*train_weight*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")








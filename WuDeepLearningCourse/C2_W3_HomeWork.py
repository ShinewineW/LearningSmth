# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:15:08 2020
@Description: 使用Tensorflow1.x
@author: Administrator
"""

#%%
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from C2_W3_HomeWork_DataSet.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)


#%%
#测试tf 1.x
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss


#%%在tf1.x中进行一个神经网络的构建和训练需要完成如下操作
    #1.首先创建一个专门用于tf的数据结构，张量，这其中会存储有可训练参数和喂入的训练
    #2.在这些张量之间编写操作，构建完成张量的计算图
    #3.调用函数初始化所有参数
    #4.创建一个会话
    #5.运行会话，这将运行上面所写的所有操作
    
#One. 计算图必须要初始化并运行之后，才能看到想看的运算结果

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
#这里是看不到结果的 为了看到结果，我们必须
sess = tf.Session()
print(sess.run(c))

#Two. 占位符操作，某些数据是在当前不赋值，在训练过程中，喂入神经网络的

x = tf.placeholder(tf.int64,name = 'x')
print(sess.run(2*x,feed_dict = {x : 3 }))
sess.close()

#这里使用placeholder来对未知的x进行赋值，然后通过feed_dict来喂入指定的数字

#%%线性函数
#X = tf.constant(np.random.randn(3,1), name = 'X')

def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b   
    """
    np.random.seed(1)
    #构建张量
    X = tf.constant(np.random.randn(3,1), name = 'X')
    W = tf.constant(np.random.randn(4,3), name = 'W')
    b = tf.constant(np.random.randn(4,1), name = 'b')
    
    #由于没有需要喂入或者声明为需要训练的量，直接构建计算图
    Y = tf.add(tf.matmul(W, X),b)
    
    sess = tf.Session()
    result = sess.run(Y)
    
    sess.close()
    
    return result

#测试线性函数的结果
# print( "result = " + str(linear_function()))
#Test OK!

#2.计算sigmoid函数
def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    #既可以使用占位符操作来将z的值喂入x，也可以使用conver_to_tensor操作来将z转为张量
    
    z = tf.convert_to_tensor(z,dtype=tf.float32)
    
    Sig_z = tf.sigmoid(z)
    
    sess = tf.Session()
    result = sess.run(Sig_z)
    sess.close()
    
    return result

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))


#3.计算总损失函数J_cost,也就是交叉熵损失
def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    z = tf.placeholder(tf.float32,name = 'z')
    y = tf.placeholder(tf.float32,name = 'y')
    
    J_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,labels = y)
    
    sess = tf.Session()
    
    J_cost = sess.run(J_cost,feed_dict = {z:logits,y:labels})
    
    sess.close()
    
    return J_cost

#测试损失函数
# logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# cost = cost(logits, np.array([0,0,1,1]))
# print ("cost = " + str(cost))
#Test OK!

#4.使用One_hot独热码来对标签进行分类
def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    #这里的one_hot具有泛用性，只需要给个label同时指定好有多少类，就可以自动实现
    #从标签到one_hot编码的转换
    C = tf.constant(C,name = "C")
    
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    
    sess = tf.Session()
    
    one_hot = sess.run(one_hot_matrix)
    
    sess.close()
    
    return one_hot

#测试Onehot编码是否成功
# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels, C = 4)
# print ("one_hot = " + str(one_hot))
#Test OK

#5.实现初始化
def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    ### START CODE HERE ###  
    ones = tf.ones(shape)  
      
    sess = tf.Session()  
      
    ones = sess.run(ones)  
      
    sess.close()  
      
    return ones  
 
   
# print ("ones = " + str(ones([3])))
# 测试OK

#%%使用tensorflow构建神经网络
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
#我们从数据集中抽取一些来观察
print(X_train_orig.shape)
print(Y_train_orig.shape)
print(X_test_orig.shape)
print(classes)

#训练数据集为1080张  64*64*3的图片
#测试数据集为120张   64*64*3的图片
#classes是一个表明有多少类的列表

# plt.imshow(X_train_orig[2])
# plt.title(np.squeeze(Y_train_orig[:,2]))


#下面开始对数据集进行处理，首先我们需要对图片展平，然后对标签进行编码
X_train_flatten = X_train_orig.reshape(-1,64*64*3).T
#将输入数据集装置成 (featrues_num,m)的形式
X_test_flatten = X_test_orig.reshape(-1,64*64*3).T

#归一化数据集
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

#将标签数据转换为独热码
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig,6)

# print ("number of training examples = " + str(X_train.shape[1]))
# print ("number of test examples = " + str(X_test.shape[1]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))
#TEST ok!

#创建占位符，将n_x,n_y放入占位符中
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    #本质为创建两个张量结构，定义为可以喂入数据的placeholder
    #确定X维度 然后喂入多少张待定
    X = tf.placeholder(dtype = tf.float32,shape = [n_x,None])
    Y = tf.placeholder(dtype = tf.float32,shape = [n_y,None])
    
    return X,Y

# X, Y = create_placeholders(12288, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))
# TEST OK!


#完成对XY的设置之后，对各层参数进行初始化
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    #初始化各层的参数和偏置
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)  
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))  
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())  
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))  
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())  
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))  
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())  
    ### END CODE HERE ### 

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


#在Tensorflow中进行正向传播
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:  
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1  
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)  
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2  
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)  
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3,Z2) + b3  
    ### END CODE HERE ###  
    
    return Z3

#根据得到的Z3 分别通过激活函数得到A3 以及通过A3计算最终的J_cost
def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))  
    ### END CODE HERE ###  
    
    return cost

#到这里之后的反向传播以及参数优化，我们就可以让tf自动帮我们完成了

#建立模型！
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []
    
    #1.初始化两类参数 占位符参数 和 可训练参数
    X, Y = create_placeholders(n_x, n_y)  
    parameters = initialize_parameters()  
    #2.完成前向传播
    Z3 = forward_propagation(X, parameters)  
    #3.完成loss计算
    cost = compute_cost(Z3, Y)  
    #4.执行反向传播和优化
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)  
    
    #5.初始化gloabal参数
    init = tf.global_variables_initializer()
    
    #6.执行计算图
    with tf.Session() as sess:
        sess.run(init)  #计算图初始化#1中参数
        
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch  
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set  
            seed = seed + 1  
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X , minibatch_Y )= minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}) 
                epoch_cost += minibatch_cost / num_minibatches  
                
            if print_cost == True and epoch % 100 == 0:  
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))  
            if print_cost == True and epoch % 5 == 0:  
                costs.append(epoch_cost)  
                
        plt.plot(np.squeeze(costs))  
        plt.ylabel('cost')  
        plt.xlabel('iterations (per tens)')  
        plt.title("Learning rate =" + str(learning_rate))  
        plt.show()  
  
        # lets save the parameters in a variable  
        parameters = sess.run(parameters)  
        print ("Parameters have been trained!")  
  
        # Calculate the correct predictions  
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))  
  
        # Calculate accuracy on the test set  
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
  
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))  
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))  
          
        return parameters  
    

parameters = model(X_train, Y_train, X_test, Y_test)

#%%如下代码实现对自定义图片进行分类
import cv2
import numpy as np
import matplotlib.pyplot as plt

my_image = cv2.imread(r"./C2_W3_HomeWork_DataSet/thumbs_up.jpg",cv2.IMREAD_UNCHANGED)

cv2.imshow("origin",my_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
my_image = np.array(my_image) 
image_show = cv2.resize(my_image,(64,64))
print(my_image.shape)
my_image = image_show.reshape(-1,1)
print(my_image.shape)
my_image_prediction = predict(my_image, parameters)

plt.imshow(image_show)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

#至此已经完成了通过tensorflow1.0来实现一个深度神经网络
#和之前的方法相比，这种方法的优势在于我们可以很方便的完成反向梯度的传播运算

             
                
  
    
    




















# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W1_HomeWork_Part2
# Description:  本次作业完成了用RNN网络来学习并输出恐龙的名字
#               在实践过程中，我们对X,Y指定一个字符表，通过独热码的方式来指定当前值应该为多少
#               输入就是这个字符对应的index， 然后通过clip手段暴力，当梯度超过某个指定值，就直接将
#               该值指定为这个最大值。 完成训练。
# Author:       Administrator
# Date:         2021/1/5
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------
##
import numpy as np
from C5_W1_HomeWork_Part2_DataSet.utils import *
import random
from random import shuffle


##导入数据集 并创建唯一字符列表，并计算数据集和词汇量
data = open(r'./C5_W1_HomeWork_Part2_DataSet/dinos.txt','r').read()
data = data.lower()
chars = list(set(data))
data_size ,vocab_size = len(data),len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

#创建两个哈希表，哈希表将字符映射到数字，同样为了节约时间 ，我们也将数字映射到字符
#这是反向的两个哈希表
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)


##下面进行模型构建，为了避免超级长度RNN出现的梯度弥散或者梯度爆炸，我们需要使用
#梯度裁剪的方法。
# 在下面的练习中，你将实现一个函数clip，该函数接收梯度字典，并在需要时返回裁剪后
# 的梯度。梯度裁剪有多种方法。我们将使用简单的按元素裁剪程序，其中将梯度向量的每
# 个元素裁剪为位于范围[-N，N]之间。通常，你将提供一个maxValue（例如10）。在此
# 示例中，如果梯度向量的任何分量大于10，则将其设置为10；并且如果梯度向量的任何
# 分量小于-10，则将其设置为-10。如果介于-10和10之间，则将其保留。
def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''
    #将字典中的每个值都检查一边 超过最大值的通通都裁剪为这个最大值
    for key,value in gradients.items():
        np.clip(value,-maxValue,maxValue,out=value)

    return gradients


# Test OK!
# np.random.seed(3)
# dWax = np.random.randn(5,3)*10
# dWaa = np.random.randn(5,5)*10
# dWya = np.random.randn(2,5)*10
# db = np.random.randn(5,1)*10
# dby = np.random.randn(2,1)*10
# gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
# gradients = clip(gradients, 10)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])


#对已经训练好的网络进行采样
#1.将第一个输入传递给网络，一般我们都指定0位置的激活a0 = 0
#2.执行正向传播，例如完成第一个rnncell之后，我们可以得到a1，y_pred1
#3.执行采样，这个y_pred1是一个由softmax函数得到的概率，这个概率表明，在at-1为某个字符的情况下，输出为某个字符的概率集合
#4.覆盖变量x，由于是一个连续的输出，因此这一步随机得到的字符将成为下一个rnncell的输入

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    #首先从输入fetch回来指定尺寸用于构建a0与x0
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]  #输出单词的长度，也就是要输出多长
    n_a = Waa.shape[1]

    #为采样创建初始化序列
    a = np.zeros((n_a,1))
    x = np.zeros((vocab_size,1))

    idx = -1  #标志位 用于判断是否结束本次采样

    counter =0 #random相关
    newline_character = char_to_ix['\n'] #得到结束标志位，也就是采样到'\n'结束
    indices = []
    #开始采样
    while (idx != newline_character and counter != 50):
        a, y_pred = rnn_step_forward(parameters,a,x)
        np.random.seed(counter+seed)
        idx = np.random.choice(range(len(y_pred)), p=y_pred.ravel())
        indices.append(idx)
        #将下一个的x输入重写为本次的预测值
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        seed += 1
        counter +=1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


#Test OK!
# np.random.seed(2)
# n, n_a = 20, 100
# a0 = np.random.randn(n_a, 1)
# i0 = 1 # first character is ix_to_char[i0]
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
#
#
# indices = sample(parameters, char_to_ix, 0)
# print("Sampling:")
# print("list of sampled indices:", indices)
# print("list of sampled characters:", [ix_to_char[i] for i in indices])


##现在可以开始构建语言模型了
# 在本部分中，你将实现一个函数，该函数执行随机梯度下降的一个步骤（梯度裁剪）。你将一次
# 查看一个训练示例，因此优化算法为随机梯度下降。提醒一下，以下是RNN常见的优化循环的
# 步骤：
#
# 通过RNN正向传播以计算损失
# 随时间反向传播以计算相对于参数的损失梯度
# 必要时裁剪梯度
# 使用梯度下降更新参数

#在utls.py文件中 提供了 实现RNN模型构建的三个模块
# def rnn_forward(X, Y, a_prev, parameters):
#     """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
#     It returns the loss' value as well as a "cache" storing values to be used in the backpropagation."""
#     ....
#     return loss, cache
#
# def rnn_backward(X, Y, parameters, cache):
#     """ Performs the backward propagation through time to compute the gradients of the loss with respect
#     to the parameters. It returns also all the hidden states."""
#     ...
#     return gradients, a
#
# def update_parameters(parameters, gradients, learning_rate):
#     """ Updates parameters using the Gradient Descent Update Rule."""
#     ...
#     return parameters

def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    #从输入中fetch尺寸用于构建a


    #前向传播
    loss,cache = rnn_forward(X,Y,a_prev,parameters)

    gradients,a = rnn_backward(X,Y,parameters,cache)

    #梯度裁剪 避免出现梯度爆炸
    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters,gradients,learning_rate)

    return loss,gradients,a[len(X)-1]


#Test OK!
# np.random.seed(1)
# vocab_size, n_a = 27, 100
# a_prev = np.random.randn(n_a, 1)
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
# X = [12,3,5,11,22,3]
# Y = [4,14,11,22,25, 26]
#
# loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
# print("Loss =", loss)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])
# print("a_last[4] =", a_last[4])

#开始训练模型
# 给定恐龙名称数据集，我们将数据集的每一行（一个名称）用作一个训练示例。每100步
# 随机梯度下降，你将抽样10个随机选择的名称，以查看算法的运行情况。请记住要对
# 数据集进行混洗，以便随机梯度下降以随机顺序访问示例。

def model(data, ix_to_char, char_to_ix, num_iterations=10000, n_a=50, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """
    n_x, n_y = vocab_size, vocab_size

    #初始化所有所需要的参数
    parameters = initialize_parameters(n_a,n_x,n_y)

    #初始化loss
    loss = get_initial_loss(vocab_size,dino_names)

    #读入数据，为所有恐龙名字加入一个列表中
    with open(r"./C5_W1_HomeWork_Part2_DataSet/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    #打乱examples
    shuffle(examples)

    a_prev = np.zeros((n_a,1))

    for j in range(num_iterations):

        #获得数据集
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        #开始训练
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        #平滑loss
        loss = smooth(loss,curr_loss)

        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters


##
parameters = model(data, ix_to_char, char_to_ix)


##


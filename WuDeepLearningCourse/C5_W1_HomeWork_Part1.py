
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W1_HomeWork_Part1
# Description:  完成了吴恩达课程C5_W1_HomeWork的作业
#               本课程构建了 LSTM 和 RNN 的基本单元 与 前向传播过程
#               同时完成了对于二者反向传播的构建  难点如下
#               对于RNN反向传播： 我们可以根据输出维度和某一方的输入维度，反向推导是否需要进行转置
#                               在整体的反向传播过程中，由于从loss处反向传播而来的梯度da是一个(a_y,m,T_x)的张量，我们需要同时考虑da和da_next
#                               所以在计算过程中 我们输入的值为   da[::,::,i] + da_next
#               对于LSTM反向传播：我们需要记住，除了输出门，其余两个门和候选ct都是根据 Ctnext 和 atnext同时作用的，因此在计算梯度时
#                               需要同时考虑这两个反向的梯度，然后相加，得到总的梯度。
#                               在整体的反向传播过程中，由于从loss处反向传播而来的梯度da是一个(a_y,m,T_x)的张量，我们需要同时考虑da和da_next
#                               所以在计算过程中 我们输入的值为   da[::,::,i] + da_next
#                               而由于ct是一个反向传播过来的隐藏在LSTM中的信息流，不存在外部loss，因此不用加法，直接输入 dc_next即可。
# Author:       Netfather
# Date:         2021/1/4
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------

##本课程使用基础的numpy搭建一个基础RNN网络
#导入必要的包
import numpy as np
from C5_W1_HomeWork_Part1_DataSet.rnn_utils import *

##
#执行一个RNN的正向传播过程，具体网络结构参看课程笔记
# 说明：
#
# 使用tanh激活计算隐藏状态：at
# 使用新的隐藏状态，计算预测。我们为你提供了一个函数：softmax。
# 将必要前向传播信息存储的缓存中 便于后续调用
# 返回 输出at以及缓存

# GRADED FUNCTION: rnn_cell_forward

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    #从输入数据中重新获得必要信息。
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    #计算下一个输出的隐藏状态 at
    at =  np.tanh(np.dot(Wax,xt) + np.dot(Waa,a_prev) + ba)
    yt =  softmax(np.dot(Wya,at)+by)

    #将必要信息存储在cache中方便反向传播计算
    cache = (at,a_prev,xt,parameters)

    return at,yt,cache

#Test OK！
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("yt_pred[1] =", yt_pred[1])
# print("yt_pred.shape = ", yt_pred.shape)

#进行RNN网络结构的搭建
#根据输入的时间长度，来实例化一个RNN的前向传播过程
# 创建一个零向量（a），该向量将存储RNN计算的所有隐藏状态。
# 将“下一个”隐藏状态初始化为a0（初始隐藏状态）。
# 开始遍历每个时间步，增量索引为t：
#      通过运行rnn_step_forward更新“下一个”隐藏状态和缓存。
#     将“下一个”隐藏状态存储在a中的t号（位置）
#     将预测存储在y中
#     将缓存添加到缓存列表中
# 返回a,y和缓存
def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """

    #应当注意到 所有的cell都是共用一组参数权重
    #从输入中fetch必要参数
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    n_x,m,Xt = x.shape
    n_y,n_a = Wya.shape
    caches = []

    #为输出一系列的激活值a和预测值y_pred进行初始化
    a = np.zeros((n_a,m,Xt))
    y_pred = np.zeros((n_y,m,Xt))

    a_prev = a0

    for i in range(Xt):
        a_prev,yt,cache = rnn_cell_forward(x[::,::,i],a_prev,parameters)
        a[::,::,i] = a_prev
        y_pred[::,::,i] = yt
        caches.append(cache)

    caches = (caches,x)

    return a,y_pred,caches


# Test OK!
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1] = ", a[4][1])
# print("a.shape = ", a.shape)
# print("y_pred[1][3] =", y_pred[1][3])
# print("y_pred.shape = ", y_pred.shape)
# print("caches[1][1][3] =", caches[1][1][3])
# print("len(caches) = ", len(caches))

##除了基础RNN我们还可以实现 长短期记忆单元 LSTM 的构建
#具体LSTM的结构参看学习笔记或者WuCourse教学视频

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the save gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the focus gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the focus gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilda),
          c stands for the memory value
    """

    #从输入中fetch必要参数
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    #fech回主要尺寸
    n_x,m = xt.shape
    n_a,m = a_prev.shape

    #1. 先将输入合并
    stack_input = np.zeros((n_a+n_x,m))
    stack_input[:n_a:,::] = a_prev
    stack_input[n_a::,::] = xt
    #2.计算三个门 遗忘门，保留门，输出门，以及一个候选Cn
    gate_forget = sigmoid( np.dot(Wf,stack_input)+bf)
    gate_update = sigmoid( np.dot(Wi,stack_input)+bi)
    gate_output = sigmoid( np.dot(Wo,stack_input)+bo)
    cn_candidate = np.tanh( np.dot(Wc,stack_input)+bc)
    #3.计算隐藏层cn和输出an
    cn = np.multiply(gate_update,cn_candidate) + np.multiply(gate_forget,c_prev)
    an = np.multiply(gate_output,np.tanh(cn))
    #4.将an通过softmax得到y_pred
    y_pred = softmax(np.dot(Wy,an) + by)
    #5.得到cache以供反向传播计算
    cache = (an, cn, a_prev, c_prev, gate_forget, gate_update, cn_candidate, gate_output, xt, parameters)

    return an,cn,y_pred,cache


# Test OK!
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", c_next.shape)
# print("c_next[2] = ", c_next[2])
# print("c_next.shape = ", c_next.shape)
# print("yt[1] =", yt[1])
# print("yt.shape = ", yt.shape)
# print("cache[1][3] =", cache[1][3])
# print("len(cache) = ", len(cache))
##
#然后根据上面搭建好的LSTM的单元模块 实现一个完整的LSTM组成的RNN单元
def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the save gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the focus gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the focus gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    #根据输入得到必要的尺寸参数
    n_x,m,T_x = x.shape
    n_a,_ = a0.shape
    n_y,_ = parameters['Wy'].shape

    #为输出实例化两个矩阵
    a = np.zeros((n_a,m,T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y,m,T_x))
    caches = []

    a_n = a0
    c_n = np.zeros((n_a, m)) #注意初始隐藏状态需要你自己初始化一个zeros矩阵

    for i in range(T_x):
        a_n, c_n, y_pred, cache = lstm_cell_forward(x[::,::,i],a_n,c_n,parameters)
        a[::,::,i] = a_n
        y[::,::,i] = y_pred
        c[::,::,i] = c_n
        caches.append(cache)

    caches = (caches,x)

    return a,y,c,caches

# Test OK!
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y,c, caches = lstm_forward(x, a0, parameters)
# print("a[4][3][6] = ", a[4][3][6])
# print("a.shape = ", a.shape)
# print("y[1][4][3] =", y[1][4][3])
# print("y.shape = ", y.shape)
# print("caches[1][1[1]] =", caches[1][1][1])
# print("c[1][2][1]", c[1][2][1])
# print("len(caches) = ", len(caches))

##下面开始反向传播算法的构建
#基础RNN的反向传播 对于一个最简单的RNN单元 我们进行反向传播
#对于一个从后向传播回来的da_next 维度为 (n_a,m)
#通过这个da_next得到前向的所有输入的梯度

def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_step_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """

    #从输入中fetch回必要参数
    at, a_prev, xt, parameters = cache
    gradients = {}

    #从参数词典中fetch必要权重矩阵
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    ba = parameters['ba']

    #反向传播列出公式
    # tanh(wxa*xt + waa*at-1 + ba) = a_next
    # tanh(x) 的导数为 1-tanh(x)的平方
    #上面式子中的中括号内容为
    #根据应当得到的输出维度，和输入某一方的维度，我们可以很容易推导这个维度是否需要转置
    temp_tanh = 1. - np.multiply((at),(at))
    dx = np.dot(Wax.T,np.multiply(temp_tanh,da_next)) #得到(n_x,m)大小
    da_prev = np.dot(Waa.T,np.multiply(temp_tanh,da_next)) #得到(n_a,m)大小
    dWax = np.dot(np.multiply(temp_tanh,da_next),xt.T) #得到(n_a,n_x)大小
    dWaa = np.dot(np.multiply(temp_tanh,da_next),a_prev.T) #得到(n_a,n_a)大小
    dba = np.sum(np.multiply(temp_tanh,da_next),axis= -1,keepdims= True)

    gradients = {
        'dxt'  : dx,
        'da_prev': da_prev,
        'dWax': dWax,
        'dWaa': dWaa,
        'dba': dba,
    }

    return gradients


# Test OK!
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Wax = np.random.randn(5,3)
# Waa = np.random.randn(5,5)
# Wya = np.random.randn(2,5)
# b = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
#
# a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)
#
# da_next = np.random.randn(5,10)
# gradients = rnn_cell_backward(da_next, cache)
# print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
# print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
# print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
# print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
# print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
# print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
# print("gradients[\"dba\"][4] =", gradients["dba"][4])
# print("gradients[\"dba\"].shape =", gradients["dba"].shape)


#根据单个单元的反向传播，我们来计算整个RNN的反向传播
def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    (caches,x) = caches
    (a1, a0, x1, parameters) = caches[0]  # t=1 时的值

    #根据输入来fetch必要的尺寸
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    #初始化用于存放梯度的dx等必要数据
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        #这个地方的加号 存疑 为什么反向传播的后向梯度是
        #da[::,::,t]加上da_prevt
        # 原因在于  da是一个根据 softmax得到的loss  这个loss的维度是一个(n_a, m, T_x)，
        # 在计算过程中，我们需要同时考虑后向传播上来的da_prevt和这个单词与标准答案之间的loss
        # 因此这里用加号
        gradients = rnn_cell_backward(da[::,::,t]+ da_prevt,caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


# Test OK!
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Wax = np.random.randn(5,3)
# Waa = np.random.randn(5,5)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
# a, y, caches = rnn_forward(x, a0, parameters)
# da = np.random.randn(5, 10, 4)
# gradients = rnn_backward(da, caches)
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)
# print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
# print("gradients[\"da0\"].shape =", gradients["da0"].shape)
# print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
# print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
# print("gradients[\"dba\"][4] =", gradients["dba"][4])
# print("gradients[\"dba\"].shape =", gradients["dba"].shape)

##构建LSTM的反向传播函数
def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the input gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    #从输入参数得到必要数据
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    #其中ft,it,cct,ot分别代表 遗忘门，更新门，候选cn，输出门的值
    #从输入数据得到必要维度
    n_x,m = xt.shape
    n_a,m = a_next.shape

    #为4个门函数计算各自的梯度
    #首先dot很好理解，这里的ot并不只是输出门，而是将sigmoid也纳入导数计算，本质上应该是
    #d(Wo*Xstack + bo) = da_next * np.tanh(c_next) * ot * (1 - ot)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    #然后是候选cct,遗忘门gate_forget，更新们gate_update的计算
    #这里前往要记住！！！！！   cn和an是同时对 候选cct,遗忘门gate_forget，更新门gate_update起作用的
    #因此他们的作用是共同的，为了计算我们需要将从dc_next得到的梯度加上da_next处得到的梯度相加，就是总的这个门函数的梯度
    #同样这里的it,ft并不仅仅只是门 同样将sigmoid也纳入了导数计算  cct纳入了tanh的导数计算
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    #做一个拆解示例：
    #从Cnext处得到梯度1  cn = Gate_forget*Cprev + Gate_update*Cn_candidate
    #dcct_from_cn = dc_next * gate_update * (1-np.square(Cn_candidate))
    #从anext处得到梯度2 an = Gate_output * tanh(Gate_forget*Cprev + Gate_update*Cn_candidate )
    #dcct_from_an = Gate_output*(1-np.square(np.square(c_next)))*da_next*(Gate_output * tanh(Gate_forget*Cprev + Gate_update*Cn_candidate ))
    #dcct = dcct_from_cn + dcct_from_an
    #后面二者的拆解和这个同理
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot *(1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)


    # dgate_out = da_next * np.tanh(c_next) * ot * (1 - ot)
    # dgate_update = (dc_next  + ot * (1 - np.square(np.tanh(c_next)))  * da_next)  * cct * it * (1 - it)
    #这个地方给的导数并不是update门本身，而是  Wf*stack_input + bf
    #其中的(dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) 这部分：将update门看成一个整体，对门求导
        #这其中的dc_next*cct是因为 遗忘门*C_prev + 更新门*Cct  = Ct
        #Ct又出现在  输出门 * tanh(c_next) = a_next

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    dWf = np.dot(dft,np.concatenate((a_prev, xt), axis=0).T)
    dWi = np.dot(dit,np.concatenate((a_prev, xt), axis=0).T)
    dWc = np.dot(dcct,np.concatenate((a_prev, xt), axis=0).T)
    dWo = np.dot(dot,np.concatenate((a_prev, xt), axis=0).T)
    dbf = np.sum(dft, axis=1 ,keepdims = True)
    dbi = np.sum(dit, axis=1, keepdims = True)
    dbc = np.sum(dcct, axis=1,  keepdims = True)
    dbo = np.sum(dot, axis=1, keepdims = True)

    da_prev = np.dot(parameters['Wf'][:,:n_a].T,dft)+np.dot(parameters['Wi'][:,:n_a].T,dit)+np.dot(parameters['Wc'][:,:n_a].T,dcct)+np.dot(parameters['Wo'][:,:n_a].T,dot)
    dc_prev = dc_next*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next
    dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot)
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients

# #Test OK
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
#
# da_next = np.random.randn(5,10)
# dc_next = np.random.randn(5,10)
# gradients = lstm_cell_backward(da_next, dc_next, cache)
# print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
# print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
# print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
# print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
# print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
# print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
# print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
# print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
# print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
# print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
# print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
# print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
# print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
# print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
# print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
# print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
# print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
# print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
# print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
# print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
# print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
# print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

#根据单个的lstm反向传播 来计算整个的反向传播
def lstm_backward(da, caches):
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        #注意到这里只有da使用了 da+da_prevt 而 dc_prevt没有
        #原因在于  da是一个根据 softmax得到的loss  这个loss的维度是一个(n_a, m, T_x)，
        #在计算过程中，我们需要同时考虑后向传播上来的da_prevt和这个单词与标准答案之间的loss
        #因此这里用加号
        #但是对于dc_prevt，这是一个隐藏的信息流，不会外露在网络结构之外，因此
        #这里不使用加号
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']


    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


# Test OK！
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y, c, caches = lstm_forward(x, a0, parameters)
#
# da = np.random.randn(5, 10, 4)
# gradients = lstm_backward(da, caches)
#
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)
# print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
# print("gradients[\"da0\"].shape =", gradients["da0"].shape)
# print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
# print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
# print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
# print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
# print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
# print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
# print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
# print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
# print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
# print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
# print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
# print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
# print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
# print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
# print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
# print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

















##


# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:11:59 2020
@Discription:本次作业只用numpy构建了卷积和池化操作，并实现了这两个操作的反向传播。其中正向操作比较简单，反向传播非常复杂，
但是由于卷积本质上为乘法和加法，因此我们站在后向传递上来的张量上，每个点每个点的将dZ分配到原来的输入上去
池化本质上是对梯度的分配操作，因此我们站在后向传递上来的张量上，按照既定的模式，或按照最大值，或按照平均来将dA分配到原来的输入上去
@author: Netfather
@Last Modified data: 2021年1月19日
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#%%卷积神经网络
#1.0填充
#优势： 允许使用CONV层而不必缩小其高度和宽度。这对于构建更深的网络很重要，因为高度/宽度会随着更深的层而缩小。
#       一个重要、特殊的例子是"same"卷积，其中高度/宽度在一层之后被精确保留。
#       有助于我们将更多信息保留在图像边缘。如果不进行填充，下一层的一部分值将会受到图像边缘像素的干扰。

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    #实现针对X输入的零填充
    #对一个输入的m个图片的数据集或者激活层，在周围0填充指定数量的像素
    
    X_out = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values=0)
    
    return X_out

# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print ("x.shape =", x.shape)
# print ("x_pad.shape =", x_pad.shape)
# print ("x[1,1] =", x[1,1])
# print ("x_pad[1,1] =", x_pad[1,1])

# fig, axarr = plt.subplots(1, 2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_pad')
# axarr[1].imshow(x_pad[0,:,:,0])
#Test OK

#2.卷积的单个步骤
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    #实现一步的卷积操作
    
    return np.sum(np.multiply(a_slice_prev,W)+b)

# np.random.seed(1)
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)

# Z = conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)
#Test OK!

#3.正向传播
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    #从上一层的激活输出中得到上一层的尺寸
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    
    #从参数中得到这一层的卷积核大小等尺寸 
    (f,f,n_C_prev,n_C) = W.shape #由于是对所有的卷积，因此n_C_prev是包含在内的
    
    #从超参数词典中得到填充大小和步长
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    #便于检查 我们先用公式计算得到的下一层的特征图尺寸
    n_H = 1+ int((n_H_prev + 2 * pad - f) / stride)
    n_W = 1+ int((n_W_prev + 2 * pad - f) / stride)
    Z = np.zeros((m,n_H,n_W,n_C))

    #运算第一步零填充
    A_prev_pad = zero_pad(A_prev, pad)  
    
    #使用单步卷积进行运算
    for i in range(m):
        #对每张图片进行计算
        for j in range(n_C):
        #对这一层的卷积核每一个卷积都进行计算   
        #通过如上两个参数，我们可以定义完成卷积参数的切片，以及图片的切片
        #下面两个参数用于定位图片的位置。
        #其中这里选择的位置定义为  k，n 对应于卷积计算完成后的下标位置
            for k in range(n_H):
            
                for n in range(n_W):
                    #对单步卷积进行切片
                    a_slice_prev = A_prev_pad[i,k*stride:k*stride+f,n*stride:n*stride+f,:]
                    b_slice = b[:,:,:,j]
                    W_slice = W[:,:,:,j]
                    assert(a_slice_prev.shape == W_slice.shape)
                    Z[i,k,n,j] = conv_single_step(a_slice_prev, W_slice, b_slice)
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z,cache

# np.random.seed(1)
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 2,
#                "stride": 1}

# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =", np.mean(Z))
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
#Test OK!

#%%池化计算

#1.正向池化
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    #实现指定模式下的池化计算
    
    #从输入参数中还原必要的数据
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    
    f = hparameters['f']
    stride = hparameters['stride']
    
    #定义完成输出的尺寸
    n_H = 1+ int((n_H_prev - f) / stride )
    n_W = 1 + int((n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m,n_H,n_W,n_C_prev))
    
    #完成池化计算
    for i in range(m):
        for j in range(n_C_prev):
            #如下是图片尺寸定义
            for k in range(n_H):
                for n in range(n_W):
                    if mode =="max":
                        A[i,k,n,j] = np.max(A_prev[i,k*stride:k*stride+f,n*stride:n*stride+f,j])
                    if mode == "average":
                        A[i,k,n,j] = np.mean(A_prev[i,k*stride:k*stride+f,n*stride:n*stride+f,j])
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache

# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"stride" : 1, "f": 4}

# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A =", A)
# print()
# A, cache = pool_forward(A_prev, hparameters, mode = "average")
# print("mode = average")
# print("A =", A)
#Test OK!

#%%卷积层的反向传播
#卷积操作中的所有操作都是乘法和加法，某种程度中，这些运算统统都是可以拆分成单独的
#求导来进行计算的
#因此对于一个卷积层的反向传播，我们只需要获得对应的后向传播上来的dZ
#然后分块的进行累加求和 得到我们想得到的dA，dW,db

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    #首先我们需要从 dZ和cache中还原所有所需要的参数
    #1.确认从后向返回的参数 m为图像个数，n_H,n_W图像尺寸,n_C提取特征数
    (m,n_H,n_W,n_C) = dZ.shape
    #2.从cache中得到正向运算中计算得到的上层激活图，参数，偏置，以及超参数
    (A_prev, W, b, hparameters) = cache
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    #3.根据参数大小 初始化对应的梯度矩阵
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))
    
    #pad过程
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    #4.开始计算 dA
    #根据卷积的公式，我们可以知道，对于每一片分片的slice
    # 我们有 Z = W * A_prev + b 然后外层是一个大求和
    # 那么反向传播中 dZ 对 dA_prev求反向传播，就应当是 W 然后外层是一个大求和
    # 站在这一层的激活图Z的角度看，第一个点，对应一个完整的卷积核
    # 将第一个点元素各自乘上卷积核的参数
    # 就对应到上一层对应卷积核大小的A_prev
    #如下循环实际完成了对于Z的每一个元素的遍历，也相当于是对上一层所有参数的一个遍历。
    for i in range(m):
        for j in range(n_C):
            #上面定位是哪副图片的哪个通道
            for k in range(n_H):
                for n in range(n_W):
                    #切片
                    vert_start = n*stride
                    vert_end = vert_start + f
                    horiz_start = k*stride
                    horiz_end = horiz_start + f
                    dA_prev_pad[i,horiz_start:horiz_end,vert_start:vert_end,:] += W[:,:,:,j] * dZ[i, k, n, j]
                    #那么dW 就是站在dz的角度对所有参数中A_prev进行累加求和计算
                    dW[:,:,:,j] += A_prev_pad[i,horiz_start:horiz_end,vert_start:vert_end,:] * dZ[i, k, n, j]
                    db[:,:,:,j] += dZ[i, k, n, j]
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

#如果上面的代码看不清楚，可以看下面的吴恩达作业版
# def conv_backward(dZ, cache):
#     """
#     Implement the backward propagation for a convolution function
    
#     Arguments:
#     dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
#     cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
#     Returns:
#     dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
#                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     dW -- gradient of the cost with respect to the weights of the conv layer (W)
#           numpy array of shape (f, f, n_C_prev, n_C)
#     db -- gradient of the cost with respect to the biases of the conv layer (b)
#           numpy array of shape (1, 1, 1, n_C)
#     """
    
#     ### START CODE HERE ###
#     # Retrieve information from "cache"
#     (A_prev, W, b, hparameters) = cache
    
#     # Retrieve dimensions from A_prev's shape
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
#     # Retrieve dimensions from W's shape
#     (f, f, n_C_prev, n_C) = W.shape
    
#     # Retrieve information from "hparameters"
#     stride = hparameters['stride']
#     pad = hparameters['pad']
    
#     # Retrieve dimensions from dZ's shape
#     (m, n_H, n_W, n_C) = dZ.shape
    
#     # Initialize dA_prev, dW, db with the correct shapes
#     dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
#     dW = np.zeros((f, f, n_C_prev, n_C))
#     db = np.zeros((1, 1, 1, n_C))

#     # Pad A_prev and dA_prev
#     A_prev_pad = zero_pad(A_prev, pad)
#     dA_prev_pad = zero_pad(dA_prev, pad)
    
#     for i in range(m):                       # loop over the training examples
        
#         # select ith training example from A_prev_pad and dA_prev_pad
#         a_prev_pad = A_prev_pad[i]
#         da_prev_pad = dA_prev_pad[i]
        
#         for h in range(n_H):                   # loop over vertical axis of the output volume
#             for w in range(n_W):               # loop over horizontal axis of the output volume
#                 for c in range(n_C):           # loop over the channels of the output volume
                    
#                     # Find the corners of the current "slice"
#                     vert_start = h * stride
#                     vert_end = vert_start + f
#                     horiz_start = w * stride
#                     horiz_end = horiz_start + f
                    
#                     # Use the corners to define the slice from a_prev_pad
#                     a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

#                     # Update gradients for the window and the filter's parameters using the code formulas given above
#                     da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
#                     dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
#                     db[:,:,:,c] += dZ[i, h, w, c]
                    
#         # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
#         dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
#     ### END CODE HERE ###
    
#     # Making sure your output shape is correct
#     assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
#     return dA_prev, dW, db

# np.random.seed(1)
# dA, dW, db = conv_backward(Z, cache_conv)
# print("dA_mean =", np.mean(dA))
# print("dW_mean =", np.mean(dW))
# print("db_mean =", np.mean(db))
#Test OK!

#%%池化层反向传播
#为了方便最大池化层的反向传播，我们需要构建一个掩码
#池化层的梯度本质上就是只有最大位置的地方 梯度为1  其他所有地方梯度为0

#构建一个掩码函数，这个函数会返回一个最大位置为1  其余位置为0的矩阵
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = np.where(x == x.max(),1,0)
    
    return mask

# np.random.seed(1)
# x = np.random.randn(2,3)
# mask = create_mask_from_window(x)
# print('x = ', x)
# print("mask = ", mask)
#Test OK!


#构建一个适用于平均池化的掩码函数，这个函数会将每个位置的数字，按照shape来还原成平均
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_H,n_W) = shape
    
    a = np.ones(shape) * dz / (n_H * n_W)
    
    return a

# a = distribute_value(2, (2,2))
# print('distributed value =', a)
#Test OK!

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    #根据输入参数还原必要参数
    (m,n_H,n_W,n_C) = dA.shape
    (A_prev, hparameters) = cache
    f = hparameters['f']
    stride = hparameters['stride']
    
    #确认参数以及输出大小
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    
    dA_prev = np.zeros_like(A_prev,dtype = float)
    
    for i in range(m):
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    #站在dA的角度 进行投影每一个A中的元素 对应前向dA的slice范围为
                    heriz_start = h*stride
                    heriz_end = heriz_start + f
                    vert_start = w * stride
                    vert_end = vert_start + f
                    a_prev_slice = A_prev[i,heriz_start:heriz_end,vert_start:vert_end,c]
                    if mode == "max": 
                        dA_prev[i,heriz_start:heriz_end,vert_start:vert_end,c] += create_mask_from_window(a_prev_slice) * dA[i,h,w,c]          
                    elif mode == "average":
                        dA_prev[i,heriz_start:heriz_end,vert_start:vert_end,c] += distribute_value(dA[i,h,w,c],(f,f))
                        
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev

# np.random.seed(1)
# A_prev = np.random.randn(5, 5, 3, 2)
# hparameters = {"stride" : 1, "f": 2}
# A, cache = pool_forward(A_prev, hparameters)
# dA = np.random.randn(5, 4, 2, 2)

# dA_prev = pool_backward(dA, cache, mode = "max")
# print("mode = max")
# print('mean of dA = ', np.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1,1])  
# print()
# dA_prev = pool_backward(dA, cache, mode = "average")
# print("mode = average")
# print('mean of dA = ', np.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1,1]) 

#TEST OK!
                        
                    
                    
    
    
    
    
    


              
    
    
                    
                    
    
    
    
                    
    
                    
                    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
#
#@File:  Swin_Transformer.py
#  Swin_Transformer 的源码解析
#@Time:  Created by Jiazhe Wang on 2023-05-20 16:37:02.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: 重点观察一下 源码中的 相对位置编码 和 shift windows 是如何实现的 重点


import timm
import torch
import math
import numpy as np
import cv2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
modles_name = timm.list_models(filter="*swin*")

# print(modles_name)

# model =  timm.models.swin_base_patch4_window7_224


# 1. 关于相对位置索引  这里是官方的代码 我们可以查看
def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    # 按步骤展示一下  到底是如何 获得 相对位置索引的
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    print("*" * 20)
    print(coords)
    # tensor([[[0, 0, 0],  # 按照window的尺寸  按照行列 分别标记每一行的 顺序  第0行全行为0  第一行全行为1 诸如此类
    #      [1, 1, 1],
    #      [2, 2, 2]],

    #     [[0, 1, 2],
    #      [0, 1, 2],
    #      [0, 1, 2]]])
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    print("*" * 20)
    print(coords_flatten)
    print(coords_flatten.size()) # torch.Size([2, 9])
    # tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
    #     [0, 1, 2, 0, 1, 2, 0, 1, 2]])  # 将获得的 这个 行列维度矩阵展平
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    print("*" * 20)
    print(relative_coords)
    # 关键的来了  将对应的
    # 1. 首先 上文可以知道  我们获得的只是一个2维的矩阵
    #    所以要获得行列关系，我们首先要做 广播  所以这里的None 是将2 维的  信息扩展到  行的维度
    # 2. 然后再做相减  获得各个坐标自己的 相对位置信息
    #    其中 0,:: 维度为  窗口上每个点的行下标 与 其他所有点的行下标的相对索引     1,:: 维度为 窗口上每个点的列下标 与 其他所有点的列下标的相对索引
    #tensor([[[ 0,  0,  0, -1, -1, -1, -2, -2, -2],    结合来看 第0个位置 为 (0,0) 当以词为原点 所有 window内其他点的索引顺次为 (0,-1) (0,-2) (-1,0) (-1,-2) 等等
        #  [ 0,  0,  0, -1, -1, -1, -2, -2, -2],              当第1个位置为 (0,0) 当以此为源点 所有 window内其他点的索引顺次为 (0,1) (0,,0) (0,-1) (-1,1) 等等
        #  [ 0,  0,  0, -1, -1, -1, -2, -2, -2],    
        #  [ 1,  1,  1,  0,  0,  0, -1, -1, -1],
        #  [ 1,  1,  1,  0,  0,  0, -1, -1, -1],
        #  [ 1,  1,  1,  0,  0,  0, -1, -1, -1],
        #  [ 2,  2,  2,  1,  1,  1,  0,  0,  0],             当第6个位置为 (0,0) 再3x3中 此时(2,1) 则 原点索引为 (2,0) (2,-1) (2,-2) 等等
        #  [ 2,  2,  2,  1,  1,  1,  0,  0,  0],
        #  [ 2,  2,  2,  1,  1,  1,  0,  0,  0]],
# 这个表中 每一行表示 展平之后  你认定原窗口中对应的哪个位置为 原点位置  在这种情况下 原窗口每个位置的下标
        # [[ 0, -1, -2,  0, -1, -2,  0, -1, -2],
        #  [ 1,  0, -1,  1,  0, -1,  1,  0, -1],
        #  [ 2,  1,  0,  2,  1,  0,  2,  1,  0],
        #  [ 0, -1, -2,  0, -1, -2,  0, -1, -2],
        #  [ 1,  0, -1,  1,  0, -1,  1,  0, -1],
        #  [ 2,  1,  0,  2,  1,  0,  2,  1,  0],
        #  [ 0, -1, -2,  0, -1, -2,  0, -1, -2],
        #  [ 1,  0, -1,  1,  0, -1,  1,  0, -1],
        #  [ 2,  1,  0,  2,  1,  0,  2,  1,  0]]])

    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2  # 注意 这里的 contiguous 是为了将转置维度之后的 内存存储也变得连续 加快训练速度
    print("*" * 20)
    print(relative_coords.max())
    print(relative_coords.min())
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    print("*" * 20) # 消除 负数
    print(relative_coords.max())
    print(relative_coords.min())
    relative_coords[:, :, 1] += win_w - 1 
    print("*" * 20)# 消除负数
    print(relative_coords.max())
    print(relative_coords.min())
    relative_coords[:, :, 0] *= 2 * win_w - 1
    # 这里 其实就是做一个 将行坐标放大的操作  这个操作确保了 经过sum之后 索引表中每个地方的坐标各不相同
    # 例如  如果不做这一步 那么 原始行号3和列号1 的位置
    # 和 原始行号1 和列好3 的位置  指向的索引值是相同的   这是不允许的
    # 因此 通过这样的操作 确保了 每个索引值各不相同
    # 会映射到一个相同的位置
    print("")
    print("*" * 20)
    print(relative_coords.shape)
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

# 首先明确一点  相对位置索引是作用于  QKT之后的  这时候的维度 如果window size 是 7  那么 此时的矩阵维度就是 49 * 49
# 其次 我们举一个 3x3 window的例子
print(get_relative_position_index(3,3))

# 如果不加入  relative_coords[:, :, 0] *= 2 * win_w - 1
# 输出的结果为
# tensor([[4, 3, 2, 3, 2, 1, 2, 1, 0],
        # [5, 4, 3, 4, 3, 2, 3, 2, 1],
        # [6, 5, 4, 5, 4, 3, 4, 3, 2],
        # [5, 4, 3, 4, 3, 2, 3, 2, 1],
        # [6, 5, 4, 5, 4, 3, 4, 3, 2],
        # [7, 6, 5, 6, 5, 4, 5, 4, 3],
        # [6, 5, 4, 5, 4, 3, 4, 3, 2],
        # [7, 6, 5, 6, 5, 4, 5, 4, 3],
        # [8, 7, 6, 7, 6, 5, 6, 5, 4]])
# 很明显 结果没有独一性  同一行 的相对位置索引能出现相同的情况 

# 2. 关于 mask的获得  我在源码中加入了 可视化部分 将mask的变化输出为具体的图片

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def num_to_rgb(val, max_val=9):
    i = (val * 255 / max_val)
    r = round(math.sin(0.024 * i + 0) * 127 + 128)
    g = round(math.sin(0.024 * i + 2) * 127 + 128)
    b = round(math.sin(0.024 * i + 4) * 127 + 128)
    return (r,g,b)

def single_bit_numpy_to_rgb(matrix, max_val=9):
    i = (matrix * 255 / max_val)
    r = np.round(np.sin(0.024 * i + 0) * 127 + 128)
    g = np.round(np.sin(0.024 * i + 2) * 127 + 128)
    b = np.round(np.sin(0.024 * i + 4) * 127 + 128)
    return np.stack(arrays= (b,g,r),axis= 2)


window_size = 7
H, W = 224,224
shift_size = window_size // 2
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1  整幅图是黑的
cnt = 0
for h in (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None)):
    for w in (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None)):
        img_mask[:, h, w, :] = cnt
        cnt += 1
print(img_mask.size())
img_temp = img_mask.numpy()
img_temp = np.squeeze(img_temp,axis= (0,3))
print(img_temp.shape)
print(img_temp.max())
print(img_temp.min())
img_temp = single_bit_numpy_to_rgb(img_temp)
print(img_temp.shape)
cv2.imwrite(filename= "img_mask.png", img=img_temp)

mask_windows = window_partition(img_mask, window_size)  # num_win, window_size, window_size, 1

# 这里取出最后一个窗口 也就是 完全各不相同窗口 来看
print(mask_windows.size())
mask_temp = mask_windows[-1,::,::,::].squeeze().numpy()
print(mask_temp.shape)
mask_temp = single_bit_numpy_to_rgb(mask_temp)
cv2.imwrite(filename= "mask_temp_last_one.png", img=mask_temp)


mask_windows = mask_windows.view(-1, window_size * window_size)
print(mask_windows.size())
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 这一步是将49 49 窗口彼此相减
# 也就是说如果原本窗口是连续的  那么 它对应的 mask就为0  如果之前彼此不连续 也就是位置无关 则为非0值
print(attn_mask.size())
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
print(attn_mask.size())



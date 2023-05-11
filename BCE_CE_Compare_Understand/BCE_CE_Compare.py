# -*- coding: utf-8 -*-
#
#@File:  BCE_CE_Compare.py
#  BCE_CE_Compare
#@Time:  Created by Jiazhe Wang on 2023-05-11 17:27:30.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: 
# #2023年5月11日 为了准备面试 此文件用于解释 BCE 二进制交叉熵和 CE 多分类交叉熵之间的区别


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def check_BCE_losses_run():
    print("测试 CE 与 BCE loss的区别")
    label = torch.tensor( data= [
        [[1.0,0.0,1.0,1.0],
        [0.0,0.0,1.0,0.0],
        [1.0,0.0,1.0,0.0]],

        [[1.0,0.0,1.0,1.0],
        [0.0,0.0,1.0,0.0],
        [1.0,0.0,1.0,0.0]]
    ])
    print(label.size())  # 假设有一个 batchsize 为2 的  分辨率为  3*4 的 标签为 0 1 的label   假设这是个二分类问题
    preds = torch.tensor( data = [
        [[0.5, 0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5]],
         [[0.5, 0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5]]
    ])
    print( preds.size())  # 对应于上面的输入 则输出为这个
    BCE_LOSS = F.binary_cross_entropy_with_logits(preds, label,)
    print("默认的BCE LOSS 输出是对所有样本，包括正负样本求平均{}".format(BCE_LOSS))
    # print(BCE_LOSS)  # 0.7241 

    BCE_LOSS = F.binary_cross_entropy_with_logits(preds, label,size_average= False)
    print("指定BCE LOSS 输出只求和，不做平均{}".format(BCE_LOSS))
    # print(BCE_LOSS)  # 17.3778

    BCE_LOSS = F.binary_cross_entropy_with_logits(preds, label, reduction = "none")
    print("指定BCE LOSS 不做任何更改，只输出单个样本的值{} ".format(BCE_LOSS))
    # tensor([[[0.4741, 0.9741, 0.4741, 0.4741],
    #      [0.9741, 0.9741, 0.4741, 0.9741],
    #      [0.4741, 0.9741, 0.4741, 0.9741]],

    #     [[0.4741, 0.9741, 0.4741, 0.4741],
    #      [0.9741, 0.9741, 0.4741, 0.9741],
    #      [0.4741, 0.9741, 0.4741, 0.9741]]])

    # 如上的参数是如何来的讷

def main():
    check_BCE_losses_run()


if __name__ == '__main__':
    main()
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

def combine_CE_BCE_losses_run():
    print("联合测试 BCE 和 CE loss 的细节")
    label = torch.tensor (data = [0.0,1.0,1.0,1.0])
    preds = torch.tensor (data = [0.9,0.4,0.6,0.7])

    BCE_LOSS = F.binary_cross_entropy(preds,label,reduction= "none")
    print("BCE_LOSS的结果如下{}".format(BCE_LOSS)) # 0.9658081531524658
    # 这里假设输入的是sigmoid 

    # 由于 CE 是针对多分类问题的 所以 我们需要按照 格式来组织数据
    # 对于输入数据 是一个 Batch = 1  Class为2 的数据 分辨率为 1*4的数据
    preds = torch.tensor(data =[  # Batch
        [ 
        [ [0.9,0.6,0.6,0.7] ],
        [ [0.1,0.4,0.4,0.3] ] 
        ] ] )
    print(preds.size())
    label = torch.tensor(data = [[[0,1,1,1]]])  # 标签是一个  二分类的标签  大小为 1* 1* 4
    print(label.size())
    CE_LOSS = F.cross_entropy(preds,label)
    print("CE_LOSS的结果如下{}".format(CE_LOSS)) # 0.5938504934310913
    # 会报出界的原因就在于 CE 的类别 最少必须为2分类
    # 否则标签就没有意义  在这种情况下  二者并不等价 而且 cross_entropy 中默认自带  softmax的 所以结果并不相等
    # 如果换成NLLloss
    preds = torch.log(preds) # 手动模拟 ln
    NLL_LOSS = F.nll_loss(preds,label,reduction= "none")  
    print("NLL_LOSS的结果如下{}".format(NLL_LOSS)) #
    # 如下说明 为什么  不带logits的BCE  为什么和 简单二分类的结果不同 
    # 原因在于  
    # BCE 每个点都会求 他的负样本结果  
    # 对于第一个位置  -(1 * ln(0.1)) = 2.30
    # CE 每个点只会计算正确的类别
    # 对于第一个位置  - (1 * ln(0.9)) = 0.10536
    # 如果要二者相等那就必须将  1位置标签的 设置成BCE loss的 输入 此时二者就是完全等价的
    # 当然输入要保证是sigmoid 和  1- sigmoid 来模拟 softmax的输入
    preds = torch.tensor(data =[  # Batch
        [ 
        [ [0.1,0.6,0.4,0.3] ] ,
        [ [0.9,0.4,0.6,0.7] ]
        ] ] )
    preds = torch.log(preds) # 手动模拟 ln
    NLL_LOSS = F.nll_loss(preds,label,reduction= "none")  
    print("NLL_LOSS的结果如下{}".format(NLL_LOSS)) # 

def check_CE_losses_run():
    print("测试 CE loss的一些细节")
    label = torch.tensor( data= [
        [   # 这个括号表示 batchsize的大小
        [[1.0,0.0,1.0,0.0],
        [1.0,1.0,0.0,1.0]],  # 类别0 标签

        [[0.0,0.0,0.0,1.0],
        [0.0,0.0,0.0,0.0]], # 类别1 标签

        [[0.0,1.0,0.0,0.0],
        [0.0,0.0,1.0,0.0]] # 类别2 标签

        ],
        [   # 这个括号表示 batchsize的大小
        [[1.0,0.0,1.0,0.0],
        [1.0,1.0,0.0,1.0]],  # 类别0 标签

        [[0.0,0.0,0.0,1.0],
        [0.0,0.0,0.0,0.0]], # 类别1 标签

        [[0.0,1.0,0.0,0.0],
        [0.0,0.0,1.0,0.0]] # 类别2 标签

        ]
    ])
    print(label.size())  # 假设有一个 batchsize 为2 的 类别为3 分辨率为  3*2 的 标签为 0 1 的label   假设这是个二分类问题
    preds = torch.tensor( data = [
        [ # 这个括号表示 batchsize的大小

        [[0.8, 0.2,0.6,0.5],
        [0.5,0.5,0.5,0.5]],  # 类别0 标签

         [[0.4, 0.3,0.2,0.1],
        [0.5,0.5,0.5,0.5]],  # 类别1 标签

         [[0.1, 0.2,0.3,0.4],
        [0.5,0.5,0.5,0.5]], # 类别2 标签
        ],

        [
        [[0.8, 0.2,0.6,0.5],
        [0.5,0.5,0.5,0.5]],

         [[0.5, 0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5]],

         [[0.5, 0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5]],
        ]

    ]) # 假设这里的结果已经经过了 sigmoid 函数， 已经给出了每个标签预测的概率
    print( preds.size())  # 对应于上面的输入 则输出为这个

    # 由于使用的版本是 pytorch1.8.0 在这个版本中  preds 和 label并不是完全对应的关系
    # 为了加快运算速度 这里的 label 使用并不是one-hot编码 而是整数编码
    # 而是 使用 0代表0类别(通常代表背景)，1代表1类别，2代表2类别 以此类推， C 代表第C个类别
    # 注意到对于图像分割  背景也是作为一个单独的类加入的 因此 如果有i个物品要预测  那么对应的类别就是 i + 1
    # 因此  对于给定的 one-hot矩阵 他必须是互斥的  也就是在分类上看 他必须 要么是0  要么是1  不存在全0的情况
    # 然后对 N , H , W 维度的矩阵填数，得到最终的 矩阵
    # 这里我们使用额外的函数将 one-hot编码的  B,C,H,W
    # 转换为  B,H,W 的  填写数字范围为  [0,2] 的数字  其中0表示背景 为第一个类别
    cross_entropy_labels = torch.argmax(label, dim=1)
    print(cross_entropy_labels)
    print(cross_entropy_labels.size())  # 当时这种方法 并不能判断 背景和 0分类的区别  
    # 所以 背景类 依然需要单独 在第一个维度列出来 因此这里的  0 表示的就是背景 1表示第一类 2 表示第二类
    # 总共是3分类问题  我们可以使用 one_hot 将 整数类型的 结果还原得到
    # one_hot_labels = F.one_hot(cross_entropy_labels)
    # print(one_hot_labels.size())
    # one_hot_labels = one_hot_labels.permute(0,3,1,2)
    # print(one_hot_labels)  # 可以发现 这里的one_hot_labels 就完全等于一开始输入的 labels
    

    print( "将label的 class维度提前 得到每个类下 每个batch的 矩阵尺寸为{}".format(label.size()))
    CE_LOSS = F.cross_entropy(preds, cross_entropy_labels) # 1.0759093761444092

    print("默认的CE LOSS 输出是对所有样本，包括正负样本求平均{}".format(CE_LOSS))
    # 如何得到的呢？   
    # 1. 首先取出 preds中的  Batch 0 的样本
        #     [[0.8, 0.2,0.6,0.5],
        # [0.5,0.5,0.5,0.5]],  # 类别0 标签

        #  [[0.4, 0.3,0.2,0.1],
        # [0.5,0.5,0.5,0.5]],  # 类别1 标签

        #  [[0.1, 0.2,0.3,0.4],
        # [0.5,0.5,0.5,0.5]], # 类别2 标签
        # ],
    # 2. 由于多分类问题的互斥性  会首先在C 维度上做 softmax 得到如下结果
    # []
    soft_tensor = torch.tensor(data = [[[0.8, 0.2,0.6,0.5],
    [0.5,0.5,0.5,0.5]],  # 类别0 标签

    [[0.4, 0.3,0.2,0.1],
    [0.5,0.5,0.5,0.5]],  # 类别1 标签

    [[0.1, 0.2,0.3,0.4],
    [0.5,0.5,0.5,0.5]], # 类别2 标签
    ])
    # print(soft_tensor.size())
    soft_tensor = F.softmax(soft_tensor,dim = 0)
    print("将原始输入 经过softmax 得到类别维度归一化的结果.{}".format(soft_tensor)) # 1.0606331825256348

    # 3. 然后再通用公式  每一个样本只计算 预测正确类别的结果 即
    # - 1/ N * Sum(yi * log (pi))
    # 对于 一副 (2,4)的图像， 
    # 第一个坐标位置的结果为  [0.4615,0.3093,0.2292] 对应的标签为 [1,0,0]  1* ln(0.4615) = 0.7732
    # 第二个坐标位置的结果为  [0.3220,0.3559,0.3220] 对应的标签为 [0,0,1]  1* ln(0.3220) = 1.1331
    # 如此就得到了 对应的结果

    CE_LOSS = F.cross_entropy(preds, cross_entropy_labels,reduction = "none")
    print("指定CE LOSS 不做任何更改，只输出单个样本的值{} ".format(CE_LOSS)) 
    # tensor([[[0.7733, 1.1331, 0.8801, 1.3459],
    # [1.0986, 1.0986, 1.0986, 1.0986]],

    # [[0.9089, 1.0083, 1.0331, 1.0986],
    # [1.0986, 1.0986, 1.0986, 1.0986]]]) 

def check_BCE_losses_run():
    print("测试 BCE loss的一些细节")
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
        [[0.8, 0.2,0.6,0.5],
        [0.3,0.6,0.5,0.7],
        [0.5,0.5,0.5,0.5]],
         [[0.5, 0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5]]
    ]) # 假设这里的结果已经经过了 sigmoid 函数， 已经给出了每个标签预测的概率
    print( preds.size())  # 对应于上面的输入 则输出为这个
    BCE_LOSS = F.binary_cross_entropy(preds, label,)
    print("默认的BCE LOSS 输出是对所有样本，包括正负样本求平均{}".format(BCE_LOSS))
    # print(BCE_LOSS)  # 0.69310.6629458069801331
    ############################################ 这里我们验证一下 BCE loss的公式 ############################################ 
    # 根据note中写的  bce公式应该是  1/N * Sum-([yi * log(pi) + (1-yi) * log(1-pi)] )
    # 所以这里 第一个为  - 1*ln(0.8) - 0 * ln(1 - 0.8) = 0.22314
    #         第二个为  - 0*ln(0.2) - (1 - 0) * ln (1 - 0.8) = 0.22314
    #         第三个为  - 1*ln(0.6) - (1 - 1) * ln (1 - 0.6) = 0.5108
    # 可以明白 每一项 都要同时考虑正负样本的情况。   同时 如果函数中带有with logits 则不需要在网络中加入sigmoid，而是公式中直接帮你计算
    # 贴出两个源地址的链接
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # 可以发现两个计算公式中不同点就是加入了sigmoid

    BCE_LOSS = F.binary_cross_entropy(preds, label,size_average= False)
    print("指定BCE LOSS 输出只求和，不做平均{}".format(BCE_LOSS))
    # print(BCE_LOSS)  #15.910699844360352

    BCE_LOSS = F.binary_cross_entropy(preds, label, reduction = "none")
    print("指定BCE LOSS 不做任何更改，只输出单个样本的值{} ".format(BCE_LOSS))
    # tensor([[[0.2231, 0.2231, 0.5108, 0.6931],
    #  [0.3567, 0.9163, 0.6931, 1.2040],
    #  [0.6931, 0.6931, 0.6931, 0.6931]],

    # [[0.6931, 0.6931, 0.6931, 0.6931],
    #  [0.6931, 0.6931, 0.6931, 0.6931],
    #  [0.6931, 0.6931, 0.6931, 0.6931]]]) 


def main():
    # check_BCE_losses_run()
    check_CE_losses_run()
    # combine_CE_BCE_losses_run()


if __name__ == '__main__':
    main()
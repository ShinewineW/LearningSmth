# -*- coding: utf-8 -*-
#
#@File:  TP_FP_TN_FN_Sen_Spe_IOU.py
#  TP_FP_TN_FN_Sen_Spe_IOU的理解
#@Time:  Created by Jiazhe Wang on 2023-05-12 15:07:48.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#Seg_Confuse_Matrix()
#@Desc: 
# 简单二分类的上述结果过于简单  这里阐述一下在图像分割中 上述结果如何计算
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt

def Seg_Confuse_Matrix():
    print("所有分割问题的精度精算都要经过混淆矩阵，先测试混淆矩阵的概念")
    # 混淆矩阵就是 横轴为 正确类别  纵轴为预测类别 每个像素点按照对应的位置 +1 -1
    # 例如对于一个包含背景的 猫狗猪 的 四分类问题 给定如下 data 和矩阵  假设图片大小为 2 * 3  batchsize 为2
    torch.manual_seed(1234) # 固定seed 确保每次的preds 都一致 方便复现
    preds = torch.randint(size= (2,4,2,3), low = 2, high = 5,dtype= torch.float32)
    # 手动做一下 softmax 得到每一类的概率
    preds_soft_max = torch.softmax(preds, dim = 1)
    # print(preds_soft_max)
    # 根据 preds_soft_max 的结果 得到对应的 preds
    
    # print(preds_soft_max)   # 列出 第0个batch 图片的 结果
    #   [0.1749, 0.1749, 0.1749],
    #   [0.7112, 0.7112, 0.3655]],

    #  [[0.1749, 0.1749, 0.4754],
    #   [0.0963, 0.0963, 0.3655]],

    #  [[0.4754, 0.4754, 0.1749],
    #   [0.0963, 0.0963, 0.1345]],

    #  [[0.1749, 0.1749, 0.1749],
    #   [0.0963, 0.0963, 0.1345]]],
    # 然后串建一个int 类型的 label
    label = torch.tensor(data=
    [
        [
            [2,1,2],
            [0,0,3]
        ],
        [
            [2,1,3],
            [3,2,1]
        ]
    ])
    # print(preds.size())
    print("label的标签大小为{}".format(label.size()))
    # 第二部 将 int类型的 label 和 输出的结果转移为混淆矩阵
    # 由于 pytorch库中不带有混淆矩阵的概念  需要额外安装一个库来实现 以方便验证我理解结果是否正确
    # 这里我们使用 sklearn 中的 confuse matrix
    # 同时为了查看传统的 Image Segmentation 任务 的输出是多少
    # 使用 segmentation-models-pytorch 来查看一个传统的分割任务的输
    # model = smp.DeepLabV3(
    # encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=11,                      # model output channels (number of classes in your dataset)
    # )
    # mask = model(torch.ones([7, 3, 256, 256]))
    # print(mask.size())
    # 通过这个样例可以明白 输出的结果 就是 B,C,H,W的形式
    # 那么首先我们要经过softmax 处理之后根据每个维度最大值来得到这个像素对应的样例
    preds_soft_max = torch.argmax(preds_soft_max, dim=1)
    print("preds的标签大小为{}".format(preds_soft_max))
    # 然后生成混淆矩阵 应该是一个 4 *4 的矩阵大小
    # 1. sklearn metric 实现方法
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # 将torch 中的 tensor张量 装置到 numpy 然后flatten 求出 混淆矩阵
    # https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py
    # 根据网页可以知道  混淆矩阵就是用如下的方式计算的
    label = label.numpy().flatten()
    preds_soft_max = preds_soft_max.numpy().flatten()
    print("True labels are {}".format(label))
    print("Predict labels are {}".format(preds_soft_max))
    cm = confusion_matrix( label, preds_soft_max)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.savefig("./Metric_Evaluation_Understand/disp.png")
    # 然后通过这个混淆矩阵 可以计算某个特定类别的 结果
    # 1. 总准确率
    # print(np.diag(cm)) # 将对角线元素单独拿出来
    acc = np.diag(cm).sum() / cm.sum()  # 对角线为TP TP / 总样本数
    print(acc)
    acc_cls = np.diag(cm) / cm.sum(axis=1) # 给出每个类的预测准确率  这里的 cm.sum 对 axis = 1 求和 其实就是计算每个类所有的预测总数
    # 同样 这里的  cm.sum 对 axis = 0 求和 其实就是计算每个类所有实际正确的总数
    print(acc_cls)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    print(iou)  # 这里给的是每个类的 iou    例如对于类别0的正确像素为2个 预测像素为类别0的有4个
    # 因此 iou =  2 / ( 2 + 4 - 2) 为50%   
    # 如此就得到了 大多数的参数验证


def main():
    Seg_Confuse_Matrix()


if __name__ == '__main__':
    main()


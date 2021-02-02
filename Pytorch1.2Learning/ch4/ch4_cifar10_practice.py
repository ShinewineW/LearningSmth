# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch4_cifar10_practice
# Description:  
# Author:       Administrator
# Date:         2021/2/2
# -------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from  torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
import os
import matplotlib.pyplot as plt

minist_train = datasets.CIFAR10('./data',True,transform = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(5),transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
                    ]),download= False)

# 加载一张照片看一下
plt.clf()
print(len(minist_train))
print(minist_train[0][0].shape,minist_train[0][1])
# 加载出来的数据集每一个会由  train 和 label组成一个元组
# for i in range(40):
#     plt.subplot(10,4,i+1)
#     plt.imshow(torch.squeeze(cifar[i][0]).numpy())
#     plt.title(cifar[i][1])
# plt.show()



##

minist_test = datasets.CIFAR10('./data',False,transform = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(5),transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]),download= False)


batchsz = 512
minist_train,minist_val = torch.utils.data.random_split(minist_train,[40000,10000])


# 从数据集中将数据加载为minist_train
minist_train = DataLoader(minist_train, batch_size=batchsz, shuffle=True)
minist_val = DataLoader(minist_val, batch_size=batchsz, shuffle=True)
minist_test = DataLoader(minist_test, batch_size=batchsz, shuffle=True)

## 2. 构建网络结构

class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(3,6,kernel_size= 5,stride=1,padding=0,padding_mode= 'zero')
        self.pool1 = nn.AvgPool2d(kernel_size= 2,stride= 2,padding= 0)

        self.conv2 = nn.Conv2d(6,16,kernel_size= 5,stride = 1,padding = 0)
        self.pool2 = nn.AvgPool2d(kernel_size= 2,stride= 2,padding= 0)

        self.Liner1 = nn.Linear(16*5*5,120)
        self.Liner2 = nn.Linear(120,84)
        self.Liner3 = nn.Linear(84,10)

    def forword(self,inputs):
        _ = self.conv1(inputs)
        _ = self.pool1(_)

        _ = self.conv2(_)
        _ = self.pool2(_)

        # print(_.shape)
        _ = torch.reshape(_,shape=[-1,16*5*5])
        _ = self.Liner1(_)
        _ = self.Liner2(_)
        outputs = self.Liner3(_)

        return outputs

Mymodel = Lenet5()
# x = torch.randn(size= [3,3,32,32])
# print(Mymodel.forword(x).shape)


## 定义损失率 优化器
cirtion = nn.CrossEntropyLoss()
optimzer = optim.AdamW(Mymodel.parameters() ,lr= 3e-4)

device = torch.device('cuda')

Mymodel.to(device)
cirtion.to(device)

for epoch in range(1):
    Mymodel.train()
    for batchidx,(x,label) in enumerate(minist_train):
        # print(x.shape,label.shape)
        x,label = x.to(device),label.to(device)
        logits = Mymodel.forword(x)
        loss = cirtion(logits,label)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        print("On {} batch: loss {}".format(batchidx,loss.item()))
    Mymodel.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for x, label in minist_val:
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            # [b, 10]
            logits = Mymodel.forword(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            # print(correct)

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)






# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch1_MnistDataSet
# Description:  
# Author:       Administrator
# Date:         2021/1/27
# -------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from  torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt
from visdom import Visdom

# 使用类的特性
# 将你要构建的网络抽象成一个类 从nn.Module中继承类属性和初始化方法
# 实现你这个子类的前向传播函数 或者一些其他的函数
viz = Visdom()
batch_size=200
learning_rate=0.01
epochs=3

viz.line([0.],[0.],win = 'train_loss',opts= dict(title = 'train loss'))  # 初始化 enviromnet 并将window初始化
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',legend=['loss', 'acc.'])) # 初始化
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)
glob_step = 0
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data,target = data.to(device),target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        viz.line([loss.item()],[glob_step],win='train_loss',update='append')
        glob_step += 1

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    viz.line([[test_loss, correct / len(test_loader.dataset)]],
                [glob_step], win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28), win='x',opts=dict(title='x'))
    viz.text(str(pred.detach().cpu().numpy()), win='pred',
                opts=dict(title='pred'))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ch1_LearnierRegression
# Description:  大致描述了一个pytorch架构的网络训练 需要完成的步骤
# Author:       Administrator
# Date:         2021/1/27
# -------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import optim
from  torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt


batch_size=200
learning_rate=0.01
epochs=10

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


w1,b1 = torch.randn(size = [200,784],requires_grad= True),torch.zeros(size = [200,],requires_grad= True)
w2,b2 = torch.randn(size = [200,200],requires_grad= True),torch.zeros(size = [200,],requires_grad= True)
w3,b3 = torch.randn(size = [10,200],requires_grad= True),torch.zeros(size = [10,],requires_grad= True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    x  = x@w1.t() + b1
    x  = F.relu(x)
    x  = x@w2.t() + b2
    x  = F.relu(x)
    x  = x@w3.t() + b3
    x  = F.relu(x)
    return x

optimizer = optim.AdamW([w1,b1,w2,b2,w3,b3],lr = 1e-3)
crition = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):

        data = torch.reshape(data,shape=[-1,28*28])
        logits = forward(data)
        loss = crition(logits,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


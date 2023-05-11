# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:44:05 2020
@Discription: 实现了对于一个简单2D函数的显示以及用tf来进行反向梯度优化
@author: Administrator
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11)** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
X,Y = np.meshgrid(x,y)
print(X.shape,Y.shape)

Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = plt.gca(projection = '3d')
ax.plot_surface(X,Y,Z)
ax.view_init(30,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#%%下面开始优化
x = tf.constant([-4,0],dtype = tf.float32) #设定起始位置
x = tf.Variable(x)

for step in range(200):
    with tf.GradientTape() as tape:
        y = himmelblau(x)
    
    grads = tape.gradient(y,[x])[0]
    # print(grads)
    
    x.assign_sub(0.01*grads)
    
    
    if step % 20 == 0:
        print('step {}: x = {} ,f(x) = {}'.format(step, x.numpy(),y.numpy()))




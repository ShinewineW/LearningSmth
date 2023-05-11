# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:42:40 2020

@author: Administrator
"""




import tensorflow as tf
import timeit

n = 1000000
with tf.device('/cpu:0'):
	cpu_a = tf.random.normal([1, n])
	cpu_b = tf.random.normal([n, 1])
	print(cpu_a.device, cpu_b.device)
	# 创建使用GPU运算的2个矩阵
with tf.device('/gpu:0'):
	gpu_a = tf.random.normal([1, n])
	gpu_b = tf.random.normal([n, 1])
	print(gpu_a.device, gpu_b.device)

def cpu_run():
	with tf.device('/cpu:0'):
         c = tf.matmul(cpu_a, cpu_b)
	return c 

def gpu_run():
	with tf.device('/gpu:0'):
         c = tf.matmul(gpu_a, gpu_b)
	return c 

	# 第一次计算需要热身，避免将初始化阶段时间结算在内
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)
	# 正式计算10次，取平均时间
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)
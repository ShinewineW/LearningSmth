# -*- coding: utf-8 -*-
#
#@File:  mAP_AP.py
#  mAP_AP
#@Time:  Created by Jiazhe Wang on 2023-05-12 16:37:29.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: 如何计算AP 和 mAP 在目标检测任务中
#

# 1. 首先 实例化一个 YOLO模型  然后输入任意一张图
# 2. 查看经过模型后 该图的结果
# 3. 将该结果的true label 和  predict label 作比较  并查看 AP 和 mAP的运作机制
# 使用断点调试来 查看 COCO Evalutor 的计算过程

# 由于相关过程非常复杂 具体请参考 这个博客  写的非常详细
# https://blog.51cto.com/u_15435490/4633871
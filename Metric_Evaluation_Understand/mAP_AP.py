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


# 如下给出一个别人写好的 且完全可以使用的 计算mAP的样例
# https://github.com/Cartucho/mAP#quick-start   这里我们手动计算一遍是否和他给的结果完全一致
# 不表示完整的结果  我们这里仅仅只以  2007_0000027.txt为例

# 不输出动画 试运行一下 如下操作
# cd Metric_Evaluation_Understand/
# python main.py -na


# 可以在output 页面查看 最终输出的结果
# 我们以 book类为例   book 类的 AP 为 50% 手动计算验证是否正确
# <class_name> <left> <top> <right> <bottom> [<difficult>]
# 在 正确选项中  给定的正确候选框有
# book 439 157 556 241
# book 437 246 518 351
# book 515 306 595 375
# book 407 386 531 476
# book 544 419 621 476
# book 609 297 636 392


# 在预测中 给定的候选框 按照置信度从高到底分别有
# <class_name> <confidence> <left> <top> <right> <bottom>
# 1. book 0.619459 413 390 515 459
# 2. book 0.462608 518 314 603 369
# 3. book 0.460851 429 219 528 247
# 4. book 0.382881 403 384 517 461
# 5. book 0.369369 405 429 519 470
# 6. book 0.298196 592 310 634 388
# 7. book 0.272826 433 272 499 341
# 8. book 0.269833 433 260 506 336


# 50.00% = book AP 
#  Precision: ['1.00', '1.00', '0.67', '0.50', '0.40', '0.50', '0.43', '0.50']
#  Recall :['0.17', '0.33', '0.33', '0.33', '0.33', '0.50', '0.50', '0.67']
# 分别按照每个confidence 计算 对应的 Precision 和 Recall曲线
# 预测边界框   Rank   TP   FP   Precison   Recall
#    1.        1      1    0     1/ 1= 1   1/6 = 0.166
#    2.        2      2    0     2/ 2= 1   2/6 = 0.333    # 这里是根据 iou的值来判断的

# 可以发现和微博中说明是完全一致的  因此这就是map的计算方式   



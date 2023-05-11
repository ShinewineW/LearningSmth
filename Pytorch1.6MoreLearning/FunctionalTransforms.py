# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         FunctionalTransforms
# Description:  This file is check the center crop function in VOC2012 dataset
# Author:       Administrator
# Date:         2021/2/8
# -------------------------------------------------------------------------------

import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


## 下面代码块用于进行中心裁剪方法的测试，通过测试可以看出，哪怕是一个不满足512 512 格式的图片，通过center_crop，会在四周填充0
test_image_path = "../data/VOCdevkit/VOC2012/JPEGImages/2007_000129.jpg"

test_image = Image.open(test_image_path)

print(test_image)

output_image = ff.center_crop(test_image,(512,512))

print(output_image)

plt.clf()

plt.subplot(1,2,1)
plt.imshow(test_image)
plt.subplot(1,2,2)
plt.imshow(output_image)

plt.show()


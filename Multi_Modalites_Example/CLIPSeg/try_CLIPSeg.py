# -*- coding: utf-8 -*-
#
#@File:  try_CLIPSeg.py
#  测试 CLIPSeg 模型的具体张量变化  已经 text 到底是如何 与 image结合的
#@Time:  Created by Jiazhe Wang on 2023-05-26 16:55:20.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: Please Fill in this description

from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

url = "https://farm3.staticflickr.com/2141/1876582597_13e6d82673_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a hair drier", "a women", "a blanket", "a window"]
inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
#  [image] * len(texts)  这一条 就暴露了 是怎么计算的
# 关键在于 language 端给与的  类别数量提示。 对于不在提示中的 是无法分类出来的
outputs = model(**inputs)

logits = outputs.logits
print(logits.shape)

# -*- coding: utf-8 -*-
#
#@File:  try_CLIP.py
#  try_CLIP 
#@Time:  Created by Jiazhe Wang on 2023-05-26 15:47:33.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: Please Fill in this description


from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = []
url = "https://farm9.staticflickr.com/8103/8515355202_692e17b43b_z.jpg"
images.append(Image.open(requests.get(url, stream=True).raw))
url = "https://farm4.staticflickr.com/3074/2435951509_b8427a04a5_z.jpg"
images.append(Image.open(requests.get(url, stream=True).raw))
url = "https://farm7.staticflickr.com/6010/5902956520_4287d523e5_z.jpg"
images.append(Image.open(requests.get(url, stream=True).raw))
url = "https://farm4.staticflickr.com/3074/2435951509_b8427a04a5_z.jpg"
images.append(Image.open(requests.get(url, stream=True).raw))
url = "https://farm4.staticflickr.com/3074/2435951509_b8427a04a5_z.jpg"
images.append(Image.open(requests.get(url, stream=True).raw))

inputs = processor(text=["a photo of a cat", "a photo of a dog","a photo of smartphone","a photo of human","a photo of kids"], images=images, return_tensors="pt", padding=True)

outputs = model(**inputs,return_loss = True)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)

# 关于CLIP 源码中的一些特点
# self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
# self.num_positions = self.num_patches + 1
# self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
# self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
# 输入的图片就是 [cls_token , image_patch] + pos_embed  
# 这里的 cls_token 是一个  可以学习的 embed_dim 维度的   image_patch 是用老办法传递过来的  torch_arrange 是固定了一个 具体的参数
# torch.Size([5, 50, 768])
# torch.Size([5, 768])
# torch.Size([5, 768])
# 经过图片特征抽取之后的特征矩阵大小为torch.Size([5, 512])
# torch.Size([5, 512])
# 经过文本特征抽取之后的特征矩阵大小为torch.Size([5, 512])
# 经过点乘之后的从图到文本的相似度矩阵大小为torch.Size([5, 5])
# 经过点乘之后的从文本到图像的相似度矩阵大小为torch.Size([5, 5])
# CLIP的对比学习loss 是怎么利用CE loss的
# tensor([[15.3892, 17.5380, 14.7160, 17.5380, 17.5380],
#         [16.2110, 17.0441, 20.5095, 17.0441, 17.0441],
#         [22.4792, 20.9596, 18.5804, 20.9596, 20.9596],
#         [19.0220, 19.3992, 18.8226, 19.3992, 19.3992],
#         [24.3545, 18.8960, 16.0485, 18.8960, 18.8960]], grad_fn=<MulBackward0>)
# tensor([0, 1, 2, 3, 4])
# CLIP的对比学习loss 是怎么利用CE loss的
# tensor([[15.3892, 16.2110, 22.4792, 19.0220, 24.3545],
#         [17.5380, 17.0441, 20.9596, 19.3992, 18.8960],
#         [14.7160, 20.5095, 18.5804, 18.8226, 16.0485],
#         [17.5380, 17.0441, 20.9596, 19.3992, 18.8960],
#         [17.5380, 17.0441, 20.9596, 19.3992, 18.8960]], grad_fn=<TBackward>)
# tensor([0, 1, 2, 3, 4])
# tensor([[1.1028e-04, 2.5085e-04, 1.3233e-01, 4.1707e-03, 8.6314e-01],
#         [2.3504e-02, 1.4342e-02, 7.1961e-01, 1.5115e-01, 9.1386e-02],
#         [2.2656e-03, 7.4352e-01, 1.0801e-01, 1.3761e-01, 8.5884e-03],
#         [2.3504e-02, 1.4342e-02, 7.1961e-01, 1.5115e-01, 9.1386e-02],
#         [2.3504e-02, 1.4342e-02, 7.1961e-01, 1.5115e-01, 9.1386e-02]],
#        grad_fn=<SoftmaxBackward>)
# 这里应该这么理解   输入的 logtis 是一个  Batch * class * 1 * 1 的输入
# 对于5个  1x1像素的输入 每个像素可以分出个类别
# 然后根据分割问题中的混淆矩阵的概念，   每个像素的标记为自己的类别  所以 按照batchsize   就是 [0,1,2,3,4] CE loss 是这么用的
# CLIP 的亮点主要就是这里 其他的结构都很简单 
# -*- coding: utf-8 -*-
#
#@File:  ViT_Transformer.py
#  ViT_Transformer 源码分析
#@Time:  Created by Jiazhe Wang on 2023-05-17 17:23:16.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: # ViT Transformer 的一些输出维度
#
import timm

model_name = timm.list_models(filter="*vit*",pretrained=True )
# print(model_name)

# 这里我们选用 vit_base_patch32_224  作为讲解
model = timm.models.vit_base_patch32_224_Debug(pretrained= False, DEBUG_MODE= True)  # 通过这种方式查看ViT的源码
# 这里我对timm的 ViT部分源码做了修改  加入了参数DEBUG MODE 这个模式下会显示流过的每一个张量的尺寸

# 目前的ViT为了解决 多通道输入的问题 都是直接使用一个  kernal_size = patch_size的卷积层 来作为输入
# 这里我们观察一下张量流动的方向
import torch

input_image = torch.randn(size=  [7,3,224,224],dtype= torch.float32)  # 输入固定为 224



output_tensor = model(input_image)

print(output_tensor.shape)   # torch.Size([7, 1000])


# ********** 传入模型的定义参数 **********
# {'patch_size': 32, 'embed_dim': 768, 'depth': 12, 'num_heads': 12}
# cls_token的初始化内容位tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        grad_fn=<SliceBackward>)
# pos_embed的初始化内容为tensor([[ 4.1948e-02,  7.7420e-03, -1.5629e-02, -1.2998e-03,  4.4916e-03,
#          -1.7070e-02,  2.2371e-02,  2.9284e-02,  3.8868e-03,  7.6740e-03,
#          -3.3087e-02,  4.7744e-03,  4.9835e-02,  2.3908e-02, -1.3799e-02],
#         [ 2.1827e-02,  2.4587e-02, -1.0690e-02, -2.0586e-02, -1.7162e-02,
#           8.0788e-03, -2.9695e-02, -2.6107e-02,  1.7171e-02,  3.5592e-03,
#           2.7556e-03, -2.9899e-02, -3.3851e-03, -5.4136e-03,  2.3720e-02],
#         [-9.5298e-06,  4.7153e-03, -3.4406e-02,  4.3662e-03, -1.2423e-02,
#          -1.1959e-02,  1.9330e-02, -1.1475e-02, -1.6144e-02,  1.6115e-02,
#          -1.0579e-02, -2.6886e-02, -9.9780e-03, -1.4588e-02, -2.2477e-02]],
#        grad_fn=<SliceBackward>)
# 图片输入模型的形状torch.Size([7, 3, 224, 224])
# 图片经过patch_embed的形状torch.Size([7, 49, 768])
# 位置编码之后的形状torch.Size([7, 50, 768])
# torch.Size([7, 1000])
# (detect3) shinewine@DL-Station:/storage/shinewine/LearningSmth$ /home/shinewine/anaconda3/envs/detect3/bin/python /storage/shinewine/LearningSmth/Transformer_Detailed_Example/VisionInTransformer_model/ViT_Transformer.py
# ********** 传入模型的定义参数 **********
# {'patch_size': 32, 'embed_dim': 768, 'depth': 12, 'num_heads': 12}
# cls_token的初始化内容位tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        grad_fn=<SliceBackward>)
# pos_embed的初始化内容为tensor([[ 2.2045e-02,  1.9564e-02,  2.0462e-02, -2.3565e-02,  5.3251e-03,
#           7.1069e-03, -8.3258e-03, -2.6330e-02,  4.2314e-02, -1.8918e-03,
#           8.9632e-03, -1.8603e-02,  7.1856e-03, -1.0358e-02,  2.9590e-02],
#         [ 6.7600e-03,  2.2548e-03,  5.6944e-03,  1.2706e-02, -1.0999e-02,
#           1.7479e-03, -3.3265e-03, -3.2514e-02,  2.5324e-02,  1.2496e-02,
#          -8.0916e-03,  3.5449e-03, -1.1251e-02, -9.9411e-03,  1.7590e-02],
#         [-2.5869e-02,  1.1028e-02, -2.9242e-04,  2.4205e-03,  1.9789e-03,
#          -3.7071e-02, -1.9341e-02, -2.0099e-02,  1.0606e-03,  1.7295e-02,
#          -6.2557e-02,  1.9395e-02, -3.6850e-05, -4.1084e-02, -1.4943e-02]],
#        grad_fn=<SliceBackward>)
# 图片输入模型的形状torch.Size([7, 3, 224, 224])
# 图片经过patch_embed的形状torch.Size([7, 49, 768])
# 位置编码之后的形状torch.Size([7, 50, 768])
# torch.Size([7, 1000])

# 通过上面的测试输出 我们可以非常明显的知道   cls_token是全0初始化  pos_embed 不需要额外的公式维护 而是模型自己学习的一个变量。



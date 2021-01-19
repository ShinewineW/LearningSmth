# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:18:34 2020
@本代码实现了yolo网络输出的后端处理，能够通过阈值判定和非极大值抑制输出目标检测的方框
本代码中一共执行了三个步骤，首先导入训练好的yolo网络，然后将网络的输出拼接到分解为有含义的张量上，最后对张量进行处理得到所需要的方框
如果要对自己的图片进行标注请遵循如下步骤
修改   image_shape = (1080., 1920.)    中的图片像素分辨率，这个修改关系到输出判定边界的scale，不修改会导致判定盒不匹配
修改   image.save(r"C4_W3_HomeWork_DataSet/out/mytest2.jpg", quality=90)
修改   redict(sess, r"C4_W3_HomeWork_DataSet/images/mytest2.jpg") 路径地址

@author: Netfather
@Last Modified data: 2021年1月19日
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from C4_W3_HomeWork_DataSet.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from C4_W3_HomeWork_DataSet.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


#具体关于yolo的阐述，请查看
#https://www.kesci.com/mw/project/5de0e21aca27f8002c4b29fb/content
#网址中阐述了这个文件所使用yolo以及相关的方法

#这个函数直接作用于yolo的输出结果，在这个结果上使用此函数可以滤除低pc值的方框
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    #box_confidence存储着在这个锚盒中，是否存在预定类别的物体
    #boxes存储着检测到的可能存在物体的边框坐标
    #box_class_probs存储这80个类的分类信息，到底是哪种类别的物体存在于这个区域中
    
    #1.得到每个分类的得分
    box_scores = box_confidence * box_class_probs
    
    #2.根据得到的最大得分，也就是最后80个类别中，得分最高的，我们指定这个网格对应的5个锚盒对应的类别
    box_classes = K.argmax(box_scores,axis = -1) #得到最大值的索引，由于axis=-1，该轴被消灭，返回的索引是一个19*19*5的矩阵
    #上面这个式子我们应当注意到，由于类别也是从0到79，所以在这里索引的值 = 类别的值
    box_class_scores = K.max(box_scores,axis = -1) #得到索引对应的值，由于axis=-1，该轴被消灭，返回的是一个19*19*5的矩阵
    #第一个矩阵中保存着坐标索引信息，第二个矩阵中保存着坐标索引对应的值的信息
    
    #3.根据最大值来和阈值进行比较，低于阈值的我们判定为无效锚盒和无效坐标，这种得到的结果我们需要抛弃
    filtering_mask = box_class_scores >= threshold
    
    #4.将得到的mask应用于上述值
    scores = tf.boolean_mask(box_class_scores, filtering_mask) #低于阈值的值为0
    boxes = tf.boolean_mask(boxes, filtering_mask) #对应位置的不满足坐标为0
    classes = tf.boolean_mask(box_classes, filtering_mask) #对应位置的不满足索引也为0
    #也就是boxes存储这方框的地址，scores和classes前者存储值，后者存储索引，也就是存储着这个到底是哪个类的消息
    return scores, boxes, classes

# with tf.Session() as test_a:
#     box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
#     boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
#     box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
#     scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.shape))
#     print("boxes.shape = " + str(boxes.shape))
#     print("classes.shape = " + str(classes.shape))
#Test OK

#%%完成上述过滤之后，你会删除掉pc值不高，也就是神经网络不自信的方框，但是
#由于没有经过非极大值抑制，此时可能会出现对于不同网格的不同锚盒，都标定了
#同一个物体，此时需要通过非极大抑制来找到和这个物体cp值最高的方框

#1.为了抑制非极大值，我们需要一个指标来指示这个方框和pc值最高的方框到底有多相似
#使用指标iOU 交并比来进行衡量

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """
    #这里输入的坐标值和yolo输出的坐标值不一样
    #yolo输出的坐标值给定的是 中心点x,y 以及方框的宽和高
    #这里给的坐标是方框的左上角和右下角
    #因此我们只需要找到左上角中较大的哪个 和右下角中较小的那个就可以计算
    #注意这里通过镜像 你会发现不管处于什么位置的结构 都是可以通过如此计算面积
    
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])  #左上角较大值
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])  #右下角较小值
    inter_area = (yi2-yi1) * (xi2-xi1)
    #注意这里的面积只是个相对概念，如果没有交际，这里的面积会变成负数
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###
    return iou


# box1 = (2, 1, 4, 3)
# box2 = (3, 0, 5, 2) 
# print("iou = " + str(iou(box1, box2)))
#Test OK!
    
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """ 
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    #max——box参数表示删除掉和pc最大值对应方框 计算iou的前十个值的方框
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    ### END CODE HERE ###

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    #gather命令是根据返回的index来从原来的矩阵中抽取符合条件的
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    ### END CODE HERE ###
    
    return scores, boxes, classes
    
# with tf.Session() as test_b:
#     scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
#     boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
#     classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
#     scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))
#Test OK!


#%%下面开始搭建模型

#如下函数能够将yolo的输出依次通过上述两个过滤器来得到我们关心的方框标签
# GRADED FUNCTION: yolo_eval

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
 ### START CODE HERE ### 

    # Retrieve outputs of the YOLO model (≈1 line)
    #从yolo输出中得到 pc，中心坐标，宽高，以及类别概率
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[:]

    # Convert boxes to be ready for filtering functions 
    #将图像的中心坐标和宽高转换为对角坐标
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    #使用第一个函数获得我们关心的pc值达标的类别的分数，对应坐标，以及对应的类
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    #此步骤调整比例以使输出适应原始的分辨率
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    #进行非极大抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    ### END CODE HERE ###
    
    return scores, boxes, classes

# with tf.Session() as test_b:
#     yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
#     scores, boxes, classes = yolo_eval(yolo_outputs)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))
#Test OK

#%%
#至此已经完成了yolo网络的输出处理，总结如下
# 输入图像维度为（608、608、3）
# 输入图像通过CNN，输出维度为（19,19,5,85）。
# 将最后两个维度展平后，输出为一定体积的形状（19、19、425）：
#      - 输入图像上方19x19网格中的每个单元格给出425个数字。
#      - 425 = 5 x 85，因为每个单元格都包含5个预测框，对应于5个锚定框，如讲座中所示。
#      - 85 = 5 + 80，其中5是因为(pc,bx,by,bh,bw)具有5个数，而80是我们要识别的类别数量
# 然后，你仅根据以下几点选择框：
#      - 分数阈值：丢弃检测到分数小于阈值的类别框
#      - 非极大值抑制：计算并集上的交点，并避免选择重叠的框
# 为你提供YOLO的最终输出。

#%%下面构建一个新的计算图 连接着yolo网络的outputs
sess = K.get_session()

class_names = read_classes(r"C4_W3_HomeWork_DataSet/model_data/coco_classes.txt")
anchors = read_anchors(r"C4_W3_HomeWork_DataSet/model_data/yolo_anchors.txt")
image_shape = (1080., 1920.)  



yolo_model = load_model(r"C4_W3_HomeWork_DataSet/model_data/yolo.h5") 

yolo_model.trainable = False # 将导入的模型冻结

yolo_model.summary()


yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))  
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

#%%
def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image( image_file, model_image_size = (608, 608))
    print(image_data.shape)
    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #注意draw_boxes函数中制定了font路径 一定要指定否则  输出的图片文字太小看不清
    # Save the predicted bounding box on the image
    image.save(r"C4_W3_HomeWork_DataSet/out/mytest2.jpg", quality=90)
    # Display the results in the notebook
    #output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(image)
    
    return out_scores, out_boxes, out_classes
#%%


out_scores, out_boxes, out_classes = predict(sess, r"C4_W3_HomeWork_DataSet/images/mytest2.jpg")

























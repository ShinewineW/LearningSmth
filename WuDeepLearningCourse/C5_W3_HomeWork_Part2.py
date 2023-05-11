# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W3_HomeWork_Part2
# Description:  本作业完成了语音关键字触发检测系统的构建
#               1. 前一大部分都是关于输入数据的处理
#                   一般处理声音数据，都是固定采样序列，将例如10s的输入分割为多少份，然后每一份的44100频率表示了声音的特征
#                   然后根据这个特征进行滑动窗口转换来计算频谱图，因此着这份作业中，输入的音频数据
#                   在时间长度上 从10s 采样为Tx = 5511
#                   声音的特征 从44100 通过傅里叶频谱变换 转换为 101个频率的声音上来
#                   因此输入的X = (batchsize ， 5511， 101)
#                   然后经过一系列的随机插入，overlap检测，标签更新，我们能够人工合成满足题意的数据集
#               2.  在完成数据集处理之后，我们通过构建模型，模型结构参看Model.png，然后导入已经训练好的h5参数，
#                   最终完成了关键字检测
# Author:       Administrator
# Date:         2021/1/15
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------
# 导入必要环境
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import random
import sys
import io
import os
import glob
import IPython
from C5_W3_HomeWork_Part2_DataSet.td_utils import *

# 试听一些数据集音频
# "activate"目录包含人们说"activate"一词的正面示例。
# "negatives"目录包含人们说"activate"以外的随机单词的否定示例。每个音频记录只有一个字。
# "backgrounds"目录包含10秒的不同环境下的背景噪音片段。
# 由于pycharm相关的支持工作并不完善，在pycharm中无法听取，但是在Jupyterbook中是可以的

IPython.display.Audio(r"C5_W3_HomeWork_Part2_DataSet/raw_data/activates/1.wav")
IPython.display.Audio(r"C5_W3_HomeWork_Part2_DataSet/raw_data/activates/1.wav")
IPython.display.Audio(r"C5_W3_HomeWork_Part2_DataSet/raw_data/backgrounds/1.wav")

# 从录音到频谱图
# 从音频的这种“原始”表示中很难弄清是否说了"activate"这个词。
# 为了帮助你的序列模型更轻松地学习检测触发词，我们将计算音频的spectrogram。
# 频谱图告诉我们音频片段在某个时刻存在多少不同的频率。

# 将音频文件切换为数据
x = graph_spectrogram(r"C5_W3_HomeWork_Part2_DataSet/audio_examples/example_train.wav")
_, data = wavfile.read(r"C5_W3_HomeWork_Part2_DataSet/audio_examples/example_train.wav")
# print("Time steps in audio recording before spectrogram", data[:,0].shape)
# print("Time steps in input after spectrogram", x.shape)
# Time steps in audio recording before spectrogram (441000,)
# Time steps in input after spectrogram (101, 5511)
# 通过以上输出，我们可以得出，整个转换后频谱图为5511个时间单元，每个时间单元一共有101中可能输出
# 对应之前的单词模型，Tx = 5511， MAX_SENTENCE_LENGTH = 101

# 定义超参数 为全局变量
Tx = 5511
n_freq = 101
Ty = 1375

# 合成单个训练示例
# 由于语音数据很难获取和标记，因此你将使用激活，否定和背景的音频片段来合成训练数据。
# 录制很多带有随机"activates"内容的10秒音频剪辑非常慢。取而代之的是，
# 录制许多肯定词和否定词以及分别记录背景噪音（或从免费的在线资源下载背景噪音）会变得更加容易。

activates, negatives, backgrounds = load_raw_audio()
# print("background len: " + str(len(backgrounds[0])))
# # Should be 10,000, since it is a 10 sec clip
# print("activate[0] len: " + str(len(activates[0])))
# # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
# print("activate[1] len: " + str(len(activates[1])))
# # Different "activate" clips can have different lengths

# 完成上一步之后，你就获得了所有在原始数据中存储的active，positive和背景数据
# 下面如何进行覆盖，也就是在背景中的某一段，覆盖为active，positive相关的音频，但是依然要保证最终的输出时间为10s
# 当插入或覆盖"activate"剪辑时，还将更新yt的标签，以便输出的该时刻之后的50个步骤标记为具有
# 目标标签1。你将训练GRU来检测何时某人完成说"activate"。例如，假设合成的"activate"剪辑在
# 10秒音频中的5秒标记处结束(恰好在剪辑的一半处)。回想一下 ，由于设定超参数Ty = 1375，
# 因此时间步长 687 = 1375 * 0.5 对应音频的5秒时刻，此时你将设置 Y688 = 1，实际上考虑
# 到实际中的应用，10s的1375份中的一份标记为1实在是过于短暂，因此实际操作中
# 我们将标签之后的50个连续值设置为1。我们有。y688 = y689 = y690 = …… = y737 = 1

# 要实现合成训练集过程，你将使用以下帮助函数。所有这些函数将使用1ms的离散时间间隔，
# 因此将10秒的音频离散化为10,000步。

# 函数 get_random_time_segment(segment_ms)返回一个随机的时间段，
# 我们可以在其中插入持续时间为segment_ms的音频片段。 通读代码以确保你了解它在做什么。
def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

# 接下来，假设你在（1000,1800）和（3400,4500）段插入了音频剪辑。
# 即第一个片段开始于1000步，结束于1800步。
# 现在，如果我们考虑在（3000,3600）插入新的音频剪辑，这是否与先前插入的片段之一重叠？
# 在这种情况下，（3000,3600）和（3400,4500）重叠，因此我们应该决定不要在此处插入片段。

# 出于此函数的目的，将（100,200）和（200,250）定义为重叠，因为它们在时间步200处重叠。
# 但是，（100,199）和（200,250）是不重叠的。

# 实现is_overlapping（segment_time，existing_segments）
# 来检查新的时间段是否与之前的任何时间段重叠。你需要执行2个步骤：
# 1. 创建一个“False”标志，如果发现有重叠，以后将其设置为“True”。
# 2. 循环遍历previous_segments的开始和结束时间。
# 将这些时间与细分的开始时间和结束时间进行比较。4
# 如果存在重叠，请将（1）中定义的标志设置为True。

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    for elements in previous_segments:
        if(elements[1] >= segment_time[0] and elements[0] <= segment_time[0]) or (elements[1] >= segment_time[1] and elements[0] <= segment_time[1]):
            break
    else:
        return False
    return True

# Test OK!
# overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
# overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
# print("Overlap 1 = ", overlap1)
# print("Overlap 2 = ", overlap2)

# 现在，让我们使用以前的辅助函数在10秒钟的随机时间将新的音频片段插入到背景中，
# 但是要确保任何新插入的片段都不会与之前的片段重叠。

# 练习：实现insert_audio_clip()以将音频片段叠加到背景10秒片段上。你将需要执行4个步骤：
#
# 1. 以ms为单位获取正确持续时间的随机时间段。
# 2. 确保该时间段与之前的任何时间段均不重叠。如果重叠，则返回步骤1并选择一个新的时间段。
# 3. 将新时间段添加到现有时间段列表中，以便跟踪你插入的所有时间段。
# 4. 使用pydub在背景上覆盖音频片段。我们已经为你实现了这一点。


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """
    segment_ms = len(audio_clip)
    # 获取要插入的audio_clip长度

    segment_time = get_random_time_segment(segment_ms)
    # 随机得到一个能够插入上述segment_ms的元组 分别代表插入的起始位置

    # 不断随机，知道找到能够插入其中，而且不会覆盖的区间
    while(is_overlapping(segment_time,previous_segments)):
        segment_time = get_random_time_segment(segment_ms)

    # 将新插入的区间更新进入previous_segments 方便后续调用
    previous_segments.append(segment_time)

    # 将audio_clip插入到指定开始位置
    new_background = background.overlay(audio_clip,position = segment_time[0])

    return new_background,segment_time

# Test OK！
# np.random.seed(5)
# audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
# print("Segment Time: ", segment_time)

# 最后，假设你刚刚插入了"activate." ，则执行代码以更新标签yt。
# 在下面的代码中，由于Ty = 1375，所以y是一个 (1,1375)维向量。

# 但是注意 如果"activate"在时间步骤结束，则设置以及最多49个其他连续值。但是，请确保你没有用完数组的末尾并尝试更新 y[0][1375]，由于，所以有效索引是 y[0][0] 至y[0][1374]。
# 因此，如果"activate" 在1370步结束，则只会得到y[0][1371] = y[0][1372] = y[0][1373] = y[0][1374] = 1

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """
    # 由于segment_end_ms是以10000步为10s 计算的结尾，因此
    # 1. 将segment_end_ms转换为对应Ty = 1375的开始位置
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # 2. 将从segment_end_y+1 开始的之后50步骤全部设定为1
    if (Ty - segment_end_y >= 51):
        y[:,segment_end_y+1:segment_end_y+51:] = 1
    else:
        y[:, segment_end_y + 1:Ty:] = 1
    return y



# Test OK!
# plt.clf()
# arr1 = insert_ones(np.zeros((1, Ty)), 9700)
# arr1 = insert_ones(arr1, 4251)
# # print (arr1.max(),arr1.min())
# plt.plot(arr1[0,:])
# plt.show()
#
# # plt.show()
# print("sanity checks:", arr1[0][0],arr1[0][1],arr1[0][1374],arr1[0][1333],arr1[0][1334],arr1[0][1335], arr1[0][634], arr1[0][635])

# 完成上面所有步骤后，我们已经可以从随机的数据中抽取样例，然后和背景10s进行融合，最终得到
# 训练用的Tx片段，和标记着正确输出的Ty片段

# 练习：实现create_training_example()。你需要执行以下步骤：
#
# 1. 将标签向量初始化为维度为的零numpy数组，shape为(1, Ty)
# 2. 将现有段的集合初始化为一个空列表
# 3. 随机选择0到4个"activate"音频剪辑，并将其插入10秒剪辑中。还要在标签向量的正确位置插入标签。
# 4. 随机选择0到2个负音频片段，并将其插入10秒片段中。

def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    # 设定随机数种子
    np.random.seed(18)

    # 直接将背景减少一定数值，用于减轻背景音量
    background = background - 20

    # 1. 初始化Ty输出
    y = np.zeros(shape = [1, Ty])

    #2. 初始化一个包含现有插入序列起始位置的列表
    previous_segments = []

    #3. 随机选择0到4个activate音频，插入10s背景中，并更新y和previous_segments
    number_of_activates = np.random.randint(0, 5) # 随机选取个数
    random_indices = np.random.randint(len(activates), size=number_of_activates) # 在指定范围内，随机选取个数个的数字
    random_activates = [activates[i] for i in random_indices] # 取出对应的activates

    for activate_index in random_activates:
        background,segment_now_index = insert_audio_clip(background,activate_index,previous_segments)
        y = insert_ones(y,segment_now_index[1])

    #4. 同理，随机选择0，2个negtive音频，插入background
    number_of_positives = np.random.randint(0, 3)  # 随机选取个数
    random_indices = np.random.randint(len(negatives), size=number_of_positives)  # 在指定范围内，随机选取个数个的数字
    random_positives = [negatives[i] for i in random_indices]  # 取出对应的activates

    for positive_index in random_positives:
        background,_ = insert_audio_clip(background,positive_index,previous_segments)

    # 标准化合成完成后的语音
    background = match_target_amplitude(background, -20.0)
    # file_handle = background.export("train" + ".wav", format="wav")
    # print("File (train.wav) was saved in your directory.")

    return y

# Test OK!
# y = create_training_example(backgrounds[0], activates, negatives)
# plt.clf()
# plt.plot(y[0])
# plt.show()

#%% 以上就是完整的数据集处理过程，实际上作业中已经安排好了所有的数据集
x_dev = np.load(r"C5_W3_HomeWork_Part2_DataSet/XY_dev/X_dev.npy")
y_dev = np.load(r"C5_W3_HomeWork_Part2_DataSet/XY_dev/Y_dev.npy")

X = np.load(r"C5_W3_HomeWork_Part2_DataSet/XY_train/X.npy")
Y = np.load(r"C5_W3_HomeWork_Part2_DataSet/XY_train/Y.npy")

# print(x_dev.shape)
# print(y_dev.shape)

# 使用keras构建模型
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

# 具体模型参数参看Dataset下的图片Model.png
# (25, 5511, 101)
# (25, 1375, 1)
# 实现model函数

def model():
    inputs = Input(shape= [Tx,n_freq])
    Conv_1D = Conv1D(196,15,strides=4)(inputs)
    # print(Conv_1D.shape)
    Conv_1D = BatchNormalization()(Conv_1D)
    Conv_1D = Activation(activation='relu')(Conv_1D)
    Conv_1D = Dropout(0.8)(Conv_1D)
    # print(Conv_1D.shape)
    # 完成1D卷积之后 shape变更为(batchsize,1375((5511 - kernalsize) / stride + 1),196)
    # 将完成卷积的1维序列通过GRU
    GRU_Senquence = GRU(128,return_sequences= True)(Conv_1D)
    # (batchsize,1375,128)
    GRU_Senquence = Dropout(0.8)(GRU_Senquence)
    GRU_Senquence = BatchNormalization()(GRU_Senquence)

    #再次通过GRU
    GRU_Senquence = GRU(128,return_sequences= True)(GRU_Senquence)
    # (batchsize,1375,128)
    GRU_Senquence = Dropout(0.8)(GRU_Senquence)
    GRU_Senquence = BatchNormalization()(GRU_Senquence)
    GRU_Senquence = Dropout(0.8)(GRU_Senquence)

    #通过密集连接层和sigmoid输出
    Dense_output = Dense(1,activation='sigmoid')(GRU_Senquence)
    # (batchsize,1375,1)


    model = Model(inputs = inputs,outputs = Dense_output)
    return model

model = model()
# x_test = np.ones(shape = [25,Tx,n_freq])
# y_test = model.predict(x_test)
# Test OK!
# model.summary()

# 开始训练
model = load_model(r'C5_W3_HomeWork_Part2_DataSet/models/tr_model.h5')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

#小试2步 检测准确率
model.fit(X, Y, batch_size = 5, epochs=1)

# loss, acc = model.evaluate(x_dev, y_dev)
# print("Dev set accuracy = ", acc)

# 但是由于模型标签严重向0倾斜，因此准确率在这里虽然高于90，但其实是不准确的，这里应该使用例如F1得分或“精确度/召回率”。

# 下面对模型进行预测
def detect_triggerword(filename):
    plt.clf()
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions

# 一旦估计了在每个输出步骤中检测到"activate"一词的可能性，就可以在该可能性高于某个阈值时触发出"chiming（蜂鸣）"声。
# 此外，在说出"activate"之后，对于许多连续值，可能接近1，但我们只希望发出一次提示音。
# 因此，每75个输出步骤最多将插入一次铃声。这将有助于防止我们为"activate"的单个实例插入两个提示音。（
# 该作用类似于计算机视觉中的非极大值抑制）
# 实现chime_on_activate（）。你需要执行以下操作：
#
# 1.遍历每个输出步骤的预测概率
# 2.当预测大于阈值并且经过了连续75个以上的时间步长时，在原始音频剪辑中插入"chime"

chime_file = r"C5_W3_HomeWork_Part2_DataSet/audio_examples/chime.wav"


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0

    audio_clip.export(r"C5_W3_HomeWork_Part2_DataSet/output/chime_output_dev2.wav", format='wav')

# Test OK!
# filename = r"C5_W3_HomeWork_Part2_DataSet/raw_data/dev/1.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.5)

# filename  = r"C5_W3_HomeWork_Part2_DataSet/raw_data/dev/2.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.4)


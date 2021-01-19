# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W3_HomeWork_Part1
# Description:  完成一个注意力模型，来进行文本的翻译
#               使用了两个不同的模型结构
#               2021年1月15日修正 原始结构由于s_prev并没有更新，导致结果完全是错误的
#               更新模型后  0.005 训练50epoch后 可以查看到比较好的结果
# Author:       Administrator
# Date:         2021/1/14
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------
# 你将建立一个神经机器翻译（NMT）模型，以将人类可读的日期（"25th of June, 2009"）
# 转换为机器可读的日期（"2009-06-25"）。
# 你将使用注意力模型来完成此任务，
# 注意力模型是序列模型中最复杂的序列之一。
# 导入必要的软件包
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional,Concatenate,Permute,Dot,Input,LSTM,Multiply
from tensorflow.keras.layers import RepeatVector,Dense,Activation,Lambda,Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model,Model
import tensorflow.keras.backend as K
import numpy as np

from faker import Faker
from tqdm import tqdm_notebook as tqdm
import random
from babel.dates import format_date
from C5_W3_HomeWork_Part1_DataSet.nmt_utils import *
import matplotlib.pyplot as plt

# 导入数据集
m = 10000
dataset, human_vocab,machine_vocab,inv_machine_vocab = load_dataset(m)

# Test
# print(dataset[:10])
# print(human_vocab)
# print(machine_vocab)
# print(inv_machine_vocab)
# 观察上述输出可以发现 dataset是一个列表，存储所有数据
# 在这里 human_voacb对应输入
# machine_vocab对应输出
# 后三个都是字典  其中  machine_vocab和inv_machine_vocab互为逆序

# 预处理数据并将原始数据映射到索引值
# 其中设定 Tx = 30 也就是最大长度为30 超过的截断
# Ty = 10 也就是最大的输出长度为10
Tx = 30
Ty = 10

#这个函数会返回
# X： 按照human_vocab将每一个char映射到 int类型 不认识的用‘ukn’代替，同时固定长度30
# Y： 按照machine_vocab将每个char映射到int类型，不认识的用‘ukn’代替，同时固定长度10
# Xoh： 按照输入大小，分为独热向量
# Yoh： 按照输入大小，转为独热向量
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)


#Test
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
# Output
# X.shape: (10000, 30)
# Y.shape: (10000, 10)
# Xoh.shape: (10000, 30, 37)
# Yoh.shape: (10000, 10, 11)
# 根据shepe 可以得到 batchsize = 10000， MAX_SENTENCE_WORDS = 30, EMBEDDING_DIMS = 37

# 再来观察一些处理完成的向量
# index = 0
# print("Source date:", dataset[index][0])
# print("Target date:", dataset[index][1])
# print()
# print("Source after preprocessing (indices):", X[index])
# print("Target after preprocessing (indices):", Y[index])
# print()
# print("Source after preprocessing (one-hot):", Xoh[index])
# print("Target after preprocessing (one-hot):", Yoh[index])
#

# 数据集处理完成，接下来开始注意力机制的编写
# 具体的过程参看 https://www.kesci.com/mw/project/5de8bb0409f741002cac101f/content

# 这里实现one_step_attention()  这个层将对每一个attention 执行一次

# 定义公用参数的层 这些层只需要定义一次，然后可以call 很多次 以实现共享权重
repeator = RepeatVector(Tx)  #用于复制St-1 Tx次 方便生成softmax
concatenator = Concatenate(axis=-1) #用于合并两个矩阵
densor1 = Dense(10, activation = "tanh") #密集连接用于维度投影
densor2 = Dense(1, activation = "relu")  #最后一层密集连接，用于生成概率
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1) #点乘层，用于将生成的注意力矩阵和原始的激活输出做乘法


# GRADED FUNCTION: one_step_attention

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # 这里a 是一个双向LSTM的输出 这里fetch必要参数
    Batchsize , Tx, Double_hidden_units = a.shape

    # 这里s 是一个从上一个输出层单向LSTM的隐藏激活
    Batchsize , Single_hidden_units = s_prev.shape

    # 1.首先将 s 重复vector次 生成和 a 维度匹配的能够合并的单元
    # 关于  RepeatVector 这个层只是用于重复特征n次 所以完全不需要指定axis
    # Input shape:
    # 2D tensor of shape (num_samples, features).
    #
    # Output shape:
    # 3D tensor of shape (num_samples, n, features).
    s_repeate = repeator(s_prev ) # 此时维度应该为 (batchsize,Tx,single_hidden_units)

    # 2. 将两个维度匹配的张量进行合并，合并维度为最后一维度
    a_s_concat = concatenator([a,s_repeate]) # 此时维度应该为 (batchsize,Tx,single_hidden_units+ Double_hidden_units)

    # 3. 将每一个词位置的 single_hidden_units+ Double_hidden_units 的特征 投影到 10维上来
    # 注意这里的 Dense层 根据官方文档的特征 也是忽视中间，只对最后一个轴的特征进行投影
    Dense_10 = densor1(a_s_concat)
    Dense_1 = densor2(Dense_10) # 此时维度应该为 (batchsize,Tx,1)

    # 4. 投影到1维特征之后 通过softmax 得到对应的注意力矩阵
    alpha_attention = activator(Dense_1)  # 此时维度应该为 (batchsize,Tx,1)

    # 5. 累加求和计算 context
    # 这里的点乘层非常的tricky 翻阅官方文档
    # 我们可以知道 axes 指定的是 我们从哪个轴开始做点成
    # 这里 输入两方分别是 (batchsize,Tx,1) (batchsize,Tx,Double_hidden_units)
    # 通过设定axes为1 我们让点乘发生在 Tx反向 这样结果为 (batchsize,1,Double_hidden_units)
    context = dotor([alpha_attention,a])
    # 输出大小为 (batchsize,1,Double_hidden_units)
    return context


# 构建用于model 的全局变量
n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)


# 下面开始构建model
# 如下model是直接使用lstm返回sequence 也可以训练 并进行预测
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    inputs = Input(shape=(Tx,human_vocab_size))
    LSTM_Input_Bi = Bidirectional(LSTM(n_a,return_sequences= True))(inputs)
    s_prev = Input(shape=(n_s,), name='s0')
    c_prev = Input(shape=(n_s,), name='c0')
    s = s_prev
    c = c_prev
    Context_list = []
    for i in range(Ty):
        # Context_list.append(one_step_attention(LSTM_Input_Bi,s_prev))
        # Fixing bugs！ 这里出现问题，可以注意到这里 s_prev是没有任何变化的，也就意味着每一次预测都是s0在输入
        # 而不是原有网络的 St-1输入！

        #2021年1月15日做出修正 将 LSTM纳入循环中进行遍历
        Context = one_step_attention(LSTM_Input_Bi,s)
        # print(Context.shape)
        # (?, 1, 64)
        # 注意这里是结果 所以只剩下 batchsize, units 中间的1 消失不见了
        s,_,c = post_activation_LSTM_cell(Context,initial_state=[s,c])
        # (?, 64)
        Final_Denseinput = tf.expand_dims(s,axis= 1)
        Context_list.append(Final_Denseinput)

    # print(len(Context_list))
    # print(Context_list[0].shape)
    # (?, 64)
    Attention_layer = Concatenate(axis= 1)(Context_list)
    # print(Attention_layer.shape)
    Dense_Output = Dense(machine_vocab_size)(Attention_layer)
    # print(Dense_Output.shape)
    Output_total = Softmax(axis= -1)(Dense_Output)
    # Output = tf.unstack(Output_total,axis = 1)
    # print(len(Output))
    #
    # for itr in Output:
    #     print(itr.shape)

    Attention_Model = Model(inputs = [inputs,s_prev,c_prev],outputs = Output_total)

    return Attention_Model


# GRADED FUNCTION: model

def modelWu(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []


    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)


    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model.summary()

# X_test = tf.ones(shape= [4,30,37])
# s0 = np.zeros((4, n_s))
# c0 = np.zeros((4, n_s))
# Y_test = model.predict([X_test,s0,c0],steps= 1)
# print(Y_test.shape)

# outputs = list(Yoh.swapaxes(0,1))
# test = np.array(outputs)
# print(test.shape)
# Test OK!

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
# #
# opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#
#
# model.fit([Xoh, s0,c0], Yoh, epochs=50, batch_size=100)

# 在25个epochs后可以达到  62左右的准确率
# 使用 0.005 的学习率 学习25个epoch 出现过拟合
# 以上是模型结构错误的结果！！！   因为错误的没有更新s0导致  重新训练全新结果
# 修改成正确模型后
# 学习率 0.005 epoch 25 之后 效果就很好了
# 学习率 0.005 epoch 50 之后 效果差不多 可以接受
# 完成构建 但是由于课程中 使用的是cell 来循环构建最终输出，因此参数的调用也会完全不一样
# 具体可以查看中的输出
# 格式输出都是正确的  但是远达不到吴恩达所给模型的权重水准

def test_Senquence_Model():
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
                'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
        prediction = model.predict([[source], s0,c0])
        prediction = np.squeeze(prediction)
        prediction = np.argmax(prediction, axis=-1)  # shape (ty,1)
        prediction = np.squeeze(prediction)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output))

def test_Wu_Model():
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
        prediction = model.predict([[source], s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output))

# model1 = modelWu(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model1.summary()
# model2 = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model2.summary()
model.load_weights(r'C5_W3_HomeWork_Part1_DataSet/models/model.h5')
# test_Wu_Model()
test_Senquence_Model()
# 关于可视化注意力值 由于在模型搭建过程中使用了 name = ‘attention_weights’
# attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)













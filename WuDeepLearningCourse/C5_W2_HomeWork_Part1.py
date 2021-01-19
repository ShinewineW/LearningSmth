# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W2_HomeWork_Part1
# Description:  本次作业分成两部分  第一部分 使用glove的词嵌入模型，用最简单的深度神经网络，只是用numpy实现了一个
#                                           网络模型，效果比较糟糕，而且无法联系上下文，对于not happy 和 happy完全无法区分
#                                   第二部分 使用keras模型，使用embedding层和LSTM函数，很快捷的构建了一个LSTM的循环神经网络
#                                   并进行训练，最后结果非常不错
# Author:       Administrator
# Date:         2021/1/11
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------

#%% 导入必要包
import numpy as np
from C5_W2_HomeWork_Part1_DataSet.emo_utils import *
import emoji
import matplotlib.pyplot as plt


## 1.先从构建一个简单的分类器开始
# 假设你有一个很小的数据集(X,Y)
# X: 包含127个句子
# Y: 为一个0到4之间的证书，对应每一个句子的表情符号，也就是有5种表情
# 读入数据
X_train,Y_train = read_csv(r'C5_W2_HomeWork_Part1_DataSet/data/train_emoji.csv')
X_test,Y_test = read_csv(r'C5_W2_HomeWork_Part1_DataSet/data/tesss.csv')

print(X_train.shape,Y_train.shape)
print(X_train[0])
print(Y_train[0])
# 训练集是一个132个batch的数据，X中存储着字符串，Y中存储着对应的整数
#
aa = max(X_train, key=len)
maxLen = len(aa.split())  # 返回一个列表，列表中存储aa的每个单词

print(aa)
print(maxLen)
#
# 将训练和测试集合的Y都转换成独热码 方便softmax使用
Y_oh_train = convert_to_one_hot(Y_train,C = 5)
Y_oh_test = convert_to_one_hot(Y_test,C = 5)

#
# 实现Emojifier-V1
# 将输入句子转换为单词向量表示，然后平均在一起
# 这里使用预训练好的50维GloVe词嵌入
word_to_index,index_to_word,word_to_vec_map = read_glove_vecs(r'C5_W2_HomeWork_Part1_DataSet/data/glove.6B.50d.txt')


#
# 检查glove词嵌入是否正常工作
word_key = 'apple'
index_key = 289846
print('the index of {} is {}'.format(word_key,word_to_index[word_key]))
print('the index {} refers to word {}'.format(index_key,index_to_word[index_key]))
print('the word {} refers to 50dimensional matrix {}'.format(word_key,word_to_vec_map[word_key]))
##
# 实现sentence_to_avg
# 将每个句子转换为小写，然后将句子拆分为单词列表。X.lower()和X.split()可能有用。
# 对于句子中的每个单词，请访问其GloVe表示。然后，将所有这些值取平均值。

# def sentence_to_avg(sentence, word_to_vec_map):
#     """
#     Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
#     and averages its value into a single vector encoding the meaning of the sentence.
#
#     Arguments:
#     sentence -- string, one training example from X
#     word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
#
#     Returns:
#     avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
#     """
#     sentence_word = sentence.split()
#     sentence_lower = [x.lower() for x in sentence_word]
#
#     # 构建输出矩阵
#     avg = np.zeros(shape = 50)
#     word_num = 0
#     for element in sentence_lower:
#         avg += word_to_vec_map[element]
#         word_num+=1
#     avg /= word_num
#     return avg
def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = (sentence.lower()).split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(50)

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)


    return avg


# Test OK!
# avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
# print("avg = ", avg)
##
#构建模型，这是一个很简单的线性模型，深度神经网络
def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    np.random.seed(1)

    m = Y.shape[0]  # 样本数量
    n_y = 5 # 输出的y的类
    n_h = 50 # 隐藏单元数量

    # 初始化必要参数
    W = np.random.randn(n_y,n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    Y_oh = convert_to_one_hot(Y,C = n_y)

    for t in range(num_iterations):
        for i in range(m):
            avg = sentence_to_avg(X[i],word_to_vec_map)
            Y_pred = np.dot(W,avg) + b
            a = softmax(Y_pred)
            cost = -np.sum(Y_oh[i]*np.log(a))

            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1),avg.reshape(1,n_h))
            db = dz

            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch:" + str(t) + "-----cost = " + str(cost))
            pred = predict(X,Y,W,b,word_to_vec_map)

    return pred,W,b

# print(X_train.shape)
# print(Y_train.shape)
# print(np.eye(5)[Y_train.reshape(-1)].shape)
# print(X_train[0])
# print(type(X_train))
# Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
# print(Y.shape)

# X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
#   'Lets go party and drinks','Congrats on the new job','Congratulations',
#   'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
#   'You totally deserve this prize', 'Let us go play football',
#   'Are you down for football this afternoon', 'Work hard play harder',
#   'It is suprising how people can be dumb sometimes',
#   'I am very disappointed','It is the best day in my life',
#   'I think I will end up alone','My life is so boring','Good job',
#   'Great so awesome'])

# print(X.shape)
# print(np.eye(5)[Y_train.reshape(-1)].shape)
# print(type(X_train))
# pred, W, b = model(X_train, Y_train, word_to_vec_map)
# print(pred.T)

# ##
# print("Training set:")
# pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
# print('Test set:')
# pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

# ##
# X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
# Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

# pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
# print_predictions(X_my_sentences, pred)

#%%
# 下面使用keras中LSTM来进行一个更加强大的神经网络的构建
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

# 注意事项，我们的网络结构是在一开始就设定好长度的，这就意味着对于长短不一
# 的句子，我们很难在一开始就设定长度，此时常用方法就是直接选择数组中单词
# 数最长的句子，然后将所有其他句子全部填充为这个长度

# 实现以下函数，将X（字符串形式的句子数组）转换为与句子中单词相对应的
# 索引数组。输出维度应使其可以赋予Embedding()（如图4所示）。
def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    # X为直接从数据集中拿过来的句子
    # word_to_index 为从embedding中拿到的数据
    # Fetch 必要数据
    m = X.shape[0]

    #构建输出数组
    X_indices = np.zeros(shape= (m,max_len))

    for i in range(m):
        Word_Len_Per_Sentence = len(X[i].split())
        X_indices[i,:Word_Len_Per_Sentence:] = [word_to_index[x] for x in X[i].lower().split()]

    return X_indices

X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)

#%%
# 实现pretrained_embedding_layer()。你将需要执行以下步骤：

# 将嵌入矩阵初始化为具有正确维度的零的numpy数组。
# 使用从word_to_vec_map中提取的所有词嵌入来填充嵌入矩阵。
# 定义Keras嵌入层。 使用Embedding()。确保在调用 Embedding()时设置trainable = False来使该层不可训练。
#     如果要设置trainable = True，那么它将允许优化算法修改单词嵌入的值。
# 将嵌入权重设置为等于嵌入矩阵
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    vocab_len = len(word_to_index) + 1  #获得词典最大存储的数量
    emb_dim = word_to_vec_map["cucumber"].shape[0]   #获得嵌入矩阵的维度
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    for word,index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]  #将字典写入emb_matrix矩阵中用于初始化
        
    embedding_layer = Embedding(vocab_len,emb_dim,trainable = False)
    
    embedding_layer.build((None,))
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer 


# Test OK
# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

#%%
# 使用keras中自带api实现一个标准的LSTM两层模块，其中使用dropout层进行连接
def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    sentece_indices = Input(shape = input_shape,dtype = 'int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    embeddings = embedding_layer(sentece_indices)
    
    X = LSTM(128, return_sequences = True)(embeddings)
    
    X = Dropout(0.5)(X)
    
    X = LSTM(128,return_sequences = False)(X)
    
    X = Dropout(0.5)(X)
    
    X = Dense(5)(X)
    
    X = Activation('softmax')(X)
    
    model = Model(inputs = sentece_indices,outputs = X)
    
    return model

model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()

#%%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

#%%

x_test = np.array(['she got me a nice present'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))










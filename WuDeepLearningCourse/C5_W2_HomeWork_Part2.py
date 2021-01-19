# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         C5_W2_HomeWork_Part2
# Description:  本作业详细展示了关于词向量的一些基本操作，
#               1. 使用余弦相似度来评估两个词在嵌入矩阵空间中的相似关系
#               2. 使用查询操作来找到两对单词的相似度，
#               3. 使用正交化操作来进行某个单词在性别方面的偏见
#               4. 使用等于化操作来进行某对单词在歧视轴上的对称
#                   3 4两点具体参看吴恩达的课程笔记。
# Author:       Administrator
# Date:         2021/1/12
# Last Modified data: 2021年1月19日
# -------------------------------------------------------------------------------
##
# 加载所需的包
import numpy as np
import sys
from C5_W2_HomeWork_Part2_DataSet.w2v_utils import *


##
# 读入glove的50维词嵌入矩阵
words,word_to_vec_map = read_glove_vecs(r'C5_W2_HomeWork_Part1_DataSet/data/glove.6B.50d.txt')
print(type(word_to_vec_map))
# words: 词汇表中的单词集合
# word_to_vec_map: 将单词映射到50维度的特征向量中


##
# 1. 比较余弦相似度
# 一个训练晚上的词嵌入矩阵，应该在同义词上存在语义相似，在反义词上存在语义相对
# 首先实现一个函数，函数接受两个维度相等的向量，然后返回这两个向量的余弦相似度
def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    U_Normal2 =  np.linalg.norm(u)
    V_Normal2 =  np.linalg.norm(v)
    res = np.dot(u,v) / (U_Normal2 * V_Normal2)
    return res

# Test OK!
# father = word_to_vec_map["father"]
# mother = word_to_vec_map["mother"]
# ball = word_to_vec_map["ball"]
# crocodile = word_to_vec_map["crocodile"]
# france = word_to_vec_map["france"]
# italy = word_to_vec_map["italy"]
# paris = word_to_vec_map["paris"]
# rome = word_to_vec_map["rome"]
#
# print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
# print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
# print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

##
# 2.单词类比任务
# 在类比任务中，我们完成句子"a is to b as c is to __"。
# 一个例子是'man is to woman as king is to queen'。
# 详细地说，我们试图找到一个单词d，以使关联的单词向量通过
# 以下方式相关：eb - ea ≈ ed - ec 。我们将使用余弦相似性
# 来衡量彼此之间的相似性。
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    # Cos_Sim_a_b = cosine_similarity(word_to_vec_map[word_a],word_to_vec_map[word_b])
    best_word = ''
    max_dis = -100
    for key in word_to_vec_map:
        if key == word_c or key == word_a or key == word_b:
            continue
        temp_dis = cosine_similarity(word_to_vec_map[word_c]-word_to_vec_map[key],word_to_vec_map[word_a]-word_to_vec_map[word_b])
        if  temp_dis > max_dis:
            max_dis =  temp_dis
            best_word = key
    return best_word

# Test OK!
# triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
# for triad in triads_to_try:
#     print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))

##
# 关于词嵌入还有一个非常关键的点，在于消除词嵌入矩阵中的偏见，
# 为了避免性别歧视，种族歧视，肤色歧视。
# 首先观察一下对于这个词嵌入矩阵，内部对于性别的定义是如何的

gap_gender = word_to_vec_map['woman'] - word_to_vec_map['man']
print(gap_gender)

# 然后我们随机取一些值，考察这些单词和这个性别定义向量的相似度
print ('List of names and their similarities with constructed vector:')

# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], gap_gender))

# 如你所见，女性名字与我们构造的向量gender的余弦相似度为正，
# 而男性名字与余弦的相似度为负。这并不令人惊讶，结果似乎可以接受。

print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], gap_gender))

# 如果我们尝试明明是中立的词汇，但是我们可以发现，电脑computer接近男性，文学家literature接近女性

##
# 消除非性别特定此的偏见
# 注意，诸如 "actor"/"actress" 或者
# "grandmother"/"grandfather"之类的词对应保持性别特定，
# 而诸如"receptionist" 或者"technology"之类的其他词语
# 应被中和，即与性别无关。消除偏见时，你将不得不区别对待这
# 两种类型的单词。
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    # 目标其实很简单，就是消除 word向量在gender 反向的投影，使得他们二者的正交和为0
    # 1. 得到word的50维度空间特征
    e = word_to_vec_map[word]

    # 2.计算二者的投影向量
    e_biascomponent = (np.dot(e, g) / np.square(np.linalg.norm(g))) * g

    # 3.将原向量 - 投影向量 就可以得到正交轴上的全新向量
    e_debiased = e - e_biascomponent
    return e_debiased


# Test OK
# e = "receptionist"
# print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], gap_gender))
#
# e_debiased = neutralize("receptionist", gap_gender, word_to_vec_map)
# print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, gap_gender))

##
# 均衡背后的关键思想是确保一对特定单词与49维 等距。均衡步骤还确保了两个均衡步骤现在与或与
# 任何其他已中和的作品之间的距离相同。图片中展示了均衡的工作方式：
def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2) / 2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = np.dot(mu, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
    e_w1B = np.dot(e_w1, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth * mu_orth))) * (e_w1B - mu_B) / np.linalg.norm(
        e_w1 - mu_orth - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth * mu_orth))) * (e_w2B - mu_B) / np.linalg.norm(
        e_w2 - mu_orth - mu_B)

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2

print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], gap_gender))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], gap_gender))
print()
e1, e2 = equalize(("man", "woman"), gap_gender, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, gap_gender))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, gap_gender))
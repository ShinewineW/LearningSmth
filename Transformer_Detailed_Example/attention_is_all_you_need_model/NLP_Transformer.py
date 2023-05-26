# -*- coding: utf-8 -*-
#
#@File:  NLP_Transformer.py
#  NLP Transformer
#@Time:  Created by Jiazhe Wang on 2023-05-15 17:30:56.
#@Author:  Copyright 2023 Jiazhe Wang. All rights reserved.
#
#@Email  wangjiazhe@toki.waseda.jp
#
#@Desc: 
# 此文件会给出  NLP 中 Transformer 每一层张量大小变化的样例。

#  https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
# 这里是一份 attention is all you need 的源代码  我们忽略训练的部分
# 直接观察transformer的具体结构

# 最核心的部分 在 /storage/shinewine/LearningSmth/Transformer_Detailed_Example/attention_is_all_you_need_model/attention-is-all-you-need-pytorch/transformer/SubLayers.py 
# 文件中

# https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer

# 我们将文件中对应的 主要结果收录到这个文件中
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


DEBUG_MODE = True
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch():
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    input_batch = torch.LongTensor(input_batch)
    output_batch = torch.LongTensor(output_batch)
    target_batch = torch.LongTensor(target_batch)
    if (DEBUG_MODE):
        print("#"*10 , "输出经过batch组成数据集之后" , "#" * 10)
        print("enc_inputs的样子:{}".format(input_batch))
        print("enc_inputs的尺寸:{}".format(input_batch.size()))
        print("dec_inputs的样子:{}".format(output_batch))
        print("dec_inputs的尺寸:{}".format(output_batch.size()))
        print("target_batch的样子:{}".format(target_batch))
        print("target_batch的尺寸:{}".format(target_batch.size())) # 注意到尺寸为 Batch * 句子长度
    return input_batch, output_batch, target_batch

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)  # 这里就是直接按照 位置编码公式写的东西
        # 和论文中的公式完全一致   pos 表示当前这个词在句子中的哪个位置    hid_idx 表示遍历所有的 隐藏维度 都计算一下
        # 所以这个公式本质上就是  对当前词在句子中的位置  我们计算一下 这个位置对所有维度512维度 每个维度 都有哪些影响
        # 所以 这个 pos 的输出尺寸就是    num_sequence x  num_dimension of Embedding
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  # 偶数去 sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 # 奇数取 cos
    output = torch.FloatTensor(sinusoid_table)
    if (DEBUG_MODE):
        print("#"*10 , "PE中PE查找表的计算" , "#" * 10)
        print("PE查找表的尺寸:{}".format(output.size()))
    return output

def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.  mask为true的地方 将注意力归零
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # 这里 分开 Q K V 输入的原因在于 在decoder中  K V 输入是 encoder提供的 而 Q 是正常的输入
        # 在 encoder中 这三者完全相等

        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        # 经过线性层 将 原本的  Batch * Len * EmbedingDim 变成对应的三个矩阵  Batch * len * (head * Dim_k)
        # 然后经过简单的 reshape 将 head 转移到第二个维度上来 即 Batch * head * len * Dim_k
        # 然后做注意力  输出的结果不变  Batch * head * len * Dim_k 
        # 然后再 还原会 Batch * len * head* Dim_k
        # 然后再用一个线性层 还原回原本的 维度 实现一次多头注意力  且尺寸对比输入尺寸 没有任何变化
        # 输出结果中的  句子长度 是由 Q矩阵决定 这也是为什么 Q一定要由 Decoder提供的原因
        # K V 决定了输出句子的 嵌入维度。
        
        # attn_mask = [batch_size x len_sequence x len_sequence]  将原本的 尺度在num_head维度上重复相应次数  
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask) # Step1  decoder输入自己对自己做 多注意力
        # 输出尺寸为  Batch * len_target_sequence * Dim_Ebbeding
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask) # Step2 这里就需要编码器和解码器互相做多头注意力
        # 输出尺寸为  Batch * len_target_sequence_of_decoder * Dim_Ebbeding_encdoer
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        # print(self.src_emb(enc_inputs).size())
        # print(self.pos_emb(torch.LongTensor([[1,2,3,4,0]])).size())
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]])) # 这里的  pos_emb直接写死了
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # 这里的作用是将 句子结束标识符 设置为true 在做自注意力的时候执行 负无穷 不将句子结束符考虑在注意力中
        if (DEBUG_MODE):
            print("#"*10 , "经过PE和 生成mask" , "#" * 10)
            print("经过词嵌入和PE的尺寸{}".format(enc_outputs.size()))
            print("经过input自己生成的mask样子\n{}".format(enc_self_attn_mask))
            print("经过input自己生成的mask尺寸{}".format(enc_self_attn_mask.size()))
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len] # enc_outputs :

        if (DEBUG_MODE):
            print(dec_inputs)
            print(dec_inputs.size()) # torch.Size([1, 5])
            print(enc_inputs)
            print(enc_inputs.size()) # torch.Size([1, 5])
            print(enc_outputs)
            print(enc_outputs.size())  # torch.Size([1, 5, 512])
        dec_outputs = self.tgt_emb(dec_inputs)
        if (DEBUG_MODE):
            print(dec_outputs.size())  # 经过  emb之后 
        

        dec_outputs = dec_outputs + self.pos_emb(torch.LongTensor([[5,1,2,3,4]])) # 这个地方写死了 decoder 的输入
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # print(dec_self_attn_mask)  # 就是一个很经典的对角矩阵  第一行表示 第一个词放入 后面的所有词都要置空
        # 第二行表示 第二个词放入 以此类推
        # tensor([[[False,  True,  True,  True,  True],
        #  [False, False,  True,  True,  True],
        #  [False, False, False,  True,  True],
        #  [False, False, False, False,  True],
        #  [False, False, False, False, False]]])  //
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # 将 dec 和 enc 中的 Pading词屏蔽
        # print(dec_enc_attn_mask) # 输出的结果中 依然是将Pading 标识符屏蔽

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, 5):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

def showgraph(attn, name):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.savefig(os.path.join( os.getcwd(), name))
    plt.show()

if __name__ == '__main__':

    # make sure that the cwd() is the location of the python script (so that every path makes sense)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']   # 三个句子分别代表 原始训练输入， decoder输入  和 最终目标输出
    # 这里的 P 并不是 终止字符  而是为了 方便多Batch的情况 用 P 来表示 Pad 填充的数字
    # Transformer Parameters
    # Padding Should be Zero index
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    # 如果我们想把 目标长度缩短 这里的目标词库中 就应该不包含 "a"
    # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'beer': 3, 'S': 4, 'E': 5}

    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # length of source
    tgt_len = 5 # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    enc_inputs, dec_inputs, target_batch = make_batch()

    for epoch in range(1):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs, greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns, "1.png")

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns,"2.png")

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns,"3.png")

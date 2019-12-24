---
layout:     post
title:      Transformer
subtitle:   学习、理解、应用
date:       2019-10-25
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 前言
看过很多相关Transformer的文章了，知识点零零碎碎，面试的东西只能回答一部分，实验只调试了别人的代码，诸多原理都还没深刻理解。今天就好好整理一下资料，调试相关代码，以便以后复习。

## 正文
谈到Transformer这个特征抽取器就需要与CNN和RNN这两个抽取器作一个对比分析：<br>
CNN像滑动窗口移动卷积核抽取特征，经过池化选择特征，局部抽取能力很强，抽取依赖关系能力较弱。<br>
LSTM建模输入序列的依赖关系，保存和更新记忆单元，过长的句子性能会下降。<br>
Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。性能提升的关键是将任意两个单词的距离是1，解决NLP中棘手的长期依赖问题。


## 回顾序列编码问题：
RNN:        y<sub>t</sub> = f (y<sub>t−1</sub>, x<sub>t</sub>)

CNN:        y<sub>t</sub> = f (x<sub>t−1</sub>, x<sub>t</sub>, x<sub>t+1</sub>)

Attention:  y<sub>t</sub> = f (x<sub>t</sub>, A, B)            

其中A,B是另外一个序列（矩阵）。如果都取A=B=X，那么就称为Self Attention，它的意思是直接将xt与原来的每个词进行比较，最后算出yt！



## transformer结构

![transformer_block](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191118transformer_block.jpg)




## self-attention

self-Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：
- 第一个过程是根据Query和Key计算权重系数
- 第二个过程根据权重系数对Value进行加权求和

而第一个过程又可以细分为两个阶段：第一个阶段根据Query和Key计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如图10展示的三个阶段。

![](https://pic2.zhimg.com/80/v2-07c4c02a9bdecb23d9664992f142eaa5_hd.jpg)

更详细的：

**The first step** in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

![](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

What are the “query”, “key”, and “value” vectors?

They’re abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

**The second step** in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

![](https://jalammar.github.io/images/t/transformer_self_attention_score.png)


**The third and forth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.
![](https://jalammar.github.io/images/t/self-attention_softmax.png)

This softmax score determines how much how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.



**The fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

**The sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

![](https://jalammar.github.io/images/t/self-attention-output.png)

**Finally**, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.
![](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)




## Multi-head attention
The Beast With Many Heads
The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. This improves the performance of the attention layer in two ways:

It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, we would want to know which word “it” refers to.

它扩展了模型关注不同位置的能力。

It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

这个是Google提出的新概念，是Attention机制的完善。不过从形式上看，它其实就再简单不过了，就是把Q,K,V通过参数矩阵映射一下，然后再做Attention，把这个过程重复做h次，结果拼接起来就行了，可谓“大道至简”了。具体来说

head<sub>i</sub>=Attention(QQ<sub>i</sub><sup>W</sup>,KK<sub>i</sub><sup>W</sup>,VV<sub>i</sub><sup>W</sup>)

W<sub>i</sub><sup>Q</sup>∈R<sup>k×^k</sup>,  K<sub>i</sub><sup>Q</sup>∈R<sup>k×^k</sup>, V<sub>i</sub><sup>Q</sup>∈R<sup>v×^v</sup>

MultiHead(Q,K,V)=Concat(head<sub>i</sub>,...,head<sub>h</sub>)

最后得到一个n×(hd~v)的序列。所谓“多头”（Multi-Head），就是只多做几次同样的事情（参数不共享），然后把结果拼接。
![](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

Attention层的好处是能够一步到位捕捉到全局的联系，因为它直接把序列两两比较（代价是计算量变为O(n2)，当然由于是纯矩阵运算，这个计算量相当也不是很严重）；相比之下，RNN需要一步步递推才能捕捉到，而CNN则需要通过层叠来扩大感受野，这是Attention层的明显优势。

## 参考
[目前主流的attention方法都有哪些？--张俊林](https://www.zhihu.com/question/68482809/answer/264632289)

[苏剑林. (2018, Jan 06). 《《Attention is All You Need》浅读(简介+代码)](https://kexue.fm/archives/4765)

[illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

[huggingface.co/transformers](https://huggingface.co/transformers/)








## 核心代码

```python
def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
##为什么 lookuptable取切片[1:,:] lookup_table是怎样排序的呢？
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5) 

    return outputs

```

## 常见问题

请介绍一下自注意力机制？

为什么需要多头注意力？

FF Network在这里起什么作用？

残差连接原理？

Encoder-Decoder分别做了什么？

请描述一个使用transformer的整体流程？

## 整体框架

深入transformer机器翻译项目的流程框架：

1. download data
2. prepro.py 预处理
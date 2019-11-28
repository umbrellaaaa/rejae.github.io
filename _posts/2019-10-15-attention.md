---
layout:     post
title:      attention 
subtitle:   summary
date:       2019-10-15
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---

## 前言
attention机制在目前学术界可谓是随处可见，但是attention又分了很多个种类，不同的场景应用不同的attention，下面就对个人所看到的比较好的文章进行重组:

[深度学习中的注意力机制(2017版)])https://blog.csdn.net/malefactor/article/details/78767781?source=post_page-----d332e85e9aad----------------------)

Attention机制的本质思想

如果把Attention机制从上文讲述例子中的Encoder-Decoder框架中剥离，并进一步做抽象，可以更容易看懂Attention机制的本质思想。
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191128attentionzjl.jpg)

我们可以这样来看待Attention机制（参考图9）：将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191128attentionzjl2.jpg)

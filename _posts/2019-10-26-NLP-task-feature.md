---
layout:     post
title:      NLP任务的特点及其类型
subtitle:   战场侦察
date:       2019-10-26
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 前言


![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191026transformer_example1.jpg)

NLP任务的特点和图像有极大的不同，上图展示了一个例子，NLP的输入往往是一句话或者一篇文章，所以它有几个特点：
- 首先，输入是个一维线性序列，这个好理解；
- 其次，输入是不定长的，有的长有的短，而这点其实对于模型处理起来也会增加一些小麻烦；
- 再次，单词或者子句的相对位置关系很重要，两个单词位置互换可能导致完全不同的意思。如果你听到我对你说：“你欠我那一千万不用还了”和“我欠你那一千万不用还了”，你听到后分别是什么心情？两者区别了解一下；
- 另外，句子中的长距离特征对于理解语义也非常关键，例子参考上图标红的单词，特征抽取器能否具备长距离特征捕获能力这一点对于解决NLP任务来说也是很关键的。上面这几个特点请记清，一个特征抽取器是否适配问题领域的特点，有时候决定了它的成败，而很多模型改进的方向，其实就是改造得使得它更匹配领域问题的特性。这也是为何我在介绍RNN、CNN、Transformer等特征抽取器之前，先说明这些内容的原因。 NLP是个很宽泛的领域，包含了几十个子领域，理论上只要跟语言处理相关，都可以纳入这个范围。但是如果我们对大量NLP任务进行抽象的话，会发现绝大多数NLP任务可以归结为几大类任务。两个看似差异很大的任务，在解决任务的模型角度，可能完全是一样的。





## 参考

[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
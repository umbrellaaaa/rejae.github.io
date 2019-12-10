---
layout:     post
title:      Glove & FastText
subtitle:   
date:       2019-11-30
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---
<p id = "build"></p>
---

## Glove介绍
要讲GloVe模型的思想方法，我们先介绍两个其他方法：

一个是基于奇异值分解（SVD）的LSA算法，该方法对term-document矩阵（矩阵的每个元素为tf-idf）进行奇异值分解，从而得到term的向量表示和document的向量表示。此处使用的tf-idf主要还是term的全局统计特征。

另一个方法是word2vec算法，该算法可以分为skip-gram 和 continuous bag-of-words（CBOW）两类,但都是基于局部滑动窗口计算的。即，该方法利用了局部的上下文特征（local context）

LSA和word2vec作为两大类方法的代表，一个是利用了全局特征的矩阵分解方法，一个是利用局部上下文的方法。

GloVe模型就是将这两中特征合并到一起的，即使用了语料库的全局统计（overall statistics）特征，也使用了局部的上下文特征（即滑动窗口）。为了做到这一点GloVe模型引入了Co-occurrence Probabilities Matrix。

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191130gloveexaminaion.jpg)






## FastText介绍

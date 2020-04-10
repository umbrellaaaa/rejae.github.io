---
layout:     post
title:      Bert范式
subtitle:   
date:       2020-4-2
author:     RJ
header-img: 
catalog: true
tags:
    - nlp

---
<p id = "build"></p>
---

<h1>Bert 模式</h1>

Bert: 掩码某个字，通过上下文预测。是一种DAE模型，破坏重建的阅读理解形式，使token与上下文产生丰富的联系，从而得到更好的词向量。


## span bert
[span bert](https://arxiv.org/pdf/1907.10529.pdf)

- First, we mask random contiguous spans, rather than random individual tokens.
- Second, we introduce a novel span-boundary objective (SBO) so the model learns to predict the entire masked span from the observed tokens at its boundary. Span-based masking forces the model to predict entire spans solely using the context in which they appear.
- Furthermore, the span-boundary objective encourages the model to store this span-level information at the boundary tokens, which can be easily accessed during the fine-tuning stage. Figure 1 illustrates our approach.

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20200402233653.png)
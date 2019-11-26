---
layout:     post
title:      A Neural Attention Model for Abstractive Sentence Summarization
subtitle:   
date:       2019-11-26
author:     RJ
header-img: 
catalog: true
tags:
    - Life

---
<p id = "build"></p>
---

## 论文
抽象式文本摘要：《[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/pdf/1509.00685.pdf)》date:2019-11-26~27

这是一篇2015/9/3号FaceBook发的一篇关于抽象式文本摘要的论文，当作我在文本摘要方向的一个领路人吧。

## 摘要
- Summarization based on text extraction is inherently limited, but generation-style abstractive methods have proven challenging to build. In this work, we propose a fully data-driven approach to abstractive sentence summarization. 
- Our method utilizes **a local attention-based model** that generates each word of the summary conditioned on the input sentence. While the model is structurally simple, it can easily be trained end-to-end and scales to a
large amount of training data. The model shows significant performance gains on the DUC-2004 shared task compared with several strong baselines.

抽取式摘要太局限，生成抽象式摘要却较难实现，FaceBook提出了一个基于局部注意力的模型，在输入句子的相关条件概率下，生成抽象式摘要的每个词。

## 1. 引入
- Summarization is an important challenge of natural language understanding. The aim is to produce
a condensed representation of an input text that captures the core meaning of the original.
- Most successful summarization systems utilize extractive approaches that **crop out and stitch together portions of the text** to produce a condensed version. In contrast, abstractive summarization attempts to produce a bottom-up summary, aspects of which may not appear as part of the original.

- We focus on the task of sentence-level summarization. While much work on this task has looked at deletion-based sentence compression techniques (Knight and Marcu (2002), among many others), studies of human summarizers show that it is common to apply various other operations while condensing, such as paraphrasing, generalization, and reordering (Jing, 2002).

在进行摘要时应用各种其他操作是很常见的，比如释义、概括和重新排序

- Past work has modeled this abstractive summarization problem either using linguistically-inspired constraints
(Dorr et al., 2003; Zajic et al., 2004) or with syntactic transformations of the input text (Cohn and Lapata, 2008; Woodsend et al., 2010). These approaches are described in more detail **in Section 6**. 
- We instead explore a fully data-driven approach for generating abstractive summaries. Inspired by the recent success of neural machine translation, we combine a neural language model with a contextual input encoder.
- Our encoder is modeled off of the attention-based encoder of Bahdanau et al. (2014) in that it learns a latent soft alignment over the input text to help inform the summary (as shown in Figure 1). Crucially both the encoder and the generation model are trained jointly on the sentence summarization task.

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126ABS1.jpg)

- The model is described in detail **in Section 3**. Our model also incorporates a beam-search decoder as well as additional features to model extractive elements; these aspects are discussed **in Sections 4 and 5**.
- This approach to summarization, which we call Attention-Based Summarization (ABS), incorporates less linguistic structure than comparable abstractive summarization approaches, but can easily scale to train on a large amount of data. Since our system makes no assumptions about the vocabulary of the generated summary it can be trained directly on any document-summary pair.1 This allows us to train a summarization model for headline-generation on a corpus of article pairs from Gigaword (Graff et al., 2003) consisting of
around 4 million articles. An example of generation is given in Figure 2, and we discuss the details
of this task **in Section 7**.

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126ABS2.jpg)

- To test the effectiveness of this approach we run extensive comparisons with multiple abstractive and extractive baselines, including traditional syntax-based systems, integer linear programconstrained systems, information-retrieval style approaches, as well as statistical phrase-based machine translation.
- **Section 8** describes the results of these experiments. Our approach outperforms a machine translation system trained on the same large-scale dataset and yields a large improvement over the highest scoring system in the DUC-2004 competition.
---
layout:     post
title:      Tidy Paper
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

## 论文集


《[Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)》
- date:2019-3

《[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)》
- date:2019-3

《[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)》
- date:2019-6

《[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)》
- date:2019-8

《[Unsupervised data augmentation for consistency training](https://arxiv.org/pdf/1904.12848.pdf)》date:2019-8

《[ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://openreview.net/pdf?id=H1eA7AEtvS)》
- date:2019-11-20~22


《[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)》 
- date:2019-11-22~25

《[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/pdf/1509.00685.pdf)》
- date:2019-11-26~27

《[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://www.aclweb.org/anthology/P18-1063.pdf)》date:2019-11-27~28

《[Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)》date:2019-12-3

《GCN》

《2018-11-GAUSSIAN ERROR LINEAR UNITS (GELUS).pdf》

《2019-9-BottleSum Unsupervised and Self-supervised Sentence Summarization.pdf》


## 文本摘要汇总

https://hongyunyan.github.io/2019/02/20/%E6%89%AB%E6%96%87%E7%AC%94%E8%AE%B02/#more
Get To The Point: Summarization with Pointer-Generator Networks结合了Extractive和Abstractive两种方式来做文本摘要

Multi-Source Pointer Network for Product Title Summarization则是偏重Extractive方向


## 经典Blog

[illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

[knowledge-distillation](https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322)




## 中文纠错论文

[Chinese Spelling Check Evaluation at SIGHAN Bake-off 2013](https://www.aclweb.org/anthology/W13-4406.pdf)

### 检错与纠错
A spelling checker should have both capabilities consisting of error detection and error correction. Spelling error detection is to indicate the various types of spelling errors in the text. Spelling error correction is further to suggest the correct characters of detected errors.

For chinese: There are no word delimiters between words and the length of each word is very short. There are several previous studies addressing the Chinese spelling check problem.

### 历史解决方案
- Chang (1995) has proposed a bi-gram language model to substitute the confusing character for error detection and correction.
- Zhang et al. (2000) have presented an approximate word-matching algorithm to detect and correct Chinese spelling errors using operations of character substitution, insertion, and deletion.
- Ren et al. (2001) have proposed a hybrid approach that combines a rule-based method and a probability-based method to automatic Chinese spelling checking.
- Huang et al.(2007) have proposed a learning model based on Chinese phonemic alphabet for spelling check. 
- Most of the Chinese spelling errors were originated from phonologically similar, visually similar, and semantically confusing characters (Liu et al., 2011). 
- Empirically, there were only 2 errors per student essay on average in a learners’ corpus
(Chen et al., 2011).

How to evaluate the falsealarm rate of a spelling check system with normal corpus was also a hard task (Wu et al., 2010). 

### 近期论文
[A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check](https://www.aclweb.org/anthology/D18-1273.pdf)

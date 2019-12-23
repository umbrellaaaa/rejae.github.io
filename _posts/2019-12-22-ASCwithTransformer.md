---
layout:     post
title:      ASC with transformer
subtitle:   
date:       2019-12-22
author:     RJ
header-img: 
catalog: true
tags:
    - Job
---
<p id = "build"></p>
---

## 前言
论文学习：

[Automatic Spelling Correction with Transformer for CTC-based End-to-End
Speech Recognition](https://arxiv.org/pdf/1904.10045.pdf)

## introduction
Conventional hybrid DNN-HMM based speech recognition system usually consists of acoustic, pronunciation and language models.

Recent works in this area attempt to rectify this disjoint training problem and simplify the training process by building speech recognition system in the so-called end-to-end framework. Two popular approaches for this are the Connectionist Temporal Classification (CTC) [10] and attention-based encoder-decoder models [11].

CTC uses intermediate label representation allowing repetitions of labels and occurrences of blank
label to identify less informative frames, which enables CTCbased acoustic models to automatically learn the alignments between speech frames and target labels.

attention based models use an attention mechanism to perform alignment between acoustic frames and recognized symbols. 
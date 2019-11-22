---
layout:     post
title:      Distilling the Knowledge in a Neural Network
subtitle:   summary
date:       2019-11-23
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 前言
随着深度学习模型越来越大，效果也不断提升的同时，所需要的计算量和成本也在不断的提高，在保证模型效果差不多的情况下，压缩模型的参数变得尤为重要，就Bert模型而言，常见的压缩模型参数的方法如下：

1. Pruning 
    - Removes unnecessary parts of the network after training. This includes weight magnitude pruning, attention head pruning, layers, and others. Some methods also impose regularization during training to increase prunability (layer dropout).

2. Weight Factorization 
    - Approximates parameter matrices by factorizing them into a multiplication of two smaller matrices. This imposes a low-rank constraint on the matrix. Weight factorization can be applied to both token embeddings (which saves a lot of memory on disk) or parameters in feed-forward / self-attention layers (for some speed improvements).

3. Knowledge Distillation 
    - Aka “Student Teacher.” Trains a much smaller Transformer from scratch on the pre-training / downstream-data. Normally this would fail, but utilizing soft labels from a fully-sized model improves optimization for unknown reasons. Some methods also distill BERT into different architectures (LSTMS, etc.) which have faster inference times. Others dig deeper into the teacher, looking not just at the output but at weight matrices and hidden activations.

4. Weight Sharing 
    - Some weights in the model share the same value as other parameters in the model. For example, ALBERT uses the same weight matrices for every single layer of self-attention in BERT.

5. Quantization 
    - Truncates floating point numbers to only use a few bits (which causes round-off error). The quantization values can also be learned either during or after training.

6. Pre-train vs. Downstream 
    - Some methods only compress BERT w.r.t. certain downstream tasks. Others compress BERT in a way that is task-agnostic.


知识蒸馏手段是其中之一，今天将要学习知识蒸馏的开山之作，论文 ：<br>
[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

##




## 参考
[All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)

---
layout:     post
title:      Text Generation a Survey 
subtitle:   summary
date:       2019-11-9
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## [Neural Text Generation: Past, Present and Beyond](https://arxiv.org/pdf/1803.07133.pdf)

## 摘要
This paper presents a systematic survey on recent development of neural text generation models. 
- Specifically, we start from recurrent neural network language models with the traditional maximum likelihood estimation training scheme and point out its shortcoming for text generation. 
- We thus introduce the recently proposed methods for text generation based on reinforcement learning, re-parametrization tricks and generative adversarial nets (GAN) techniques. 
- We compare different properties of these models and the corresponding techniques to handle their common problems such as gradient vanishing and generation diversity. 
- Finally, we conduct a benchmarking experiment with different types of neural text generation models on two well-known datasets and discuss the empirical results along with the aforementioned model properties.

summary: 介绍了传统的RNN+MLE的模型并指出其缺点； 引出了当前TG的前沿方法RL和GAN； 对比这些模型和相应方法来处理这些模型共同的问题如：梯度消失和文本多样性。

## 2. On Training Paradigms of RNNLMs

- 监督学习中的NTG一般采用MLE, 但是MLE会产生exposure bias(暴露偏差)。
There is no guarantee that the model will still behave normally in those cases where the prefixs are a little bit different from those in the training data. The effect of exposure bias becomes more obvious and serious as the sequence becomes longer, making MLE less useful when the model is applied to long text
generation tasks.



## 总结
- This paper presents an overview of the classic and recently proposed neural text generation models. The development of RNNLMs are discussed in detail with three training paradigms, namely supervised learning, reinforcement learning and adversarial training. 
- Supervised learning methods with MLE objective are the most widely adopted solution for NTG but they probably cause **exposure bias** problem. RL-based and adversarial training methods could address exposure bias but usually suffer from **gradient vanishing and mode collapse** problems. Thus various techniques, including reward rescaling and hierarchical architectures, are proposed
to alleviate such problems. This paper also provides a unified view of MLE and RL-based models, which also explains why pretraining with MLE is usually necessary in for RL-based models. 
- The paper also raises a question about whether the effectiveness of RNNLMs is still limited, along with an opinion and corresponding reasons. We hope that this paper could shed a new light on neural text generation landscape and its future research.

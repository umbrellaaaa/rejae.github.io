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

summary: 介绍了传统的RNN+MLE的模型并指出其缺点； 引出了当前TG的前沿方法RL和GAN； 对比这些模型和相应方法来处理这些模型共同的问题如：**梯度消失和文本多样性**。

## 1. Introduction
Given the ground truth sequence sn = [x0, x1, ..., xn−1] and <br>
a θ-parametrized language model Gθ(x|context) = Pˆ(x|context) (similarly hereinafter), <br>
a typical NNLM adopts an approximation as:<br>

**Pˆ(xt|context) ≈ P(xt|xt−n+1, xt−n+2, ..., xt−1)**

However, the n-gram paradigm is theoretically impossible to capture long-term dependencies, according to some previous criticism [Rosenfeld, 2000]. To address this problem, recurrent neural network language model (RNNLM) [Mikolov et al., 2010] is developed, which is a more general implementation for a language model with Markov property. A typical RNNLM uses a recurrent neural network (RNN) to auto-regressively encode previous variant-length inputs into a “hidden” vector, which is then used during the inference of the next token. This procedure can be formulated as:<br>

**Pˆ(xt|context) ≈ P(xt|RNN(x0, x1, ..., xt−1))** <br>

之前做的charRNN Model就属于这种。

In this paper,  We carefully discuss different properties of these models and the corresponding techniques to handle their common problems such as gradient vanishing during training and generation diversity. Compared to a previous work [Xie, 2017] that is mainly on sequence-to-sequence (Seq2Seq) models, this paper focuses more on recently proposed **RL and GAN based methods** while Seq2Seq is a special case of the basic MLE methods. Finally, we conduct a benchmarking experiment with different types of neural text generation models on two well-known datasets and discuss the empirical results along with the aforementioned model properties. We hope this paper could provide with useful directions for further researches in this field.

可以再试试seq2seq做文本生成。

## 2. On Training Paradigms of RNNLMs

- 监督学习中的NTG一般采用MLE, 但是MLE会产生exposure bias(暴露偏差)。
There is no guarantee that the model will still behave normally in those cases where the prefixs are a little bit different from those in the training data. The effect of exposure bias becomes more obvious and serious as the sequence becomes longer, making MLE less useful when the model is applied to long text generation tasks.

### 2.1 NTG with Supervised Learning

**Maximum Likelihood Estimation**

Typically, classical neural language models are trained through maximum likelihood estimation (MLE, a.k.a. teacher forcing) [Williams and Zipser, 1989]. MLE is natural for RNNLMs, since it regards the generation problem as a sequential multi-label classification and then directly optimizes the multi-label cross entropy. The objective of a language model Gθ trained via MLE can be formulated as
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191122TGMLE.jpg)
where s0 represents the empty string.

It is important to note that, up to now, most existing applied state-of-the-art NTG models adopt MLE as their training objective [Karpathy and Fei-Fei, 2015; Hu et al., 2017]. MLE has better convergence rate and training robustness compared to other algorithms.

Although text generation is actually an unsupervised learning task, there do exist some supervised metric that are goodapproximations of the ground truth under some constraints. These algorithms focus on directly optimizing some supervised metric. Some of them may include someuseful tricksto help in alleviating some specific problems.

## 总结
- This paper presents an overview of the classic and recently proposed neural text generation models. The development of RNNLMs are discussed in detail with three training paradigms, namely supervised learning, reinforcement learning and adversarial training. 
- Supervised learning methods with MLE objective are the most widely adopted solution for NTG but they probably cause **exposure bias** problem. RL-based and adversarial training methods could address exposure bias but usually suffer from **gradient vanishing and mode collapse** problems. Thus various techniques, including reward rescaling and hierarchical architectures, are proposed
to alleviate such problems. This paper also provides a unified view of MLE and RL-based models, which also explains why pretraining with MLE is usually necessary in for RL-based models. 
- The paper also raises a question about whether the effectiveness of RNNLMs is still limited, along with an opinion and corresponding reasons. We hope that this paper could shed a new light on neural text generation landscape and its future research.

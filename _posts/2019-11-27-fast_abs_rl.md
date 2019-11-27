---
layout:     post
title:      
subtitle:   
date:       2019-11-27
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---
<p id = "build"></p>
---

# 1. 坚持每天看一篇论文

## 论文：
强化句子选择重写快速摘要：《[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://www.aclweb.org/anthology/P18-1063.pdf)》date:2019-11-27~28

本来打算花两天时间看抽象式文本摘要论文，但是毕竟是4年前的论文了，为了加快一点脚步，就不细细深究了，今天找到一篇比较近的论文，发布于2018年7月的ACL会议上，通过这篇文章应该能查阅文本摘要研究的前后历史。所以我们开始吧。

## 摘要
- Inspired by how humans summarize long documents, we propose an accurate and fast summarization model that first selects salient sentences and then rewrites them abstractively (i.e., compresses and paraphrases) to generate a concise overall summary.
- We use a novel sentence-level policy gradient method to bridge the nondifferentiable computation between these two neural networks in a hierarchical way, while maintaining language fluency.
- Empirically, we achieve the new state-of-theart on all metrics (including human evaluation) on the CNN/Daily Mail dataset, as well as significantly higher abstractiveness scores.
- Moreover, by first operating at the sentence-level and then the word-level, we enable parallel decoding of our neural generative model that results in substantially faster (10-20x) inference speed as well as 4x faster training convergence than previous long-paragraph encoder-decoder models. We also demonstrate the generalization of our model on the test-only DUC2002 dataset, where we achieve higher scores than a state-of-the-art model.

论文的出发点总是以人为本，人类怎样在很长的文章中作摘要的呢？第一步就是找显著的句子，然后通过压缩和释义的方式重写。作者使用了句子级的策略梯度方法在保持语言流畅性的同时，解决了两个神经网络(extractor,abstractor)之间的不可微计算，得到了很好的效果。


## 1. Introduction

- The task of document summarization has two main paradigms: extractive and abstractive. The former method directly chooses and outputs the salient sentences (or phrases) in the original document (Jing and McKeown, 2000; Knight and Marcu, 2000; Martins and Smith, 2009; BergKirkpatrick et al., 2011). The latter abstractive approach involves rewriting the summary (Banko et al., 2000; Zajic et al., 2004), and has seen substantial recent gains due to neural sequence-to-sequence models (Chopra et al., 2016; Nallapati et al., 2016; See et al., 2017; Paulus et al., 2018).

今天尝试一下seq2seq进行ABS.

- Abstractive models can be more concise by performing generation from scratch, but they suffer from slow and inaccurate encoding of very long documents, with the attention model being required to look at all encoded words (in long paragraphs) for decoding each generated summary word (slow, one by one sequentially). Abstractive models also suffer from redundancy (repetitions), especially when generating multi-sentence summary.

-----------------------------

- To address both these issues and combine the advantages of both paradigms, we propose a hybrid extractive-abstractive architecture, with policy-based reinforcement learning (RL) to bridge together the two networks.
- Similar to how humans summarize long documents, our model first uses an extractor agent to select salient sentences or highlights, and then employs an abstractor network to rewrite (i.e., compress and paraphrase) each of these extracted sentences.
- To overcome the non-differentiable behavior of our extractor and train on available document-summary pairs without saliency label, we next use actorcritic policy gradient with sentence-level metric rewards to connect these two neural networks and to learn sentence saliency.
- We also avoid common language fluency issues (Paulus et al., 2018) by **preventing the policy gradients from affecting the abstractive summarizer’s word-level training**, which is supported by our human evaluation study.
- Our sentence-level reinforcement learning takes into account the word-sentence hierarchy, which better models the language structure and makes parallelization possible.
- **Our extractor combines reinforcement learning and pointer networks, which is inspired by Bello et al. (2017)’s attempt to solve the Traveling Salesman Problem.**
- Our abstractor is a simple encoder-aligner-decoder model (with copying) and is trained on pseudo
document-summary sentence pairs obtained via simple automatic matching criteria.

---------------------------------------

- Thus, our method incorporates the abstractive paradigm’s advantages of concisely rewriting sentences and generating novel words from the full vocabulary, yet it adopts intermediate extractive behavior to improve the overall model’s quality, speed, and stability.
- Instead of encoding and attending to every word in the long input document sequentially, our model adopts a human-inspired coarse-to-fine approach that first extracts all the salient sentences and then decodes (rewrites) them (in parallel). This also avoids almost all redundancy issues because the model has already chosen non-redundant salient sentences to abstractively summarize (but adding an optional final reranker component does give additional gains by removing the fewer across-sentence repetitions).

--------------------------------------------

- Empirically, our approach is the new state-ofthe-art on all ROUGE metrics (Lin, 2004) as well
as on METEOR (Denkowski and Lavie, 2014) of the CNN/Daily Mail dataset, achieving statistically significant improvements over previous models that use complex long-encoder, copy, and coverage mechanisms (See et al., 2017).
- The test-only DUC-2002 improvement also shows our model’s better generalization than this strong abstractive system. In addition, we surpass the popular lead-3 baseline on all ROUGE scores with an
abstractive model.
- Moreover, our sentence-level abstractive rewriting module also produces substantially more (3x) novel N-grams that are not seen in the input document, as compared to the strong flat-structured model of See et al. (2017).
- This empirically justifies that our RL-guided extractor has learned sentence saliency, rather than
benefiting from simply copying longer sentences.
- We also show that our model maintains the same
level of fluency as a conventional RNN-based model because the reward does not leak to our abstractor’s word-level training.
- Finally, our model’s training is 4x and inference is more than 20x faster than the previous state-of-the-art. The optional final reranker gives further improvements while maintaining a 7x speedup.

----------------------------------------------------

**Overall, our contribution is three fold:** 

1. First we propose a novel sentence-level RL technique for the well-known task of abstractive summarization, effectively utilizing the word-then-sentence hierarchical structure without annotated matching sentence-pairs between the document and ground truth summary.
2. Next, our model achieves the new state-of-the-art on all metrics of multiple versions of a popular summarization dataset (as well as a test-only dataset) both extractively and abstractively, without loss in language fluency (also demonstrated via human evaluation and abstractiveness scores).
3. Finally, our parallel decoding results in a significant 10-20x speed-up over the previous best neural abstractive summarization system with even better accuracy.

引入读到这里，我是否需要进军强化学习的领域？2333...文中的策略梯度解决了两个模型之间不可微的关系，这不是很理解。

总的来说，该论文结合了抽取式和生成式各自的优势，增加了并行机制提高了速度，通过避免策略梯度影响ABS字级训练，增加了流畅性。其中extractor还结合了强化学习和指针网络。

## 2. Model
In this work, we consider the task of summarizing a given long text document into several (ordered) highlights, which are then combined to form a multi-sentence summary.Formally, given a training set of document-summary pairs：

<center>{xi, yi}<sup>N</sup><sub>i=1</sub></center>

our goal is to approximate the function h : 

<center>X → Y,  X = {xi}<sup>N</sup><sub>i=1</sub>, Y = {yi}<sup>N</sup><sub>i=1</sub></center>

such that h(xi) = yi , 1 ≤ i ≤ N. Furthermore, we assume there exists an abstracting function g defined as:∀s ∈ Si , ∃d ∈ Di such that g(d) = s, 1 ≤ i ≤ N,where Si is the set of summary sentences in xi and Di
the set of document sentences in yi. i.e., in any given pair of document and summary, every summary sentence can be produced from some document sentence. For simplicity, we omit subscript i in the remainder of the paper. 


![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191127model_symbol.jpg)

### 2.1 Extractor Agent

The extractor agent is designed to model f, which can be thought of as extracting salient sentences
from the document. We exploit a hierarchical neural model to learn the sentence representations of
the document and a ‘selection network’ to extract sentences based on their representations.

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191127figure1.jpg)

#### 2.1.1 Hierarchical Sentence Representation
We use a temporal convolutional model (Kim, 2014) to compute rj , the representation of each individual sentence in the documents (details in supplementary). To further incorporate global context of the document and capture the long-range semantic dependency between sentences, a bidirectional LSTM-RNN (Hochreiter and Schmidhuber, 1997; Schuster et al., 1997) is applied on the convolutional output. This enables learning a strong representation, denoted as hj for the j-th sentence in the document, that takes into account the context of all previous and future sentences in the same document.






# 2. 扩展NLP知识面


# 3. 实践知识到代码上


# 4. 一道leetcode, 几页剑指


# 5. 博客总结一天所学

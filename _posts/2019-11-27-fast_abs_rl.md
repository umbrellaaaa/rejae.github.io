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

使用CNN提取句子的特征，再传递到Bi-lstm中提取长距离的语义依赖来得到句子的更好表示。
### 2.1.2 Sentence Selection
Next, to select the extracted sentences based on the above sentence representations, we add another
LSTM-RNN to train a Pointer Network (Vinyals et al., 2015), to extract sentences recurrently. We
calculate the extraction probability by:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191127figure2.jpg)

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191127figure3.jpg)

### 2.2 Abstractor Network
The abstractor network approximates g, which compresses and paraphrases an extracted document sentence to a concise summary sentence. We use the standard encoder-aligner-decoder (Bahdanau et al., 2015; Luong et al., 2015). We add the copy mechanism to help directly copy some out-of-vocabulary (OOV) words (See et al., 2017). For more details, please refer to the supplementary.

## 3 Learning
- Given that our extractor performs a nondifferentiable hard extraction, we apply standard policy gradient methods to bridge the backpropagation and form an end-to-end trainable (stochastic) computation graph.
- However, simply starting from a randomly initialized network to train the whole model in an end-to-end fashion is infeasible. When randomly initialized, the extractor would often select sentences that are not relevant, so it would be difficult for the abstractor to learn to abstractively rewrite.
- On the other hand, without a well-trained abstractor the extractor would get noisy reward, which leads to a bad estimate of the policy gradient and a sub-optimal policy.
- **We hence propose optimizing each sub-module separately using maximumlikelihood (ML) objectives: train the extractor to select salient sentences (fit f) and the abstractor to generate shortened summary (fit g). Finally, RL is applied to train the full model end-to-end (fit h).**

随机初始化参数训练模型，会导致extractor抽取不相干的句子出来，这样使得abstractor的抽象重写变得困难。与此同时，没有良好的abstractor的抽象结果，会给extractor带去噪声奖励，所以作者通过最大似然目标分别优化二者。

### 3.1 Maximum-Likelihood Training for Submodules

### 3.2 Reinforce-Guided Extraction

### 3.3 Repetition-Avoiding Reranking



## 4 Related Work
- Early summarization works mostly focused on extractive and compression based methods (Jing and
McKeown, 2000; Knight and Marcu, 2000; Clarke and Lapata, 2010; Berg-Kirkpatrick et al., 2011;
Filippova et al., 2015).
- Recent large-sized corpora attracted neural methods for abstractive summarization (Rush et al., 2015; Chopra et al., 2016). Some of the recent success in neural abstractive models include hierarchical attention (Nallapati et al., 2016), coverage (Suzuki and Nagata, 2016; Chen et al., 2016; See et al., 2017), RL based metric optimization (Paulus et al., 2018), graph-based attention (Tan et al., 2017), and the copy mechanism (Miao and Blunsom, 2016; Gu et al., 2016; See et al., 2017).

------------------------------------------

- Our model shares some high-level intuition with extract-then-compress methods. Earlier attempts
in this paradigm used Hidden Markov Models and rule-based systems (Jing and McKeown, 2000), statistical models based on parse trees (Knight and Marcu, 2000), and integer linear programming based methods (Martins and Smith, 2009; Gillick and Favre, 2009; Clarke and Lapata, 2010; BergKirkpatrick et al., 2011). Recent approaches investigated discourse structures (Louis et al., 2010; Hirao et al., 2013; Kikuchi et al., 2014; Wang et al., 2015), graph cuts (Qian and Liu, 2013), and parse trees (Li et al., 2014; Bing et al., 2015). For neural models, Cheng and Lapata (2016) used a second neural net to select words from an extractor’s output. Our abstractor does not merely ‘compress’ the sentences but generatively produce novel words. Moreover, our RL bridges the extractor and the abstractor for end-to-end training.

------------------------------------------

**Reinforcement learning has been used to optimize the non-differential metrics of language generation and to mitigate exposure bias (Ranzato et al., 2016; Bahdanau et al., 2017).**

- Henß et al. (2015) use Q-learning based RL for extractive summarization.
- Paulus et al. (2018) use RL policy gradient methods for abstractive summarization
- utilizing sequence-level metric rewards with curriculum learning (Ranzato et al., 2016) 
- utilizing weighted ML+RL mixed loss (Paulus et al., 2018) for stability and language fluency.
- We use sentence-level rewards to optimize the extractor while keeping our ML trained abstractor decoder fixed, so as to achieve the best of both worlds.

------------------------------------------

- Training a neural network to use another fixed network has been investigated in machine translation for better decoding (Gu et al., 2017a) and real-time translation (Gu et al., 2017b).
- They used a fixed pretrained translator and applied policy gradient techniques to train another task-specific network.
- In question answering (QA), Choi et al. (2017) extract one sentence and then generate the answer from the sentence’s vector representation with RL bridging.
- Another recent work attempted a new coarse-to-fine attention approach on summarization (Ling and Rush, 2017) and found desired sharp focus properties for scaling to larger inputs (though without metric improvements).
- Very recently (concurrently), Narayan et al. (2018) use RL for ranking sentences in pure extraction-based summarization and C¸ elikyilmaz et al. (2018) investigate multiple communicating encoder agents to enhance the copying abstractive summarizer.


Finally, there are some loosely-related recent works: 
- Zhou et al. (2017) proposed selective gate to improve the attention in abstractive summarization.
- Tan et al. (2018) used an extract-then-synthesis approach on QA, where an extraction model predicts the important spans in the passage and then another synthesis model generates the final answer. 
- Swayamdipta et al. (2017) attempted cascaded non-recurrent small networks on extractive QA, resulting a scalable, parallelizable model.
- Fan et al. (2017) added controlling parameters to adapt the summary to length, style, and entity preferences. However, none of these used RL to bridge the non-differentiability of neural models.



# 2. 扩展NLP知识面

## pointer-network
什么是pointer-network?

Pointer network 主要用在解决组合优化类问题(TSP, Convex Hull等等)，实际上是Sequence to Sequence learning中encoder RNN和decoder RNN的扩展，主要解决的问题是输出的字典长度不固定问题（输出字典的长度等于输入序列的长度）。

在传统的NLP问题中，采用Sequence to Sequence learning的方式去解决翻译问题，其输出向量的长度往往是字典的长度，而字典长度是事先已经订好了的（比如英语单词字典就定n=8000个单词）。而在组合优化类问题中，比如TSP(旅行商)问题，输入是城市的坐标序列，输出也是城市的坐标序列，而每次求解的TSP问题城市规模n是不固定的。每次decoder的输出实际上是每个城市这次可能被选择的概率向量，其维度为n，和encoder输入的序列向量长度一致。如何解决输出字典维度可变的问题？Pointer network的关键点在如下公式:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191127TSP.jpg)

其中 e<sub>j</sub> 是encoder的在时间序列j次的隐藏层输出， d<sub>i</sub> 是decoder在时间序列i次的隐藏状态输出，这里的 ui=[u<sup>i</sup><sub>1</sub>,...mu<sup>i</sup><sub>j</sub>] 其维度为n维和输入保持一致，对 u<sub>i</sub>  直接求softmax就可以得到输出字典的概率向量，其输出的向量维度和输入保持一致。其中 v<sup>T</sup>,W<sub>1</sub>,W<sub>2</sub>  均为固定维度的参数，可被训练出来。

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/Ptr_net.jpg)

如何实现pointer-network?

我们首先来看传统注意力机制的公式：
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/attention.jpg)

其中是encoder的隐状态，而是decoder的隐状态，v,W1,W2都是可学习的参数，在得到之后对其执行softmax操作即得到。这里的就是分配给输入序列的权重，依据该权重求加权和，然后把得到的拼接（或者加和）到decoder的隐状态上，最后让decoder部分根据拼接后新的隐状态进行解码和预测。

根据传统的注意力机制，作者想到，所谓的正是针对输入序列的权重，完全可以把它拿出来作为指向输入序列的指针，在每次预测一个元素的时候找到输入序列中权重最大的那个元素不就好了嘛！于是作者就按照这个思路对传统注意力机制进行了修改和简化，公式变成了这个样子：
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191127pointernetworkf.jpg)

第一个公式和之前没有区别，然后第二个公式则是说Pointer Networks直接将softmax之后得到的当成了输出，其承担指向输入序列特定元素的指针角色。

所以总结一下，传统的带有注意力机制的seq2seq模型的运行过程是这样的，先使用encoder部分对输入序列进行编码，然后对编码后的向量做attention，最后使用decoder部分对attention后的向量进行解码从而得到预测结果。但是作为Pointer Networks，得到预测结果的方式便是输出一个概率分布，也即所谓的指针。换句话说，传统带有注意力机制的seq2seq模型输出的是针对输出词汇表的一个概率分布，而Pointer Networks输出的则是针对输入文本序列的概率分布。


## copy-machinism

# 3. 实践知识到代码上
调试seq2seq, google代码过去了几年，由于版本更新和windows的不适配，遇到了很多难题，好在都调试通过了，写到知乎上总结一下吧。

# 4. 一道leetcode, 几页剑指


# 5. 博客总结一天所学
- 今天主要是看了seq2seq相关的代码，调试了一个简单的机器翻译，而teacher forcing，attention，beam search需要进一步学习。
- 整理了一块NLP面试的常见问题集，每天看看并且作答一下。
- 至于FastText需要掌握，明天学习和调试。
- 爬虫+验证码+卷积图像识别项目也要慢慢跟进 
- 文本摘要的Seq2seq明天接着做

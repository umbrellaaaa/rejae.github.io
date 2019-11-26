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

# 每天坚持一篇论文

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

## 2. 背景
<sub></sub>  <sup></sup>
We begin by defining the sentence summarizationtask. Given an input sentence, the goal is to produce a condensed summary. Let the input consist of a sequence of M words x<sub>1</sub>, . . . , x<sub>M</sub> coming from a fixed vocabulary V of size |V| = V .
We will represent each word as an indicator vector

x<sub>i</sub> ∈ {0, 1}<sup>V</sup> for i ∈ {1, . . . , M}, 

sentences as a sequence of indicators, and X as the set of possible inputs. Furthermore define the notation x[i,j,k] to indicate the sub-sequence of elements i, j, k.
A summarizer takes x as input and outputs a shortened sentence y of length N < M. We will assume that the words in the summary also come from the same vocabulary V and that the output is a sequence y1, . . . , yN . Note that in contrast to related tasks, like machine translation, we will assume that the output length N is fixed, and that the system knows the length of the summary before generation.Next consider the problem of generating summaries. Define the set

Y ⊂ ({0, 1}<sup>V</sup> , . . . , {0, 1}<sup>V</sup> ) 

as all possible sentences of length N, i.e. for all i and y ∈ Y, yi is an indicator. We say a system is abstractive if it tries to find the optimal sequence from this set Y,
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126formula1.jpg)

under a scoring function s : X ×Y → R. Contrast this to a fully extractive sentence summary which transfers words from the input:
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126formula2.jpg)

or to the related problem of sentence compression that concentrates on deleting words from the input:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126formula3.jpg)

While abstractive summarization poses a more difficult generation challenge, the lack of hard constraints gives the system more freedom in generation and allows it to fit with a wider range of training data.
In this work we focus on factored scoring functions, s, that take into account a fixed window of
previous words:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126formula4.jpg)

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126paper1.jpg)

**即句子x and 窗口y<sub>c</sub>**=映射=>> 下一字y<sub>i+1</sub>

## Model
The distribution of interest, p(yi+1|x, yc; θ), is a conditional language model based on the input sentence x. Past work on summarization and compression has used a noisy-channel approach to split and independently estimate a language model and a conditional summarization model (Banko et al., 2000; Knight and Marcu, 2002; Daume III and ´ Marcu, 2002), i.e.,

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126f1.jpg)
where p(y) and p(x|y) are estimated separately.
Here we instead follow work in neural machine translation and directly parameterize the original distribution as a neural network. The network contains both a neural probabilistic language model and an encoder which acts as a conditional summarization model.

上式中，p(y)和p(x|y)是分开计算的，而我们的网络模型不这样做，新模型依照NMT直接参数化原始分布为一个神经网络，该模型包含一个神经概率语言模型和一个作为条件摘要模型的编码器。

### 3.1 Neural Language Model
The core of our parameterization is a language model for estimating the contextual probability of
the next word. The language model is adapted from a standard feed-forward neural network language model (NNLM), particularly the class of NNLMs described by Bengio et al. (2003). The full model is:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126f2.jpg)
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126f3.jpg)

### 3.2 Encoders
Note that without the encoder term this represents a standard language model. By incorporating in enc and training the two elements jointly we crucially can incorporate the input text into generation. We discuss next several possible instantiations of the encoder. Bag-of-Words Encoder Our most basic model simply uses the bag-of-words of the input sentence embedded down to size H, while ignoring properties of the original order or relationships between neighboring words. We write this model as:
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191126f4.jpg)

For summarization this model can capture the relative importance of words to distinguish content words from stop words or embellishments. Potentially the model can also learn to combine words; although it is inherently limited in representing contiguous phrases.


## 参考
[paperweekly](https://zhuanlan.zhihu.com/p/21388469)






# 2. 扩展NLP知识面
[好的想法从哪里来？](https://zhuanlan.zhihu.com/p/93765082)作者清华教授知远先生今天发的知乎文章：

什么算是好的想法？ 好————学科发展角度的”好“；研究实践角度的”好“

如何产生新的想法呢？总结有三种可行的基本途径：实践法；类比法；组合法

初学者应该怎么做？

与阅读论文、撰写论文、设计实验等环节相比，如何产生好的研究想法，是一个不太有章可循的环节，很难总结出固定的范式可供遵循。像小马过河，需要通过大量训练实践，来积累自己的研究经验。不过，对于初学者而言，仍然有几个简单可行的原则可以参考。
- 一篇论文的可发表价值，取决于它与已有最直接相关工作间的Delta
- 兼顾摘果子和啃骨头
- 注意多项研究工作的主题连贯性
- 注意总结和把握研究动态和趋势，因时而动

补充：

学术研究和论文发表，对个人而言也许意味着高薪资和奖学金，但其最终的目的还是真正的推动学科的发展。所以，要做经得起考验的学术研究，关键就在”真“与”新“，需要我们始终恪守和孜孜以求。

著名历史学家、清华校友何炳棣先生曾在自传《读史阅世六十年》中提及著名数学家林家翘的一句嘱咐：“要紧的是不管搞哪一行，千万不要做第二等的题目。” 具体到每个领域，什么是一等的题目本身见仁见智，其实更指向内心“求真”的态度。

# 3. 实践知识到代码上


# 4. 一道leetcode, 几页剑指


# 5. 博客总结一天所学

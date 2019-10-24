---
layout:     post
title:      BLEU Score
subtitle:   the evaluation of NMT
date:       2019-10-24
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---
## 前言
之前看论文的实验结果的时候，有一个叫BLEU的评价指标，一直不知道其原理，通过这篇论文<br>
[BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)<br>
深入了解这个评价机制及其原理<br>
一．BLEU是什么？
　　
首先要看清楚我们本篇文章的主人公是怎么拼写的——{B-L-E-U}，而不是{B-L-U-E}，简直了…..我叫了它两天的blue（蓝色）才发现原来e在u之前~~如果真要念出它的名字，音标是这样的：[blε：][blε：]（波勒）。 
　　 
　　BLEU的全名为：bilingual evaluation understudy，即：双语互译质量评估辅助工具。它是用来评估机器翻译质量的工具。当然评估翻译质量这种事本应该由人来做，机器现在是无论如何也做不到像人类一样思考判断的（我想这就是自然语言处理现在遇到的瓶颈吧，随便某个方面都有牵扯上人类思维的地方，真难），但是人工处理过于耗时费力，所以才有了BLEU算法。

　　BLEU的设计思想与评判机器翻译好坏的思想是一致的：机器翻译结果越接近专业人工翻译的结果，则越好。BLEU算法实际上在做的事：判断两个句子的相似程度。我想知道一个句子翻译前后的表示是否意思一致，显然没法直接比较，那我就拿这个句子的标准人工翻译与我的机器翻译的结果作比较，如果它们是很相似的，说明我的翻译很成功。因此，BLUE去做判断：一句机器翻译的话与其相对应的几个参考翻译作比较，算出一个综合分数。这个分数越高说明机器翻译得越好。（注：BLEU算法是句子之间的比较，不是词组，也不是段落） 
　　 
　　BLEU是做不到百分百的准确的，它只能做到个大概判断，它的目标也只是给出一个快且不差自动评估解决方案。

二．BLEU的优缺点有哪些？<br>
　　
优点很明显：方便、快速、结果有参考价值 <br>
　　 
缺点也不少，主要有： <br>

    1. 不考虑语言表达（语法）上的准确性； 
    2. 测评精度会受常用词的干扰； 
    3. 短译句的测评精度有时会较高； 
    4. 没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定；



<p id = "build"></p>
---
<h1>BLEU: a Method for Automatic Evaluation of Machine Translation</h1>



<h2>Introduction </h2>
这个评价机制出现的原因是由于早期机器翻译需要大量的人工进行评价，效率很低，花费很高，且无法重用。<br>
一个评价机器翻译性能的指标应该具备哪些条件呢？<Br>
1. a numerical “translation closeness” metric
2. a corpus of good quality human reference translations

即首先可以进行数值度量，其次还需要一个高质量的人类翻译作参考。

We fashion our closeness metric after the highly successful word error rate metric used by the speech recognition community, appropriately modified for multiple reference translations and allowing for legitimate differences in word choice and word order. 

**The main idea** is to use a weighted average of variable length phrase matches against the reference translations. This view gives rise to a family of metrics using various weighting schemes.

- In Section 2, we describe the baseline metric in
detail. 
- In Section 3, we evaluate the performance of
BLEU. 
- In Section 4, we describe a human evaluation
experiment. 
- In Section 5, we compare our baseline
metric performance with human evaluations.

<h2>2 The Baseline BLEU Metric</h2>

example1:
```python
Example 1.
Candidate 1: It is a guide to action which
ensures that the military always obeys
the commands of the party.
Candidate 2: It is to insure the troops
forever hearing the activity guidebook
that party direct.
Although they appear to be on the same subject, they
differ markedly in quality. For comparison, we provide three reference human translations of the same
sentence below.
Reference 1: It is a guide to action that
ensures that the military will forever
heed Party commands.
Reference 2: It is the guiding principle
which guarantees the military forces
always being under the command of the
Party.
Reference 3: It is the practical guide for
the army always to heed the directions
of the party.
```
简单的比较Candidate和reference中的词贴合程度就可以看出candidate1优于candidate2。

The primary programming task for a BLEU implementor is to compare n-grams of the candidate with
the n-grams of the reference translation and count
the number of matches. These matches are positionindependent. The more the matches, the better the
candidate translation is. For simplicity, we first focus on computing unigram matches.

BLEU主要任务是比较candidate与reference中的n-grams，以及相应n-grams的count。

简单的unigram在使用时，MT系统可能会过度生成“合理”的单词，从而导致不太可能但却非常精确的译文，如下面的示例2所示。

```python
Example 2.
Candidate: the the the the the the the.
Reference 1: The cat is on the mat.
Reference 2: There is a cat on the mat.
Modified Unigram Precision = 2/7
```

Modified n-gram precision is computed similarly
for any n: all candidate n-gram counts and their
corresponding maximum reference counts are collected. 

The candidate counts are **clipped by their
corresponding reference maximum value**, summed,
and divided by the total number of candidate ngrams

修改后的unigram裁剪了单词the的重复次数，因为在reference中的最大次数也才2，所以避免了过度生成单词的句子还能获得较高分数的缺陷。


至此，简单的评价与改进的多元精度评价我们已经大概了解。改进的多元精度（modified n-gram precision）在文本段落翻译质量评估中的使用，只不过是把多个句子当成一个句子罢了。

随后将多个改进的多元精度（modified n-gram precision）进行组合，再加上译句较短惩罚（Sentence brevity penalty ），就得到了最终的公式：<br>
BLEU=BP∗exp(∑n=1~N  wn∗logpn)   <br>

BP={1  if…c>r  else  exp(1−r/c)}


## 后记
参考：
[机器翻译自动评估-BLEU算法详解](https://blog.csdn.net/qq_31584157/article/details/77709454)
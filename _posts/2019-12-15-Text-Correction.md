---
layout:     post
title:      中文纠错综述
subtitle:   
date:       2019-12-15
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---
<p id = "build"></p>
---

## 前言
接下来一段时间都集中在中文纠错方面的内容上，尤其是拼音纠错这一块。面对未知，最好的办法就是从全局到局部，对目标有十分清楚的认识才能将工作完成的更好。

## 1. 问题分析
### 1.1 definition
中文文本纠错，常见类型包括：

- 谐音字词纠错，如 配副眼睛-配副眼镜
- 混淆音字词纠错，如 流浪织女-牛郎织女
- 字词顺序颠倒纠错，如 伍迪艾伦-艾伦伍迪
- 字词补全，如 爱有天意-假如爱有天意
- 形似字纠错，如 高梁-高粱
- 中文拼音推导，如 xingfu-幸福
- 中文拼音缩写推导，如 sz-深圳
- 语法错误，如 想象难以-难以想象

当然，针对确定场景，这些问题并不一定全部存在，比如输入法中需要处理1,2,3,4，搜索引擎需要处理1,2,3,4,5,6,7，ASR 后文本纠错只需要处理1、2，其中5主要针对五笔或者笔画手写输入等。

### 1.2 features
paper 上方案大多是基于英文纠错的，但中英文纠错区别还是蛮大的。了解清这些区别，有利于我们进行算法选型和调优工作。

- 边界词: 由于中文不存在词边界，一方面导致纠错时必须考虑上下文，另一方面拼写错误也会导致分词结果变化。(这就要求尽量基于字符进行建模，保证召回。)

- 字符集: 英文拼写错误通常与正确拼写的编辑距离在1-2，且由于英文字符集仅有26个，可以简单地针对字符级别的错误进行建模。可是中文常用字符集约7000。(这就要求具体计算过程中非常注意效率。)

- 错误类型: 英文拼写错误通常为 insert、delete、substitute 和 ，而由于中文字符通常可以单独成词，insert、delete、transposite 则体现在了中文的语法错误，通常的拼写错误主要集中于 transposite。(这就要求距离计算中充分优化超参，以突显某些操作的重要性。)

- session 信息: 交互环境可以提供大量参考信息，如领域、候选词表、热度词表等。(这就要求要充分利用 session 信息，并提供多级算法。)

### 1.3 evaluation
- 评测数据

中文输入纠错的评测数据主要包括 SIGHAN Bake-off 2013/2014/2015 这三个数据集，均是针对繁体字进行的纠错。其中，只有 SIGHAN Bake-off 2013 是针对母语使用者的，而另外两个是针对非母语使用者。

- 训练数据

虽然没有公开训练数据，但在明确特定场景下纠错任务的 Features 后，我们很容易根据正确文本，通过增删改构造大量的训练样本。

- 评价指标

虽然文本纠错具体会分为错误识别和错误修正两部分，并分别构造评价指标。但考虑到端到端任务，我们评价完整的纠错过程：

该纠的，即有错文本记为 P，不该纠的，即无错文本记为 N
对于该纠的，纠对了，记为 TP，纠错了或未纠，记为 FP
对于不该纠的，未纠，记为 TN，纠了，记为 FN。
通常场景下，差准比查全更重要，FN 更难接受，可构造下述评价指标：

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191215Fscore.png)
​	


## 2. 主流技术
中文本纠错的 paper 很多，整体来看，可以统一在一个框架下，即三大步：

- 错误识别: 该阶段主要目的在于，判断文本是否存在错误需要纠正，如果存在则传递到后面两层。
这一阶段可以提高整体流程的效率。

- 生成纠正候选: 该阶段主要目的在于，利用一种或多种策略（规则或模型），生成针对原句的纠正候选。
这一阶段是整体流程召回率的保证，同时也是一个模型的上限。

- 评价纠正候选: 该阶段主要目的在于，在上一阶段基础上，利用某种评分函数或分类器，结合局部乃至全局的特征，针对纠正候选进行排序，最终排序最高（如没有错误识别阶段，则仍需比原句评分更高或评分比值高过阈值，否则认为不需纠错）的纠正候选作为最终纠错结果。

大部分的模型基本上可以划分为这三阶段，大多模型省略第一阶段，认为所有文本都默认需要纠正，部分模型会将三阶段联合建模，在逐个构造候选的同时进行评分和筛选，本质上都属于这个框架。

### 2.1 错误识别的主要方法
[10] 利用最大熵分类进行错误识别。

[8] 基于字符级别的词向量。给定待纠错的句子，对每个字符进行判定，看给定上下文时该字符的条件概率是否超过一定阈值，如果没有超过，那么判定有错。

[13] 使用 N-gram LM，对句子里的字符打分，得分低的地方视为待纠错位置。将待纠错位置与上下文组合进行词典查词，当所有组合在词典中都查找不到，则将其视为错字。

### 2.2 生成纠正候选的主要方法
困惑集，是中文文本纠错任务中较为关键的数据之一，用于存储每个字词可能被混淆的错别字词的可能。困惑集的数据格式是 key-value 格式，key 为中文中的常用字词，value 为该字词可能的错误形式。key 可以仅基于字符，也可以包含词语。通常一个 key 对应多个 value。

错误形式，主要分为两大类，分别是发音混淆或者是形状混淆。形状混淆，通常是五笔输入笔画输入手写输入带来的错误。发音混淆最为常见，可分为相同读音、相同音节不同音调、相似音节相同音调、相似音节不同音调。

困惑集的质量很大程度上决定了中文纠错的上限。

- 利用困惑集进行直接替换

  [1] 假设句子中每个字符都存在错误，利用困惑集逐个替换每个字符，生成所有可能的纠正组合。这种方式可以保证召回，但效率和FN不理想。

  [2] 假设句子中每个单字都存在错误，即先分词然后针对单个字符的词，利用困惑集逐个替换，生成所有可能的纠正组合。同样效率不高。
利用困惑集和规则进行有选择替换

  [7] 在分词后利用一系列规则进行困惑集的替换。针对单字词，将困惑集中的所有可能替换均加入候选；针对多字词，若该词不在词表中，尝试对每个汉字进行替换，若替换后词出现在词表，则加入候选；针对多字词，若该词在词表中，不做任何处理。
利用困惑集和词表或语言模型进行有选择替换
这类方法主要有两种思路：一是过滤掉正确的部分，减少替换的次数；一是对于常见的错误构建模板或词表，遇到之后直接替换，避免替换的产生。

  [9] 训练字符级别的 N-gram 模型，选择频数超过阈值的为有效序列。对文本分词后得到单词组成的序列，检查这些序列是否在词表或者 N-gram 中出现过，如没有，则对该序列的单字进行替换。

  [3] 利用未登录词识别，找到无意义的单音节词素，利用困惑集进行替换

  [3] 由于谷歌1T一元数据中包含了很多拼写错误，可以利用其构造修正词典，利用纠错词对直接进行拼写替换。具体步骤为：对1T一元数据中出现频率低的词用困惑集替换，如果新的词频率很高，则作为纠错词对候选；计算每一个纠错词对中两个词在另一个大语料上的频数，如果 原词频数 / 修改词频数 < 0.1，那么将纠错词对写入修正词典。

  [6] 统计语料构造高可信度模板，利用模版过滤一部分正确的序列，只对剩余的部分进行替换工作。主要维护三类数据，模板、长词词表、常用错误词表。

  [5] 对困惑集进行扩充，并对每一个拼写错误构建倒排索引，拼写错误为索引词，潜在正确结果为检索内容，对于每个潜在正确内容，利用汉字出现的频率进行排名。预测同时，在监测阶段维护一个错词修正表，每次替换之后不在词表的词均加入错词表，最终找到正确结果的词加入正确词表，每次结束之后构建错词修正表。如果下次预测到的时候直接利用错词修正表进行调整。

- 利用模型生成

模型生成的纠错候选，基本上可以考虑所有的可能，并且利用其本身的评分函数，可以在生成纠错候选的过程中进行预筛选。

目前效果比较好的方式有 HMM 和 基于图理论 的方法，而利用 SMT 进行生成的效果没有这两种好。

虽然方式比较多，但都可以看做基于贝叶斯的信道噪声模型：

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191215bayesnoise.png)

可见，生成一个纠错候选的决定因素有两个，一个是候选 T 的语言模型，一个是条件概率模型也称为 error model。不同类型方法的主要区别就在于错误模型。如果只考虑替换错误，从而理解为一个对齐之后的字符错误模型。


  [4] 利用 HMM 思想进行纠错候选生成，其中错误模型利用 LD 估计。不过 HMM 模型很大一个问题是其一阶马尔科夫性无法建模长距离依赖。

  [11] 利用图模型进行纠错候选生成。利用困惑集替换目前的每一个字得到拓展后的节点集合。边的权重由替换概率（原始汉字同困惑集汉字的相似程度）和这个词的条件概率（由语言模型计算）得到。并且提供了解决连续两个字以上都是错误而无法解决的问题的方法。

  [7] 利用 SMT 进行纠错工作。由于中文纠错不需要调序，因此这里不需要考虑对齐模型，只需要计算翻译模型和翻译后的语言模型。

### 2.3 评价纠正候选的主要方法
- 利用语言模型进行评价

  [11] 利用句子的困惑度进行评分，更关注句子整体

  [11] 利用互信息进行评分，更关注局部，如 
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191215215513MI.png)
  [9] 利用 SMT 的预测部分进行评分。

  [7] 利用前向算法加字符级别的语言模型进行评分。

  [1, 2, 3, 5, 6] 只是单纯的利用全句的语言模型进行排序。

- 利用分类器进行评价
  [7] 将原任务转化为一个二分类问题，利用 SVM 的置信度评分进行排序。对于每个位置的字符，如果候选和原句不同，则从候选与原句中抽取出相应位置的这些字符组成列表。由 SVM 对每个字符对进行评分，主要特征包括：基本的上下文字符级别特征，PMI特征，词典／语言模型特征。

  [4] 则是以一个整句为单位进行评分。设置了两轮排序，第一轮采用简单易获取的特征搭配 LR，进行初筛；第二轮采用全部的特征搭配 SVM。特征们包括 LM 特征、字典特征、LD 特征、分词特征，业务特征等。

### 2.4 其他
- 自动机

自动机可以实现高效的字符匹配过程。其中，Levenshtein自动机通过构建一个有限状态自动机，可以准确识别出和某个目标单词相距在给定编辑距离内的所有字符串集合。

这可以实现快速的候选生成，作为困惑集的补充。

- 统计信息

在纠错时，除了 Ngram 信息，还有下述统计信息可以作为特征使用：互信息，共现词，拼音的混淆规则、稳定度、相似度，N-gram 距离（Solr 在用方案）。

### 2.5 总结
影响纠错效果的主要因素有如下几点：

- 困惑集：主要影响召回率，纠错首先需要的就是构建一个好的困惑集，使其尽可能小但是包涵绝大多数情况。
- 语言模型：在纠错任务中，常常使用两种语言模型，一种是基于字符级别的，主要用于错误的发现，一般字符级别的阶数在1到5之间。还有一种是词级别的，主要用于排序阶段。
词表：词表主要用于判断替换字符之后是否可以成词，词表最好是比较大的常用词表加上需要应用的领域词表。
- 语料：根据 [12] 提供的方式，确实可以利用大规模的互联网语料估计错误拼写，而且语料也应用于语言模型的生成。
从模型选择上，SMT 更适合用于评分阶段，图模型是一个比较好的分词同纠错一起完成的模型，SVM也是评分阶段的常用手段。

## 3. 实践
下面，以语音控制系统 ASR 产生的的中文文本为例，进行文本纠错，简单描述下主要思路。

### 3.1 收集先验知识
- 词表
- 领域类别词表
- 意图类别词表
- 领域内实体词表
- 语言模型
- 利用领域内和通用语料，生成 N-gram 语料。
- 注意平滑。
- 困惑集
- 收集字符、词级别的困惑集
- 根据词表生成困惑集
- 纠错对照表
- 常用 易错字词-正确字词 对照表
- 收集，并利用 [3] 生成
- 该数据在纠错中具有高优先级
- 热词信息
- 利用日志信息，生成关键词的热度信息
- 训练数据
- 利用领域内和通用语料，随机产生错误（同音字、谐音字、字词乱序、字词增删等），构造训练样本

### 3.2 任务目标
该场景下仅处理如下类型问题：谐音纠错，混淆音纠错，乱序纠错，字词补全。

支持同时处理上述错误类型，当同时处理时，优先顺序为：谐音纠错，混淆音纠错，乱序纠错，字词补全。

引入热词干预、纠错对照表干预

充分利用 session 信息。

在确定领域前，主要处理谐音纠错，混淆音纠错，可用资源有领域类别词表、意图类别词表、基于通用语料的其他先验。

在确定领域后，主要处理字词补全、乱序纠错、谐音纠错，混淆音纠错，可充分利用领域内先验。

### 3.3 算法流程

- 错误识别

  基于字向量使用 Self-attention 针对每个字符的二分类判别器

- 基于字符的双向 N-gram LM

  分词后，针对单字词，认为有错；针对多字词，若该词不在词表中，认为有错

  对于出现在纠错对照表中的认为有错

- 根据 session 信息，高效利用字典信息

生成纠正候选

对于认为有错的字词利用困惑集进行逐一替换，生成纠正候选
基于拼音利用编辑距离自动机生成候选
利用 HMM、图模型、Seq2Seq 生成
根据 session 信息，高效利用字典信息
评价纠正候选
利用多类统计特征，训练判别模型
热词具有较高优先级
如果候选句子中没有分数比原句更高或者与原始评分相比得分不高于阈值的，则认为原句没有错误。否则，得分最高的候选句即作为纠错结果输出。

## References
[0] hqc888688, https://blog.csdn.net/hqc888688/article/details/74858126

[1] Yu He and Guohong Fu. 2013. Description of HLJU Chinese spelling checker for SIGHAN Bakeoff 2013. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing. 84–87.

[2] Chuanjie Lin and Weicheng Chu. 2013. NTOU Chinese spelling check system in SIGHAN Bake-off 2013. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing. 102–107.

[3] Yuming Hsieh, Minghong Bai, and Kehjiann Chen. 2013. Introduction to CKIP Chinese spelling check system for SIGHAN Bakeoff 2013 evaluation. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing. 59–63.

[4] Zhang, S., Xiong, J., Hou, J., Zhang, Q., & Cheng, X. 2015. HANSpeller++: A Unified Framework for Chinese Spelling Correction. ACL-IJCNLP 2015, 38.

[5] Jui-Feng Yeh, Sheng-Feng Li, Mei-Rong Wu, Wen-Yi Chen, and Mao-Chuan Su. 2013. Chinese word spelling correction based on N-gram ranked inverted index list. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing. 43–48.

[6] Tinghao Yang, Yulun Hsieh, Yuhsuan Chen, Michael Tsang, Chengwei Shih, and Wenlian Hsu. 2013. Sinica- IASL Chinese spelling check system at SIGHAN-7. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing. 93–96.

[7] Liu, X., Cheng, F., Duh, K. and Matsumoto, Y., 2015. A Hybrid Ranking Approach to Chinese Spelling Check. ACM Transactions on Asian and Low-Resource Language Information Processing, 14(4), p.16.

[8] Guo, Z., Chen, X., Jin, P. and Jing, S.Y., 2015, December. Chinese Spelling Errors Detection Based on CSLM. In Web Intelligence and Intelligent Agent Technology (WI-IAT), 2015 IEEE/WIC/ACM International Conference on (Vol. 3, pp. 173-176).

[9] Hsunwen Chiu, Jiancheng Wu, and Jason S. Chang. 2013. Chinese spelling checker based on statistical machine translation. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing.49–53.

[10] Dongxu Han and Baobao Chang. 2013. A maximum entropy approach to Chinese spelling check. In Proceedings of the 7th SIGHAN Workshop on Chinese Language Processing. 74–78.

[11] Zhao, H., Cai, D., Xin, Y., Wang, Y. and Jia, Z., 2017. A Hybrid Model for Chinese Spelling Check. ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), 16(3), p.21.

[12] Hsieh, Y.M., Bai, M.H., Huang, S.L. and Chen, K.J., 2015. Correcting Chinese spelling errors with word lattice decoding. ACM Transactions on Asian and Low-Resource Language Information Processing, 14(4), p.18.

[13] Yu J, Li Z. Chinese spelling error detection and correction based on language model, pronunciation, and shape[C]//Proceedings of The Third CIPS-SIGHAN Joint Conference on Chinese Language Processing. 2014: 220-223.

[14] Lin C J, Chu W C. A Study on Chinese Spelling Check Using Confusion Sets and? N-gram Statistics[J]. International Journal of Computational Linguistics & Chinese Language Processing, Volume 20, Number 1, June 2015-Special Issue on Chinese as a Foreign Language, 2015, 20(1).

[15] Chen K Y, Lee H S, Lee C H, et al. A study of language modeling for Chinese spelling check[C]//Proceedings of the Seventh SIGHAN Workshop on Chinese Language Processing. 2013: 79-83.

[16] Zhao J, Liu H, Bao Z, et al. N-gram Model for Chinese Grammatical Error Diagnosis[C]//Proceedings of the 4th Workshop on Natural Language Processing Techniques for Educational Applications (NLPTEA 2017). 2017: 39-44.

[17] Zheng B, Che W, Guo J, et al. Chinese Grammatical Error Diagnosis with Long Short-Term Memory Networks[C]//Proceedings of the 3rd Workshop on Natural Language Processing Techniques for Educational Applications (NLPTEA2016). 2016: 49-56.

[18] Xie P. Alibaba at IJCNLP-2017 Task 1: Embedding Grammatical Features into LSTMs for Chinese Grammatical Error Diagnosis Task[J]. Proceedings of the IJCNLP 2017, Shared Tasks, 2017: 41-46.


[中文(语音结果)的文本纠错综述 Chinese Spelling Check](https://blog.csdn.net/lipengcn/article/details/82556569)

[中文输入纠错任务整理](https://blog.csdn.net/hqc888688/article/details/74858126)
---
layout:     post
title:      CSC
subtitle:   
date:       2020-6-1
author:     RJ
header-img: 
catalog: true
tags:
    - project

---
<p id = "build"></p>
---

<h1>中文语音识别纠错项目总结</h1>


## 研究背景
语音识别文本纠错任务本质上属于文本自动校对的任务之一

常见的文字录入技术和方法主要有键盘录入、语音识别、OCR识别、手写识别。
自动纠错是文本自动校对的一个重要组成部分, 它为自动查错时侦测出的错误字符串提供修改建议, 辅助用户改正错误。
修改建议的有效性是衡量自动纠错性能的主要指标, 它有两点要求:
- 提供的修改建议中应该含有正确或合理的建议;
- 正确或合理的修改建议应尽可能排列在所有建议的前面。

因此, 纠错修改建议的产生算法及排序算法是自动纠错研究的两个核心课题。

---------------------------------
## 历史信息
20世纪60年代，国外研究人员就开展英文文本自动校对的相关研究：

国外学者一般将英文文本错误分为两类：非词错误和真词错误。非词错误，指错误的词不是英文词典里的词，真词错误，指错误的词存在于英文词典中，但不符合上下文语境。中文文本自动校对方法的研究开始于20世纪90年代初借鉴英文文本的研究，国内学者也将中文文本错误分为“真词错误”和“非词错误”。

由于中文汉字的特殊性，电子文本中不会出现字典中没有的字，所以从词的角度区分，将“非词错误”定义为因词中的一个或多个汉字出现替换、插入或删除，使得词串变成不是词典中的词，如“动作迅束”中的“迅束”。而中文“真词错误”则指一个词用错成字典中的另外一个词而形成的错误，如“火山暴发”中的“暴发”。

目前我们项目采用的是基于字的模型，所以没有用到词级别的字典信息。后面可能会遇到，先Mark一下。

--------------------------
## 常见中文错误
- 多字
- 少字
- 别字：为了规避【三】【司】限城市明显过剩的市场风险。
- 混合

多字和少字问题的检错和纠错是个难点，在分析ASRT识别AI_SHELL的test数据集的时候，对于7176条数据，有994条识别后的结果长度不匹配。对于长度不匹配的数据，由于字的位置移动了，而且多字和少字的数目是无法确定的，所以很难定位到错误位置，进而很难验证检错和纠错的准确性。
所以目前的纠错任务普遍面向长度相等的别字类纠错。


## 纠错方案

- kenlm：kenlm统计语言模型工具。
- rnn_attention模型：参考Stanford University的nlc模型，该模型是参加2014英文文本纠错比赛并取得第一名的方法。
- rnn_crf模型：参考阿里巴巴2016参赛中文语法纠错比赛CGED2018并取得第一名的方法。
- seq2seq_attention模型：在seq2seq模型加上attention机制，对于长文本效果更好，模型更容易收敛，但容易过拟合。
- transformer模型：全attention的结构代替了lstm用于解决sequence to sequence问题，语义特征提取效果更好。
- bert模型：中文fine-tuned模型，使用MASK特征纠正错字。
- conv_seq2seq模型：基于Facebook出品的fairseq，北京语言大学团队改进ConvS2S模型用于中文纠错，在NLPCC-2018的中文语法纠错比赛中，是唯一使用单模型并取得第三名的成绩

## iqiyi Faspell

A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm.
- Fast: 模型很快。它显示无论是在绝对时间消耗还是时间复杂度上，速度都要快于以往最先进的模型。
- Adaptable： 模型适应性强。繁简体正确识别，OCR, ASR，CFL外语学习者拼写查错。我们所知,所有先前的最先进的模型只关注繁体中国文字。
- Simple: 仅由Mask model 和 Filter组成，而不是多个模型融合。
- Powerful: 与sota近似的F1，以及在iqiyi ocr数据上的（78.5% in detection and 73.4% in correction)

## 目前纠错的瓶颈

1. 纠错需要的数据量不足，导致模型过拟合。
（A Hybrid Approach to Automatic Corpus Generationfor Chinese Spelling Check） 腾讯出的文章，即通过生成的方式增加数据量。具体是用OCR和ASR模型，得到识别后的错误数据。这只能视作一种增加数据的实用方法。

2. 在利用字符相似度时，缺乏灵活性和混淆性。
inflexibility： 一个字在不同的场景有不同的候选，而候选字多了会影响精度，候选字少了会影响召回。（bert的pretraining，建立Mask概率和错字映射关系，解决这个问题）
insufficiency：字符相似度通过确定一个阈值，产生候选混淆集，候选集中相似的字被统一处理，而字的相似度差异没有充分使用。
Faspell解决方法：
DAE+decoder采用seq2seq范式，与Enc-Dec类似。Enc-Dec中，编码器提取语义信息，解码器生成体现该信息的文本。DAE+decoder模型中，DAE根据上下文特征从被破坏的文本中重建文本，以提供候选文本，decoder通过合并其他特征来选择最佳候选文本。


##  主要的贡献

主要贡献：
提出了一种比Liu等人(2010)和Wang等人提出的更精确的字符相似度量化方法。
提出了一个经验有效的解码器，过滤候选字符，以获得尽可能高的精度而减少影响召回。
- 高精度，通常需要较少的候选且候选中有正确的字； 
- 高召回通常需要更多的候选字以覆盖正确的字；

提高精度而减少影响召回，这就要求候选字符覆盖面要少而精。（模型中，根据top-k保证候选字符的召回，根据filter保证候选字符的精度）


## 两个模块

OCR : 对于字进行笔画拆解，作为度量形近字的标准。
ASR : 对拼音进行编辑距离的计算。


![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/asr_ocr_2020313.png)

注意到：MC、CC、JO、K、V 分别代表中文拼音，粤语，日文，韩文，越南的发音。由于faspell是在拼写数据集SIGHAN13 – 15上进行测试的，另外四种发音在我们的纠错模型中会引入额外噪声，所以选择去掉。

## 自动纠错的两个核心课题

- 候选token的产生算法
- 候选token的排序算法

Faspell模型，在token的产生算法上对Bert的改进体现在以下方面：

对于原生Bert， Mask的都是正确句子中的单词，并且采取8,1,1的策略。即对于一个句子中15%的字，80%的情况下，字符用[MASK]替换，10%保持原字不变，10%采取随机替换。

由于我们是对错误句子进行纠错，所以增加了一个错误句子的掩码策略，对错字进100%MASK，并且其label为正确的字符。这样就建立了错字到正确字的映射关系。而其他字，按照生成的错字概率文件，依照概率进行MASK。

在生成tf_recod数据的时候，对于一个样本中的错字数目，生成error_num+1个错误文本。并且为了保证不要过拟合，也生成了对应error_num+1个正确文本。所以对于一个错误为三个字的样本，会生成2*（3+1）共8个样本，起到了数据增强的作用。而bert自带的dupe_factor会生成10条random MASK的句子，这样就有80个tfrecord产生。


Faspell模型，在token的排序算法上的优点体现在以下方面：
根据错误分布图手动绘制经验有效的过滤曲线
对不同rank的错误采用不同的rank过滤曲线

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/2020313filter.png)

## 语音识别的错误分布散点图



![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/2020313scatters.png)

上图：候选top1与原字相同

flag = 5 * confidence + 2 * similarity - 5 > 0


![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/scatter_unequal_origin.png)

上图：候选top1与原字不同

flag=2.5 * confidence+5 * similarity-2 > 0



可以看到，与iqiyi数据错误分布集中在confidence一侧不同的是，语音识别错误大多集中在similarity一侧。（这里confidence是x轴， similarity是y轴）

即similarity发音错误在这里是主要的，在语音识别纠错场景中这是显然的。

## 纠错结果样例

```python
Error_num : 2    TC: 2
为了规避三司限城市明显过剩的市场风险——ASR 
为了规避三四线城市明显过剩的市场风险——Origin
为了规避三四线城市明显过剩的市场风险——Correction

Error_num : 3    TC: 3
以及提供服务中的变相涨价货价格欺临型为——ASR 
以及提供服务中的变相涨价或价格欺诈行为——Origin
以及提供服务中的变相涨价或价格欺诈行为——Correction

Error_num : 3    TC: 3
促进张略性新兴常夜健康发展——ASR 
促进战略性新兴产业健康发展——Origin
促进战略性新兴产业健康发展——Correction
```

当然，这里只是展示了纠错检错的能力，基于字级别的纠错模型缺点也是显而易见的：

- 连字错误不能很好的解决
- 当然还有所有模型都会遇到的首字出错问题

具体的计算：
```
"original_sentence": "促进张略性新兴常夜健康发展",  
"corrected_sentence": "促进战略性新兴产业健康发展",
"num_errors": 3

{"error_position": 2,"original": "张","corrected_to": "战","candidates": {"战": 0.9999768733978271,"可": 5.78226581637864e-06,"突": 4.176694801572012e-06,"略": 2.3967293145688018e-06,"竞": 1.312395284003287e-06},"confidence": 0.99997687339271,"similarity": 0.6666666666667,"sentence_len": 13}

 {"error_position": 7,"original": "常","corrected_to": "产","candidates": {"产": 0.623263418674469,"市": 0.17305584251880646,"行": 0.1218152791261673,"领": 0.03233199566602707,"业": 0.026191240176558495},"confidence": 0.623263418674,"similarity": 0.6666666666667,"sentence_len": 13}

{ "error_position": 8,"original": "夜","corrected_to": "业","candidates": {"业": 0.9888125658035278,"域": 0.01115991361439228,"区": 1.8983115296578035e-05,"济": 7.419361281790771e-06,"志": 2.869157071927475e-07},"confidence": 0.9888125658035278,"similarity": 1.0,"sentence_len": 13}

```

## 评估

```
Bert without finetune:

performance of round 0:
corretion:
char_p=2688/4960= 0.5419354838709678
char_r=2688/20622=0.13034623217922606
sent_p=513/3485=0.1472022955523673
sent_r=513/5488=0.09347667638483965
sent_a=1201/6182=0.1942736978324167
detection:
char_p=4780/4960=0.9637096774193549
char_r=4780/20622=0.23179129085442732
sent_p=743/3485=0.21319942611190817
sent_r=743/5488=0.13538629737609328
sent_a=1431/6182=0.2314784859268845
In 2092 falsely corrected characters, 1637 are because of absent correct candidates.
In 180 falsely detected characters, 127 are because of absent correct candidates.

```

```
对于候选top1不同的，直接作error进行召回

performance of round 0:
corretion:
char_p=6904/15316= 0.4507704361452076
char_r=6904/20622=0.33478809038890506	
sent_p=996/5225=0.190622009569378	
sent_r=996/5488=0.1814868804664723	
sent_a=1669/6182=0.26997735360724684
detection:
char_p=13828/15316=0.9028466962653434
char_r=13828/20622=0.6705460188148579	
sent_p=1962/5225=0.3755023923444976	
sent_r=1962/5488=0.35750728862973763
sent_a=2635/6182=0.42623746360401166
In 6924 falsely corrected characters, 4988 are because of absent correct candidates.
In 1488 falsely detected characters, 442 are because of absent correct candidates.

```

```
对于候选top1不同的，flag=2.5 * confidence+5 * similarity-2 > 0

performance of round 0:
corretion:
char_p=7078/14467= 0.4892513997373332
char_r=7078/20622=0.34322568131122105
sent_p=1029/5180=0.19864864864864865
sent_r=1029/5488=0.1875
sent_a=1703/6182=0.27547719184729863
detection:
char_p=13124/14467=0.9071680376028202
char_r=13124/20622=0.6364077199107749
sent_p=1861/5180=0.3592664092664093
sent_r=1861/5488=0.33910349854227406
sent_a=2535/6182=0.41006146878033
In 6046 falsely corrected characters, 4363 are because of absent correct candidates.
In 1343 falsely detected characters, 385 are because of absent correct candidates.

```

## 纠错结果对比

Correction F1 score:

Bert origin:   F1 = 0.21014776014385114

Flag True:     F1 = 0.38441630722137193

Finetune flag: F1 = 0.40343127475847135

Detection F1 score:

Bert origin:   F1 = 0.37370025799390194

Flag True:     F1 = 0.7696117990816752

Finetune flag: F1 = 0.7480406965145773

## 主要的改变

Iqiyi模型原本是依据Bert得出的confidence排序候选。我们这里是根据confidence和similarity联合排序，具体的采用加权：

	item['confidence'] + 0.4 * item["similarity"])

多轮纠错与单轮纠错的取舍：

由于iqiyi面向的是SIGHAN13 – 15数据，里面的错误是拼写错误，且错误较少，且人们拼写错误通常是有迹可循的。
语音识别的错误，由于噪声、发音等一系列问题，让语音识别的错误有时候很难纠正， 甚至人类也无法很好的进行纠错。这个时候如果采用多轮纠错，即每次纠错最有把握的那个字，纠错轮数就成了一个硬性影响，即无法很好的确定纠错轮数的大小。

有一个能使用纠错轮数的方法是，在预设的纠错最后一轮，对所有还可能存在的错误进行全部纠错。但是，这点受语音识别系统的限制，如果语音识别足够高，那么前几轮纠错大概率纠错正确的情况下，再进行全纠纠会很好。但是ASR识别差一点的话，最后一轮会引入更多错误。

过滤曲线的选取：

候选top1与原字不同

flag=2.5 * confidence+5 * similarity-2 > 0

候选top1与原字相同

flag = 5 * confidence + 2 * similarity - 5 > 0

## 展望
传统的纠错方法没有很好的对候选集作出动态变化以适应不同场景。预训练模型解决了这方面的问题。

不同场景的候选排序依旧是一个问题。如何在语音识别文本纠错这个方向更好的对候选结果进行排序呢？

这就不仅仅是confidence和similarity的加权问题了。

将confidence和similarity更自然的结合起来是优化候选排序的关键。下一步是研究新的模型，对候选排序进行优化。

此外，引入知识图谱进行纠错也是一个不错的idea，后续也可以继续研究。





















## 论文写作
[论文写作方法与工具](https://yuanxiaosc.github.io/2019/04/08/%E8%AE%BA%E6%96%87%E5%86%99%E4%BD%9C%E6%96%B9%E6%B3%95%E4%B8%8E%E5%B7%A5%E5%85%B7/)

## 数据集

1. NLPCC 2018 GEC官方数据集NLPCC2018-GEC， 训练集trainingdata[解压后114.5MB]，该数据格式是原始文本，未做切词处理。
2. 汉语水平考试（HSK）和lang8原始平行语料HSK+Lang8[190MB]，该数据集已经切词，可用作数据扩增
3. 以上语料，再加上CGED16、CGED17、CGED18的数据，经过以字切分，繁体转简体，打乱数据顺序的预处理后，生成用于纠错的熟语料(nlpcc2018+hsk)，网盘链接:https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA 密码:m6fg [130万对句子，215MB]

## 相关论文


[语音识别后文本纠错处理 2006](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CPFD&dbname=CPFD9908&filename=ZGZR200608001016&v=MjIzNTFGWmVzT0NoTkt1aGRobmo5OFRuanFxeGRFZU1PVUtyaWZaZVp2Rnl2a1U3ZkxJVnNSUHlyUmZMRzRIdGZNcDQ5)

1 ) 建立基于全信息语音识别文本常识知识库。

2 ) 语音识别输 出结果的语法、 语义和语用错误识别。

3 ) 语音识别输出结果的错误 纠正。

4 ) 应用于“ 奥运 多语言智能信息服务系统关键技术及其示范系统研究” 项目的终端部分, 提高终端语音识别结果的正确性。


[基于智能手机平台的语音识别后文本处理的应用 2007](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD2007&filename=2007167220.nh&v=MzE0NjNMdXhZUzdEaDFUM3FUcldNMUZyQ1VSN3FmWnVSdkZ5RG1VcnJOVjEyN0diSytHZFBPcjVFYlBJUjhlWDE=)

[基于自然语言处理的语音识别后文本处理 2008](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD2008&filename=2008139962.nh&v=MTA4NzBqS3JaRWJQSVI4ZVgxTHV4WVM3RGgxVDNxVHJXTTFGckNVUjdxZlp1UnZGeURoVjdySlYxMjdGcks3Rjk=)

[一种基于实例语境的汉语语音识别后文本检错纠错方法 2009](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CPFD&dbname=CPFD0914&filename=ZGZR200907001107&v=MjM4ODJqTXFJOUZaZW9QQ3hOS3VoZGhuajk4VG5qcXF4ZEVlTU9VS3JpZlplWnZGeXZrVTdmTElWc1JQeXJSZkxHNEh0)

[语音识别后文本处理系统中文本语音信息评价算法研究 2010](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD2011&filename=2010264962.nh&v=MjAzMjRSN3FmWnVSdkZ5RG1VcnJOVjEyNkhyRytHdGpLclpFYlBJUjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1U=)

[基于实例语境的语音识别后文本检错与纠错研究 2010](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD2011&filename=2010224540.nh&v=Mjg0NzZyV00xRnJDVVI3cWZadVJ2RnlEbVVyck1WMTI2SHJHNkd0VElyNUViUElSOGVYMUx1eFlTN0RoMVQzcVQ=)

[基于生物实体语境的语音识别后文本纠错算法研究 2012](http://gb.oversea.cnki.net/KCMS/detail/detail.aspx?filename=1012334233.nh&dbcode=CMFD&dbname=CMFDREF)

[基于短语翻译模型的中文语音识别纠错算法 2017](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CPFD&dbname=CPFDLAST2018&filename=SEER201710001087&v=MDgxMzNMRzRIOWJOcjQ5Rlplc0hDeE5LdWhkaG5qOThUbmpxcXhkRWVNT1VLcmlmWmVadkZ5dmtVN2ZNSlY0VE5pak9m)

[中文拼写检错和纠错算法的优化及实现 2019](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201901&filename=1019601236.nh&v=MDcwODc0SDlQUHFaRWJQSVI4ZVgxTHV4WVM3RGgxVDNxVHJXTTFGckNVUjdxZlp1UnZGQ25rVjdyTFZGMjZGN1c=)

[面向领域的语音转换后文本纠错研究 2019](http://new.gb.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1019622321.nh&v=MTY0ODJDVVI3cWZadVJ2RnlEbFc3dk1WRjI2RjdXNkhOTE9ycEViUElSOGVYMUx1eFlTN0RoMVQzcVRyV00xRnI=)

## 相关参考
[语音识别技术简史](https://36kr.com/p/5237890)

[阿里论文](https://developer.aliyun.com/group/paper)

[A Hybrid Approach to Automatic Corpus Generationfor Chinese Spelling Check](https://www.aclweb.org/anthology/D18-1273.pdf)
腾讯的V-style & P-style数据生成。其中V-style是我们需要的：

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20200203093830.png)


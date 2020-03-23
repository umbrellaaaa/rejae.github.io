---
layout:     post
title:      csc paper
subtitle:   
date:       2020-3-7
author:     RJ
header-img: 
catalog: true
tags:
    - paper

---
<p id = "build"></p>
---

## 论文研读

### p1  N-gram model + word dictionary

Gabriel Fung, Maxime Debosschere, Dingmin Wang, Bo Li, Jia Zhu, and Kam-Fai Wong. 2017. Nlptea 2017 shared task – Chinese spelling check. In Proceedings of the 4th Workshop on Natural Language Processing Techniques for Educational Applications (NLPTEA 2017), pages 29–34, Taipei, Taiwan. Asian Federation of Natural Language Processing.


[Chinese Spelling Error Detection and Correction Based on Language
Model, Pronunciation, and Shape](https://www.aclweb.org/anthology/W14-6835.pdf)

Spelling check is an important preprocessing task when dealing with user generated texts such as tweets and product comments. 

Compared with some western languages such as English, Chinese spelling check is more complex because there is no word delimiter in Chinese written texts and misspelled characters can only be determined in word level. Our system works as follows. 
- First, we use character-level n-gram language models to detect potential misspelled characters with low probabilities below some predefined threshold. 
- Second, for each potential incorrect character, we generate a candidate set based on pronunciation and shape similarities.
- Third, we filter some candidate corrections if the candidate cannot form a legal word with its neighbors according to a word dictionary.
- Finally, we find the best candidate with highest language model probability. If the probability is higher than a predefined threshold, then we replace the original character; or we consider the original character as correct and take no action.

Our preliminary experiments shows that our simple method can achieve relatively high precision but low recall.

model shown as :

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20200307091554.png)

论文提供了一种准确较高、召回较低的纠错方法。

- Character级别 n-gram language model。
- 拼音和字形召回候选
- 词典过滤掉部分无效候选
- 取最高语言模型打分
- 高于既定阈值则认为是替换候选

Based on n-gram language model and judging acharacter whether it can form a legal word with
its neighbors, a simple approach is proposed to detect and correct the spelling errors in
traditional Chinese text. 

To find the spelling errors in sentence, the language model and a word dictionary are both used. And in order to reduce the false positive rate, the system only treats the character as a spelling error when the best candidate has been found.

### p2 

Chao-Lin Liu, Min-Hua Lai, Yi-Hsuan Chuang, and Chia-Ying Lee. 2010. Visually and phonologically similar characters in incorrect simplified chinese words. In Proceedings of the 23rd International Conference on Computational Linguistics:
Posters, pages 739–747, Beijing, China. Association for Computational Linguistics.

[Visually and phonologically similar characters in incorrect simplified chinese words](https://www.aclweb.org/anthology/C10-2085.pdf)

Visually and phonologically similar characters are major contributing factors for errors in Chinese text. 
- By defining appropriate similarity measures that consider extended Cangjie codes, we can identify visually similar characters within a fraction of a second.
- Relying on the pronunciation information noted for individual characters in Chinese lexicons, we can compute a list of characters that are phonologically similar to a given character. 

We collected 621 incorrect Chinese words reported on the Internet, and analyzed the causes of these errors. 83% of
these errors were related to phonological similarity, and 48% of them were related to visual similarity between the involved characters. Generating the lists of phonologically and visually similar characters, our programs were able to contain more than 90% of the incorrect characters in the reported errors. 


we consider four categories of phonological similarity between two characters:
- same sound and same tone (SS), 
- same sound and different tone (SD), 
- similar sound and same tone (MS), 
- similar sound and different tone (MD).

人类在中文文本中所使用的错误字符，通常不是在视觉上(第2.2.1节)，就是在语音上(第2.2.2节)与相应的正确字符相似，或者两者兼而有之(Chang, 1995;**Liu et al**.， 2010;Yu和Li, 2014)。OCR产生的错误字符具有视觉相似性也是事实。

### p3  MASS

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and TieYan Liu. 2019. Mass: Masked sequence to sequencepre-training for language generation. arXiv preprint arXiv:1905.02450.

[MASS: Masked Sequence to Sequence Pre-training for Language Generation](http://proceedings.mlr.press/v97/song19d/song19d.pdf)

MASS结构：

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/mass.png)

Unlike BERT or a language model that pre-trains **only the encoder or decoder**, MASS is carefully designed to
pre-train the encoder and decoder jointly in two steps:

1) By predicting the fragment of the sentence that is masked on the encoder side, MASS can force the encoder to understand the meaning of the unmasked tokens, in order to predict the masked tokens in the decoder side;

2) By masking the input tokens of the decoder that are unmasked in the source side, MASS can force the decoder rely more on the source representation other than the previous tokens in the target side for next token prediction, better facilitating the joint training between encoder and decoder.

Encoder的输入中部分MASK, 是为了让MASS学习并理解被MASK的字符的含义

Decoder中将Encoder中的输入进行反转，Mask的字符Unmask, 而未MASK的字符进行MASK, 这样做使得Decoder将注意力放在Encoder中被Mask的字符，这样促进了E-D的联合学习。

这里完全符合我们语音识别纠错的框架， 输入中MASK错误字符，Encoder具有Bert的MASK预测token的特性，而Decoder中，引入错误位置的拼音作为辅助资料。

Faspell中，是直接建立错误字符到正确字符的映射，并没完全利用到音近，因为是独立confidence和similarity的计算。

但是在decoder端，如果我们加入拼音，进行解码，这就太完美了。

MASS 本质是为了提高生成模型的质量，而我们的文本纠错，具备：

含有部分错字的句子；目标的得到正确的句子。

使用seq2seq的encoder框架，encoder端采用Bert MASK机制， Decoder端采用反转MASK的拼音作为输入。

这样建立了 错字--正确字映射（Encoder）, 错字拼音--正确字推断（Decoder）

在Encoder端，的TokensB 能否替换成 拼音序列？  

原本的预训练是  今天天气很好   我也觉得  A<SEP>B   MASK其中的天字， 通过上下文预测 天字；  而A,B判断NSP任务

现在要改成      今天天气很好  jin1 tian1 tian1  qi4 hen3 hao3   MASK其中的第一个天字，  MASK 其中的第一个 tian1

我们通过上下文预测 天字， 通过A中的今[MASK]天气很好, 以及B中的jin1 [tan2] tian1  qi4 hen3 hao3来推断 MASK的字

这样建立了错误拼音到正确汉字的映射。



天 tian1
|   |
电 dian4

how about 天-dian4

Encoder :今[MASK]天气很好  
           
Decoder : [MASK] dian1 [MASK][MASK][MASK][MASK]

这样，建立了上下文context + 拼音信息引入的联合关系，而不再是简单的上下文confidence和拼音编辑距离的加权关系。

可能存在的疑惑： 为什么错误拼音就能映射到正确的拼音对应的汉字？

在文本正确的情况下，我们有了上下文的信息，再根据拼音就能极大概率的推断正确的字，而如果这个拼音是错误的，或者类似的，怎么保证映射到正确的字？

思考 faspell是根据错字映射到候选字，然后结合similarity取最终的字。

而在Decoder端加入拼音信息，就像阅读理解任务中，给了你一个提示一样，让你能联想。

两情诺是久长时，

nuo4 --> ruo4

为了规避三司限城市明显过剩的市场风险 

但是遇到错误大一点的情况, 错误出现在首部会很难预测，究其原因是没有用到拼音去预测：

免报价格会跟风上涨吗

mian3 bao4  --> mian4 bao1  这里我们应该将tone取消掉，因为走进胡同想走出来不容易，根据context判断走哪个胡同还相对简单。

mian bao --!  



免报价格会跟风上涨吗	面包价格会跟风上涨吗	猪糖价格会跟风上涨吗	2

结合拼音信息，mian bao ，那么将会很好预测。

## 多轮对话中的文本纠错
应该用到生成模型MASS，不仅仅是单个句子的纠错。



## 基于掩码语言模型的语音识别纠错

整理文件：

- 预训练样本：AI_shell_dev, thshua_dev
- 测试样本： AI_shell_test, thshua_test
- char_meta文件



## 深入seq2seq模型

![https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/202039seq2seq.jpg]

MASS: 将Bert 改装为 seq2seq 模型

BERT本身是以学习表示为目标，或者说是NLU方面的探索，随机MASK作为预训练任务被证明效果突出，然而会存在pretraining和fine-tuning数据不一致的问题，如果你想把BERT拼在Encoder端和Decoder端，就要考虑改变BERT的预训练方式。

MASS略微修改了BERT的mask方式，在Encoder端连续mask一个子序列，让Decoder用前i-1个词预测第i个，就这样把BERT套进Seq2seq框架，一起pretraining即可。

[Seq2seq框架下的文本生成](https://zhuanlan.zhihu.com/p/71695633)


## 核心问题

如何将拼音向量和字向量发生关系，这样就能利用mask错字后的正确上下文向量与拼音向量发生关系。

- Bert vocab中没有拼音的 id.
- Bert是掩码一个字，预测这个字。如何掩码一个字，预测它的拼音？  免 ；  （面，mian)
- 如何让字能预测拼音，这个问题看上去并不难啊

1. 仔细查看Bert的vocab, 发现有21128行，后面的很多行对于我们项目而言是不可能用到的，所以可以将拼音字表放在最后的vocab里。接下来就是finetune预训练模型的问题了。

2. 如此以来，MASK一个汉字就不仅仅是通过上下文预测这个汉字了，而还要预测这个汉字的拼音。正好NSP是没有用了的，所以我们将tokenB与tokenA表示为一样的结构，通过上下文的字来预测这个拼音，那么我们就能建立上下文字与拼音的联系了。但是这里缺乏了对应字和对应拼音的联系。

在根源上，token A: [免][报]价格会跟风上涨吗  label [面，mian][包,bao]； 我们本质上将Bert改造为多分类问题

在后续finetune上，token B: [mian][bao]价格会跟风上涨吗。 我们用拼音重建，建立错误拼音到正确汉字的映射。从而达到了错字到正确字映射同样的效果，然后再使用Mass模型，这样就能充分利用错字和错误拼音的信息。

接下来就是解决将Bert改造成多分类的问题了。参考[Multi-label Text Classification using BERT](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)

label_ids: one-hot encoded labels for the text

其实就是在vocab范围内，取top2吧。中间有个开关，前15000是汉字，后15000是拼音。

这样解决问题后，就是在MASS中使用我们上面修改过的模型就可以了。不过还要注意的是MASS的解码端，我们还不太熟悉。



### 1.修改vocab字表

```python
#Bert vocab共 21128行

with open('vocab.txt','r',encoding='utf-8') as f:
    data = f.readlines()
    data = data[671:7992]
    print(data[0],data[-1])
    print('常用汉字总数：',len(data))

from pypinyin import lazy_pinyin
style = Style.TONE3
pinyin_set=set()
for item in data:
    item = item.strip()
    pinyin_set.add(lazy_pinyin(item,style=style)[0])

print('对应拼音总数：',len(pinyin_set))

# 对vocab的最后1196行进行替换
with open('vocab.txt','r',encoding='utf-8') as f_in:
    data = f_in.readlines()
    print(len(data))
    with open('reform_vocab.txt','w',encoding='utf-8') as f_out:
        for i in range(21128-1196):
            f_out.write(data[i])
        for i in pinyin_set:
            f_out.write(i+'\n')

from pypinyin import lazy_pinyin,Style
style = Style.TONE3
print(lazy_pinyin('聪明的小兔子',style=style))

```

### finetune bert MASK theory and multilabels

BERT 15% 8,1,1

#### 在大规模语料下，对bert进行finetune，建立汉字和拼音的联系

tokenA 汉字序列MASK汉字 预测 汉字和对应拼音  （建立了汉字和拼音的单点对应关系）

tokenB 汉字序列将汉字替换成拼音 预测拼音对应的汉字 （建立了上下文汉字 联合 拼音 预测对应汉字的关系。）

创建tf_record数据，其中的label要变成一个元组，（面，mian）

在通过上下文预测MASK字的时候，不是简单地拿label就可以了，这里需要预测top2， 理论上讲，top1应该是对应的字，top2为对应的拼音。

损失计算，可以直接相加，相当于每条样本，有两个loss，不做加权应该也可以。

但是这里存在一个问题，我们根据上下文，预测某个被MASK的字，生成的候选是相近的字，如果我们用multilable, 那么岂不是会生成字和拼音一起的候选？ 这个时候，我们要采取过滤，取前15000vocab中的top5汉字，取后15000中的top3拼音。

这里我们要深入Bert的MASK 和 NSP loss的计算, 以便于修改：

```python
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True) # input_tensor向量 与 output_weights词表权重映射矩阵
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)
```


注意到：
```python
  def get_pooled_output(self): # 获取句子的output
    return self.pooled_output

  def get_sequence_output(self):

  #获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output
```
再仔细分析一下MLM模型，上下文是有真实向量的，12层encoder就是根据上下文向量推断MASK位置的向量的信息，最后拿到这个信息后，接一个dense_layer，相当于对生成的特征进行一个组合，然后映射。


```
关于Bert掩码语言模型的源码问题,具体在get_masked_lm_output方法中，不太明白拿到最后一层对应MASK位置的向量后，还接了一个随机初始化的dense_layer，如下：

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)
```

Bert打印信息：
```
INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/query/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/query/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/key/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/key/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/value/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/self/value/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/dense/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/dense/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/LayerNorm/beta:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/attention/output/LayerNorm/gamma:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/intermediate/dense/kernel:0, shape = (768, 3072)
INFO:tensorflow:  name = bert/encoder/layer_10/intermediate/dense/bias:0, shape = (3072,)
INFO:tensorflow:  name = bert/encoder/layer_10/output/dense/kernel:0, shape = (3072, 768)
INFO:tensorflow:  name = bert/encoder/layer_10/output/dense/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/output/LayerNorm/beta:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_10/output/LayerNorm/gamma:0, shape = (768,)

INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/query/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/query/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/key/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/key/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/value/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/self/value/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/dense/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/dense/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/LayerNorm/beta:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/attention/output/LayerNorm/gamma:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/intermediate/dense/kernel:0, shape = (768, 3072)
INFO:tensorflow:  name = bert/encoder/layer_11/intermediate/dense/bias:0, shape = (3072,)
INFO:tensorflow:  name = bert/encoder/layer_11/output/dense/kernel:0, shape = (3072, 768)
INFO:tensorflow:  name = bert/encoder/layer_11/output/dense/bias:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/output/LayerNorm/beta:0, shape = (768,)
INFO:tensorflow:  name = bert/encoder/layer_11/output/LayerNorm/gamma:0, shape = (768,)

INFO:tensorflow:  name = bert/pooler/dense/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = bert/pooler/dense/bias:0, shape = (768,)

# 对应权重矩阵信息，对MASK字的特征信息进行组合
INFO:tensorflow:  name = cls/predictions/transform/dense/kernel:0, shape = (768, 768)
INFO:tensorflow:  name = cls/predictions/transform/dense/bias:0, shape = (768,)

INFO:tensorflow:  name = cls/predictions/transform/LayerNorm/beta:0, shape = (768,)
INFO:tensorflow:  name = cls/predictions/transform/LayerNorm/gamma:0, shape = (768,)
INFO:tensorflow:  name = cls/predictions/output_bias:0, shape = (30522,)

INFO:tensorflow:  name = cls/seq_relationship/output_weights:0, shape = (2, 768)
INFO:tensorflow:  name = cls/seq_relationship/output_bias:0, shape = (2,)


```
注意控制： label_ids = tf.reshape(label_ids, [-1]) ## [batch_size, 126]


create_model 返回 get_masked_lm_output的结果：

loss, per_example_loss, ||||   log_probs, probs

masked_lm_loss, masked_lm_example_loss, ||||   self.masked_lm_log_probs, self.probs
```python
        # create model
        masked_lm_loss, masked_lm_example_loss, self.masked_lm_log_probs, self.probs = self.create_model(
            self.input_ids,
            self.input_mask,
            self.segment_ids,
            self.masked_lm_positions,
            self.masked_lm_ids,
            self.masked_lm_weights,
            is_training,
            config.bert_config)

        # prediction
        self.masked_lm_predictions = tf.argmax(self.masked_lm_log_probs, axis=-1, output_type=tf.int32)
        self.top_n_predictions = tf.nn.top_k(self.probs, k=config.topn, sorted=True, name="topn")
```

注意到返回的结果和prediction的内容。



#### 在后续任务中，同faspell原理，对Bert在纠错数据集上进行finetune.

错误汉字 映射到正确汉字；  错误拼音结合上下文映射正确汉字。

修改Bert Mask策略使字和拼音产生联系

今天构思了如何将拼音与字建立联系的方法，以及通过上下文字向量 + 待预测字的无tone拼音 预测MASK汉字及有tone拼音的方法。

该方法具有可行性，目前的难点是模型的Loss计算的修改，以及张量的详细变换问题。这要详细分析模型，需要花几天的时间。

待构建好模型结构后，先进行正确文本的finetune，使汉字与拼音建立联系，具备通过拼音推断汉字的能力，以及通过上下文字向量+拼音向量预测对应正确汉字和正确拼音的能力。

下一步，参考faspell中修改MASK机制的方法： 错字到正确字映射

使用 MASS模型，  在Encoder端掩码错字，在Decoder端，将无tone的错误拼音输入，预测正确的字和拼音。这一步骤为我们模型具体的finetune

## 继续进行

Bert run_pretraining 357row : name_to_features

修改dict:将dict后面的替换成拼音

构建tfrecord数据,增加py_label。 填充py_masked_id

修改loss计算

- 第一阶段，建立联系： 字向量与py

给定正确句子，掩码后预测，会包含字和拼音。top1就是其正确拼音

- 第二阶段，破碎重组： 错字，错拼 与 正字，正拼

给定错误句子，掩码后，错字映射到正确字建立了联系； 错误拼音到正确拼音建立了联系。


目标：

encoder: 促进[张]略性新兴[常][夜]健康发展

decoder：————[zhang]---[chang][ye]-----

张的上下文：促进 X 略性新兴[][]健康发展 + 拼音信息（zhang）

而拼音破碎重组，zhang会和zhan产生联系。而汉字与拼音又有联系

所以： 促进[MASK]略性新兴[][]健康发展 + 拼音信息（chang ye）===》》 产业

对于难一点的，之前的字模型很难解决

免报价格会涨价吗？  

[MASK][MASK]价格会涨价吗？ + [mian3][bao4] ==>> 面包

考虑到音调的问题。。。可能会比较麻烦，当时处理的时候，如果不带声调，只有400多个id， 加了音调有1300个id。

之前其实想将tokenB改成拼音预测汉字。其实这个有必要做。单向的字到拼音的预测是不够的，我们还需要拼音到字的联系。

玩转二者后，通过掩码一个位置可以得到汉字，也可以得到这个位置的拼音。

然后使用MASS的原理，掩码一个片段，在解码器，使用片段拼音预测掩码的汉字。

这里，我们需要BERT完成77%的检错，相当于我们将检错和纠错分开。纠错部分用MASS.

还需要训练MASS这个模型。。。

思考一下，之前的span-bert的做法，掩码一个片段，预测这个片段和我们的纠错不是很像吗？

它是随机掩码片段，我可以看看他怎么解码的，如果有操作的空间，那么就动他手了。
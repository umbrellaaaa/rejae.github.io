---
layout:     post
title:      Bert Encoding
subtitle:   summary
date:       2019-11-2
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## BERT使用详解(实战)
BERT模型，本质可以把其看做是新的word2Vec。对于现有的任务，只需把BERT的输出看做是word2vec，在其之上建立自己的模型即可了。

### 1.下载BERT<br>
BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340M parameters
BERT-Base, Cased: 12-layer, 768-hidden, 12-heads , 110M parameters
BERT-Large, Cased: 24-layer, 1024-hidden, 16-heads, 340M parameters
BERT-Base, Multilingual Cased (New, recommended): 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
BERT-Base, Multilingual Uncased (Orig, not recommended) (Not recommended, use Multilingual Cased instead): 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
BERT-Base, Chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
前4个是英文模型，Multilingual 是多语言模型，最后一个是中文模型(只有字级别的) 其中 Uncased 是字母全部转换成小写，而Cased是保留了大小写。

BERT源码 可以在Tensorflow的GitHub上获取。

###本文的demo地址，需要下载BERT-Base, Chinese模型，放在根目录下

### 2.加载BERT<br>
官方的源码中已经有如何使用BERT的demo。demo中使用了TPUEstimator 封装，感觉不好debug。其实BERT的加载很简单。

直接看代码
```python
import tensorflow as tf
from bert import modeling
import os

# 这里是下载下来的bert配置文件
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
#  创建bert的输入
input_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="segment_ids")

# 创建bert模型
model = modeling.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
)

#bert模型参数初始化的地方
init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
use_tpu = False
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型

(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
```

复制代码,上面是按照源码，做了提取。<br>
下面的代码也可以加载模型
```python
import tensorflow as tf
from bert import modeling
import os

pathname = "chinese_L-12_H-768_A-12/bert_model.ckpt" # 模型地址
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")# 配置文件地址。
configsession = tf.ConfigProto()
configsession.gpu_options.allow_growth = True
sess = tf.Session(config=configsession)
input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="segment_ids")

with sess.as_default():
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())# 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
    saver.restore(sess, pathname)
    print(1)
```

复制代码
这里就很清晰了，就是常用的TensorFlow模型加载方法。

### 3.使用模型<br>
获取bert模型的输出非常简单，使用 model.get_sequence_output()和model.get_pooled_output() 两个方法。

output_layer = model.get_sequence_output()# 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个<br>

output_layer = model.get_pooled_output() # 这个获取句子的output
复制代码
那么bert的输入又是什么样子的呢？ 看下面代码

```python
def convert_single_example( max_seq_length,
                           tokenizer,text_a,text_b=None):
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)# 这里主要是将中文分字
  if tokens_b:
    # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
    # 因为要为句子补上[CLS], [SEP], [SEP]
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
  # (a) 两个句子:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) 单个句子:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 这里 "type_ids" 主要用于区分第一个第二个句子。
  # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
  # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
  # 创建mask
  input_mask = [1] * len(input_ids)
  # 对于输入进行补0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数
```

复制代码
上面的代码是对单个样本进行转换，代码中的注释解释的很详细了，下面对参数说明下: max_seq_length ：是每个样本的最大长度，也就是最大单词数。 tokenizer ：是bert源码中提供的模块，其实主要作用就是将句子拆分成字，并且将字映射成id text_a ： 句子a text_b ： 句子b

### 4.值得注意的地方<br>
1，bert模型对输入的句子有一个最大长度，对于中文模型，我看到的是512个字。
2，当我们用model.get_sequence_output()获取每个单词的词向量的时候注意，头尾是[CLS]和[SEP]的向量。做NER或seq2seq的时候需要注意。
3，bert模型对内存的要求还是很高的，运行本文的demo的时候，如果内存不足，可以降低batch_size和max_seq_length来试下。
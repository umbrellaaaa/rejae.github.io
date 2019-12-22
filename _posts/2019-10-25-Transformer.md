---
layout:     post
title:      Transformer
subtitle:   学习、理解、应用
date:       2019-10-25
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 前言
看过很多相关Transformer的文章了，知识点零零碎碎，面试的东西只能回答一部分，实验只调试了别人的代码，诸多原理都还没深刻理解。今天就好好整理一下资料，调试相关代码，以便以后复习。

## 正文
谈到Transformer这个特征抽取器就需要与CNN和RNN这两个抽取器作一个对比分析：<br>
CNN像滑动窗口移动卷积核抽取特征，经过池化选择特征，局部抽取能力很强，抽取依赖关系能力较弱。<br>
LSTM建模输入序列的依赖关系，保存和更新记忆单元，过长的句子性能会下降。<br>
Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。性能提升的关键是将任意两个单词的距离是1，解决NLP中棘手的长期依赖问题。


## 回顾序列编码问题：
RNN:        y<sub>t</sub> = f (y<sub>t−1</sub>, x<sub>t</sub>)

CNN:        y<sub>t</sub> = f (x<sub>t−1</sub>, x<sub>t</sub>, x<sub>t+1</sub>)

Attention:  y<sub>t</sub> = f (x<sub>t</sub>, A, B)            

其中A,B是另外一个序列（矩阵）。如果都取A=B=X，那么就称为Self Attention，它的意思是直接将xt与原来的每个词进行比较，最后算出yt！



## transformer结构

![transformer_block](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191118transformer_block.jpg)




## self-attention

self-Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：
- 第一个过程是根据Query和Key计算权重系数
- 第二个过程根据权重系数对Value进行加权求和

而第一个过程又可以细分为两个阶段：第一个阶段根据Query和Key计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如图10展示的三个阶段。

![](https://pic2.zhimg.com/80/v2-07c4c02a9bdecb23d9664992f142eaa5_hd.jpg)

更详细的：

**The first step** in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

![](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

What are the “query”, “key”, and “value” vectors?

They’re abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

**The second step** in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

![](https://jalammar.github.io/images/t/transformer_self_attention_score.png)


**The third and forth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.
![](https://jalammar.github.io/images/t/self-attention_softmax.png)

This softmax score determines how much how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.



**The fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

**The sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

![](https://jalammar.github.io/images/t/self-attention-output.png)

**Finally**, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.
![](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)




## Multi-head attention
The Beast With Many Heads
The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. This improves the performance of the attention layer in two ways:

It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, we would want to know which word “it” refers to.

它扩展了模型关注不同位置的能力。

It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

这个是Google提出的新概念，是Attention机制的完善。不过从形式上看，它其实就再简单不过了，就是把Q,K,V通过参数矩阵映射一下，然后再做Attention，把这个过程重复做h次，结果拼接起来就行了，可谓“大道至简”了。具体来说

head<sub>i</sub>=Attention(QQ<sub>i</sub><sup>W</sup>,KK<sub>i</sub><sup>W</sup>,VV<sub>i</sub><sup>W</sup>)

W<sub>i</sub><sup>Q</sup>∈R<sup>k×^k</sup>,  K<sub>i</sub><sup>Q</sup>∈R<sup>k×^k</sup>, V<sub>i</sub><sup>Q</sup>∈R<sup>v×^v</sup>

MultiHead(Q,K,V)=Concat(head<sub>i</sub>,...,head<sub>h</sub>)

最后得到一个n×(hd~v)的序列。所谓“多头”（Multi-Head），就是只多做几次同样的事情（参数不共享），然后把结果拼接。
![](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

Attention层的好处是能够一步到位捕捉到全局的联系，因为它直接把序列两两比较（代价是计算量变为O(n2)，当然由于是纯矩阵运算，这个计算量相当也不是很严重）；相比之下，RNN需要一步步递推才能捕捉到，而CNN则需要通过层叠来扩大感受野，这是Attention层的明显优势。

## 参考
[目前主流的attention方法都有哪些？--张俊林](https://www.zhihu.com/question/68482809/answer/264632289)

[苏剑林. (2018, Jan 06). 《《Attention is All You Need》浅读(简介+代码)](https://kexue.fm/archives/4765)

[illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

[huggingface.co/transformers](https://huggingface.co/transformers/)


[mc.ai](https://mc.ai/transformer-architecture-attention-is-all-you-need-2/)





## 核心代码

```python
        with tf.name_scope("positionEmbedding"):
            embedded_position = self._position_embedding()

        embedded_representation = embedded_words + embedded_position

        with tf.name_scope("transformer"):
            for i in range(self.config["num_blocks"]):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    with tf.name_scope("multi_head_atten"):
                        # 维度[batch_size, sequence_length, embedding_size]
                        multihead_atten = self._multihead_attention(inputs=self.inputs,
                                                                    queries=embedded_representation,
                                                                    keys=embedded_representation)
                    with tf.name_scope("feed_forward"):
                        # 维度[batch_size, sequence_length, embedding_size]
                        embedded_representation = self._feed_forward(multihead_atten,
                                                                     [self.config["filters"],
                                                                      self.config["embedding_size"]])

            outputs = tf.reshape(embedded_representation,
                                 [-1, self.config["sequence_length"] * self.config["embedding_size"]])

        output_size = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(outputs, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # 计算交叉熵损失
        self.loss = self.cal_loss() + self.config["l2_reg_lambda"] * self.l2_loss
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

```

## _position_embedding()

pytorch版本
``` 
#keep dim 0 for padding token position encoding zero vector
position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
return torch.from_numpy(position_enc).type(torch.FloatTensor)
```

```python
    def _position_embedding(self):
        """
        生成位置向量
        :return:
        """
        batch_size = self.config["batch_size"]
        sequence_length = self.config["sequence_length"]
        embedding_size = self.config["embedding_size"]

        # 生成位置的索引，并扩张到batch中所有的样本上
        position_index = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batch_size, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        position_embedding = np.array(
            [
                    [pos / np.power(10000, (i - i % 2) / embedding_size)
                        for i in range(embedding_size)]
                    
                for pos in range(sequence_length)
            ]
        )


        # 然后根据奇偶性分别用sin和cos函数来包装
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        embedded_position = tf.nn.embedding_lookup(position_embedding, position_index)

        return embedded_position
```

## _layer_normalization()

```python
    def _layer_normalization(self, inputs):
        """
        对最后维度的结果做归一化，也就是说对每个样本每个时间步输出的向量做归一化
        :param inputs:
        :return:
        """
        epsilon = self.config["ln_epsilon"]

        inputs_shape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]
        params_shape = inputs_shape[-1]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)

        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs
```


## _multihead_attention()

```python
    def _multihead_attention(self, inputs, queries, keys, num_units=None):

        num_heads = self.config["num_heads"]  # multi head 的头数

        if num_units is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)


        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)


        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))/(K_.get_shape().as_list()[-1] ** 0.5)

        mask = tf.tile(inputs, [num_heads, 1])
        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)

        key_masks = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(queries)[1], 1])   
        masked_similarity = tf.where(tf.equal(key_masks, 0), paddings,similarity) 
        weights = tf.nn.softmax(masked_similarity)

        query_masks = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(keys)[1]])
        masked_weights = tf.where(tf.equal(query_masks, 0), paddings, weights) 

        outputs = tf.matmul(masked_weights, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        outputs += queries
        outputs = self._layer_normalization(outputs)
        return outputs

```

## _feed_forward()

```python
    def _feed_forward(self, inputs, filters):
        """
        用卷积网络来做全连接层
        :param inputs: 接收多头注意力计算的结果作为输入
        :param filters: 卷积核的数量
        :return:
        """

        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layer_normalization(outputs)

        return outputs
```

## pytorch Transformer

```python
## AI算法基本流程：

1. 数据准备 & 预处理  ==>>  trian  val  test  文件
2. 配置文件 hparams.py
3. 模型组件 modules.py 相应核心方法
4. 模型构建 models.py Transformer/CNN/RNN
5. 工具方法 utils.py
6. 模型训练 train.py 

------------------------------------------------------------------------

### 配置文件 hparams.py
```python
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train1', default='iwslt2016/segmented/train.de.bpe',
                             help="german training segmented data")
    parser.add_argument('--train2', default='iwslt2016/segmented/train.en.bpe',
                             help="english training segmented data")
    parser.add_argument('--eval1', default='iwslt2016/segmented/eval.de.bpe',
                             help="german evaluation segmented data")
    parser.add_argument('--eval2', default='iwslt2016/segmented/eval.en.bpe',
                             help="english evaluation segmented data")
    parser.add_argument('--eval3', default='iwslt2016/prepro/eval.en',
                             help="english evaluation unsegmented data")

    ## vocabulary
    parser.add_argument('--vocab', default='iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")

--------------------------------------------------------------
### 模型组件 modules.py 相应核心方法




---------------------------------------------------------------
### 模型构建 models.py Transformer
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, sents1, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries


```

utils.py中相应的保存超参数的方法：
```python
def save_hparams(hparams, path):
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)
```





```


## 常见问题

请介绍一下自注意力机制？

为什么需要多头注意力？

FF Network在这里起什么作用？

残差连接原理？

Encoder-Decoder分别做了什么？

请描述一个使用transformer的整体流程？

## 整体框架

深入transformer机器翻译项目的流程框架：

1. download data
2. prepro.py 预处理

```
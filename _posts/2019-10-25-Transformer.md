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
Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。性能提升的关键是将任意两个单词的距离是1，解决NLP中棘手的长期依赖问题
## 从以下四个方面对比CNN,RNN,Transformer：

语义特征提取能力；
长距离特征捕获能力；
任务综合特征抽取能力；
并行计算能力及运行效率


## transformer结构

![transformer_block](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191118transformer_block.jpg)


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
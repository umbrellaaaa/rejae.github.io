---
layout:     post
title:      Tidy Tensorflow
subtitle:   
date:       2019-11-26
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---
<p id = "build"></p>
---

## For what? 
记录Tensorflow的用法, 从源代码学习编程方法。

## tf.reduce_sum 维度缩减

```python

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()
a = [
    [1, 2, 3],
    [4, 5, 6]
]
b = tf.reduce_sum(a, axis=-1)
print(b)
得到:
[6,15]
```

## tf.tile 维度扩展

```python
a: [[1,2],[3,4],[5,6]]  #即3*2维度

tf.tile(a,[2,2])得到：

[   
    [1,2,1,2],[3,4,3,4],[5,6,5,6]

    [1,2,1,2],[3,4,3,4],[5,6,5,6]

]

#即tf.tile按照第二个参数，对原输入作相应维度的扩张
#应用场景： transformer 在对q,k,v进行split后，输出input的batch维度扩展成原来的num_heads倍，所以对相应mask需要扩倍
#利用tf.tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度100
mask = tf.tile(inputs, [num_heads, 1])
#将mask扩展成Q_的大小，需要在axis=1即time_step那个维度扩展，且那个维度也取与q,k,v相同的维度
keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

```

## tf.where
```python
#维度[batch_size * numHeads, queries_len, key_len]
maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,scaledSimilary)  
```

## 模型保存相关

初始化saver
```python

xxxxxx weight histgram summary  xxxxxxxxx

    with tf.variable_scope("encode_fc-layer%d" % (l + 1)):
        input_data = tf.layers.dense(inputs=input_data, units=args.layers[l], activation=tf.nn.tanh
                                        , kernel_initializer=xavier_initializer()
                                        , kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight),
                                        name="layer%d" % (l + 1), reuse=tf.AUTO_REUSE)
        if self.flag:
            Weights = tf.get_default_graph().get_tensor_by_name(
                os.path.split(input_data.name)[0] + '/kernel:0')
            Biases = tf.get_default_graph().get_tensor_by_name(
                os.path.split(input_data.name)[0] + '/bias:0')
            #  写进日志
            tf.summary.histogram(os.path.split(input_data.name)[0] + "/weights",
                                    Weights)  # name命名，Weights赋值
            tf.summary.histogram(os.path.split(input_data.name)[0] + "/bias", Biases)
xxxxxx weight histgram summary  xxxxxxxxx

xxxxxx loss scalar summary  xxxxxxxxx

      tf.summary.scalar('loss_with_jacobian', self.loss_with_jacobian)
        self.train_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.train_optimizer_with_jacobian = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_with_jacobian)
xxxxxx loss scalar summary  xxxxxxxxx

------------------------------------------------------------------------

init = tf.global_variables_initializer()

self.merged = tf.summary.merge_all()
self.saver = tf.train.Saver(max_to_keep=50)
self.save_path = ""

self.sess.run(init)
self.summary_writer = tf.summary.FileWriter("logs/", graph=self.sess.graph)

------------------------------------------------------------------------

for it in range(args.num_jacobian_iters):
    start = time.time()

    feed_dict = {}
    feed_dict[self.x] = np.reshape(x_data, (args.trj_nums, (args.seq_length + 1), args.state_dim))
    if args.control_input: feed_dict[self.u] = u_data[:, :args.seq_length]

    # Find loss and execude training operation
    fetches = [self._jaco_bian_loss, self.loss_with_jacobian, self.loss, self.loss_reconstruction,
                self.train_optimizer_with_jacobian]
    try:
        _jaco_bian_loss, total_loss_with_jacobian, loss, rec_loss, _ = self.sess.run(fetches=fetches,
                                                                                        feed_dict=feed_dict)
        summary_str = self.sess.run(fetches=self.merged, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, it)


```


获取checkpoint的模型
```python
def get_checkpoint_id_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()[1:]
        d_list = [item.split(':')[1].strip().replace('\"', '') for item in sentences]
        print(id_list)
        return id_list

```

循环调用保存的模型查看参数和图像关联，分析原因：
```python
for item in id_list:
    sess = deep_koopman.sess
    deep_koopman.saver.restore(sess, 'models/'+item)
```

## Transformer代码

注意Q,K,V都是输入input_x， 三者经过dense返回经过权重矩阵x*W 的结果，Q_, K_, V_, 我们使用这Q_,K_矩阵计算相似度, 再与V_计算。也即是输入经过同维线性变换后，进行相似度计算。
```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

num_heads = 8
a = [
    [[0, 0], [2, 2], [3, 3]],
    [[4, 4], [5, 5], [6, 6]]
]
b = tf.reduce_sum(a, axis=-1)
print(b)
c = tf.abs(tf.reduce_sum(a, axis=-1))
print(c)
d = tf.sign(tf.abs(tf.reduce_sum(a, axis=-1)))
print(d)


key_masks = tf.tile(d, [num_heads, 1]) # (h*N, T_k)

key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(a)[1], 1]) # (h*N, T_q, T_k)

print(key_masks)

# outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
# paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
```
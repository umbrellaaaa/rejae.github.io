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
# 利用tf.tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度100
mask = tf.tile(inputs, [num_heads, 1])
#将mask扩展成Q_的大小，需要在axis=1即time_step那个维度扩展，且那个维度也取与q,k,v相同的维度
keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

```

## tf.where
```python
# 维度[batch_size * numHeads, queries_len, key_len]
maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,scaledSimilary)  
```
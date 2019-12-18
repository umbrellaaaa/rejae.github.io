---
layout:     post
title:      work day1
subtitle:   
date:       2019-12-31
author:     RJ
header-img: 
catalog: true
tags:
    - Learning
---
<p id = "build"></p>
---

## 月度学习内容

## Transformer





## 价值问题
1.  why we add(e,pe) but not concat(e,pe)?
[tensor2tensor/issues/1591](https://github.com/tensorflow/tensor2tensor/issues/1591)
[参考](https://www.zhihu.com/question/350116316/answer/860242432)
```
Apart from saving some memory, is there any reason we are adding the positional embeddings instead of concatenating them. It seems more intuitive concatenate useful input features, instead of adding them.

From another perspective, how can we be sure that the Transformer network can separate the densely informative word embeddings and the position information of pos encoding?


连环三问：为什么学到位置信息？这不会影响原始words embedding么？怎么学到位置信息？

首先输入为原始words的embedding + PE，就是说输入考虑了位置。这里说考虑位置最核心的就是不同句子的相同位置PE值是一样的（不论那句话，第一个词的PE值肯定是一样的，所以见到这个PE我就知道它是第一个）然后计算loss更新embedding时候，是根据误差来的，这个误差又考虑了位置（因为输入加了PE，计算结果就有这个信息）。

所以更新得到的embedding时候会考虑这个位置PE的值。（从loss角度理解，先从误差来看，因为加了PE，相对于不加，这可能导致误差变大或者变小，但是目标是最小化loss，则优化方向就是加了PE之后，我如何更新words的embedding才能使loss最小。

那就是说，训练过程中得到的embedding是加了位置后还得使loss最小化得到。说明embedding考虑了这个位置信息。。。

感觉说来说去还是一件事。。总结一下因为模型更新embedding是最小化loss，而loss的输入又是考虑了每个词PE位置信息，所以最终通过SGD最小化loss更新embedding参数时是考虑了每个词的位置的。。。

至于为什么用sin cos固定函数这就是另一个话题了。个人觉得不是很重要。所以这么弄就需要很好的调参数，因为PE作为输入计算误差，如果PE和Embedding量级差别大，则可能导致loss中PE和embedding信息谁主导loss，太大则emebdding训练不好；太小，位置信息则又太弱。需要打印PE和emebdding值，仔细分析，小心调参。
```

2. 思考embedding+position_emb之后就使用dropout的意义在哪里？Why should we use (or not) dropout on the input layer?

```
why we do:
People generally avoid using dropout at the input layer itself. But wouldn't it be better to use it?

Adding dropout (given that it's randomized it will probably end up acting like another regularizer) should make the model more robust. It will make it more independent of a given set of features, which matter always, and let the NN find other patterns too, and then the model generalizes better even though we might be missing some important features, but that's randomly decided per epoch.

Is this an incorrect interpretation? What am I missing?

Isn't this equivalent to what we generally do by removing features one by one and then rebuilding the non-NN-based model to see the importance of it?

why not:

Why not, because the risks outweigh the benefits.

It might work in images, where loss of pixels / voxels could be somewhat "reconstructed" by other layers, also pixel/voxel loss is somewhat common in image processing. But if you use it on other problems like NLP or tabular data, dropping columns of data randomly won't improve performance and you will risk losing important information randomly. It's like running a lottery to throw away data and hope other layers can reconstruct the data.

In the case of NLP you might be throwing away important key words or in the case of tabular data, you might be throwing away data that cannot be replicated anyway else, like gens in a genome, numeric or factors in a table, etc.

I guess this could work if you are using an input-dropout-hidden layer model as you described as part of a larger ensemble though, so that the model focuses on other, less evident features of the data. However, in theory, this is already achieved by dropout after hidden layers.

```

3. key mask和 query mask的意义？
```
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1))) # (N, T_k)   
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

针对emb,将最后embedding_size一维reduce_sum,emb的shape就转为[batch,seq], 将batch维度tile成multi_head的倍数，这样相当于[batch,(w1,w2,...,wn)]其中由于sign将w1,w2,...替换成了1，0，-1，当wi是[PAD]时候，wi被padding。key mask就是为了不受 补全短句的positional encoding的影响。 query mask只需要变换一下维度直接与keymask对应相乘就好了。
```
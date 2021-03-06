---
layout:     post
title:      November work
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

## 1.Transformer

[理解transformer模型](https://www.tensorflow.org/tutorials/text/transformer#optimizer)





### 价值问题
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



4. FFNN的作用？






## 2. Seven Applications of Deep Learning for Natural Language Processing
[Jason Brownlee 7 NLP application](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/)

1. Text Classification
2. Language Modeling
3. Speech Recognition
4. Caption Generation
5. Machine Translation
6. Document Summarization
7. Question Answering

## 3. 回顾NLP历史 温故而知新
[A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/pdf/1510.00726.pdf)




## 环境搭建

1. putty
2. winscp
3. 环境配置

conda install -c <channel> <software>  example:  conda install -c <channel> <software>

conda install ipykernel

conda remove -n py36 --all

source activate 环境名称

python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"

jupyter kernelspec list

jupyter kernelspec remove z1  


start_jupyter.sh

nohup jupyter notebook --ip=192.168.100.76 --allow-root &

http://192.168.100.xxx:8889/?token=xxx


[服务器外部jupyter访问](https://blog.csdn.net/mmc2015/article/details/52439212)




## Linux

1. jupyter一直在后台运行，而且开启了多个，怎么解决？

ps -aux | grep jupyter 查看运行的jupyter进程

jupyter notebook list  查看所有连接，本地打开连接，点击quit，结束jupyter任务

cat /proc/driver/nvidia/version

NVRM version: NVIDIA UNIX x86_64 Kernel Module  384.69  Wed Aug 16 19:34:54 PDT 2017

GCC version:  gcc 版本 4.8.5 20150623 (Red Hat 4.8.5-16) (GCC)

nvcc --version   Cuda compilation tools, release 8.0, V8.0.61

[cudatoolkit 太高](https://zhuanlan.zhihu.com/p/64376059)

[tf_version cuda cudnn](https://blog.csdn.net/qq_27825451/article/details/89082978)


linux复制指定目录下的全部文件到另一个目录中，linux cp 文件夹
linux复制指定目录下的全部文件到另一个目录中
复制指定目录下的全部文件到另一个目录中
文件及目录的复制是经常要用到的。linux下进行复制的命令为cp。
假设复制源目录 为 dir1 ,目标目录为dir2。怎样才能将dir1下所有文件复制到dir2下了
如果dir2目录不存在，则可以直接使用
cp -r dir1 dir2
即可。
如果dir2目录已存在，则需要使用
cp -r dir1/. dir2
如果这时使用cp -r dir1 dir2,则也会将dir1目录复制到dir2中，明显不符合要求。
ps:dir1、dir2改成对应的目录路径即可。

cp -r /home/www/xxx/statics/. /home/www/statics
如果存在文件需要先删除
rm -rf /home/www/statics/*
否则会一个个文件提示你确认，使用cp -rf 也一样提示



## code 

warm up start:
```
    global_step = tf.train.get_or_create_global_step()
    lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)
```


## 年度总结

2019就要说再见了，回望这一年，有玩乐，有学习，有工作，有爱情。

是时候整理一下行囊了。

目前已经在凯天学习和工作半个月了，学习和工作的效果还是挺不错的，只是身体似乎有点疲惫，这应该是我玩手机游戏导致的。正好，明儿就换新手机了，新的手机是拿来学习和工作的，大屏幕且曲面屏要有意义，更舒适的看手机。

2020年，有很多事要去做，自己的小论文，自己的研究内容，技术的提升，个人项目。

更重要的，如何分配和平衡自己的时间，锻炼自己的身体和意志。

今晚要确定新的一年的安排：

## 语言模型与句子纠错

## 小论文

## 文本生成

## Bert运用

## 身体强度与街舞

## 面试准备

## pytorch & tensorflow

## 跟进最新论文

## 图像处理


这一年要掌握NLP的大部分工业应用，融汇贯通，后面的工作需要6K的实习工资。

这一年，少打游戏，多出去运动走走，保护好自己的眼睛。
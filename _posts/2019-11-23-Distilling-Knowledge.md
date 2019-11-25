---
layout:     post
title:      Distilling the Knowledge in a Neural Network
subtitle:   summary
date:       2019-11-23
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 前言
随着深度学习模型越来越大，效果也不断提升的同时，所需要的计算量和成本也在不断的提高，在保证模型效果差不多的情况下，压缩模型的参数变得尤为重要，就Bert模型而言，常见的压缩模型参数的方法如下：

1. Pruning 
    - Removes unnecessary parts of the network after training. This includes weight magnitude pruning, attention head pruning, layers, and others. Some methods also impose regularization during training to increase prunability (layer dropout).

2. Weight Factorization 
    - Approximates parameter matrices by factorizing them into a multiplication of two smaller matrices. This imposes a low-rank constraint on the matrix. Weight factorization can be applied to both token embeddings (which saves a lot of memory on disk) or parameters in feed-forward / self-attention layers (for some speed improvements).

3. Knowledge Distillation 
    - Aka “Student Teacher.” Trains a much smaller Transformer from scratch on the pre-training / downstream-data. Normally this would fail, but utilizing soft labels from a fully-sized model improves optimization for unknown reasons. Some methods also distill BERT into different architectures (LSTMS, etc.) which have faster inference times. Others dig deeper into the teacher, looking not just at the output but at weight matrices and hidden activations.

4. Weight Sharing 
    - Some weights in the model share the same value as other parameters in the model. For example, ALBERT uses the same weight matrices for every single layer of self-attention in BERT.

5. Quantization 
    - Truncates floating point numbers to only use a few bits (which causes round-off error). The quantization values can also be learned either during or after training.

6. Pre-train vs. Downstream 
    - Some methods only compress BERT w.r.t. certain downstream tasks. Others compress BERT in a way that is task-agnostic.


知识蒸馏手段是其中之一，今天将要学习知识蒸馏的经典文章，论文 ：<br>
[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

## 摘要

- A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions [3].
- Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. 
- Caruana and his collaborators [1] have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. 

We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by **distilling the knowledge** in an ensemble of models into a single model. 

We also introduce **a new type of ensemble** composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.

![](https://pic1.zhimg.com/v2-e6717bfe17182c4b6176a598b81ce176_r.jpg)

## 1. 引入
- Many insects have a larval form that is optimized for extracting energy and nutrients from the environment and a completely different adult form that is optimized for the very different requirements of traveling and reproduction. 
- In large-scale machine learning, we typically use very similar models for the training stage and the deployment stage despite their very different requirements: 
    - For tasks like speech and object recognition, training must extract structure from very large, highly redundant datasets but it does not need to operate in real time and it can use a huge amount of computation.
    - Deployment to a large number of users, however, has much more stringent requirements on latency and computational resources. 
- The analogy with insects suggests that we should be willing to train very cumbersome models if that makes it easier to extract structure from the data. The cumbersome model could be an ensemble of separately trained models or a single very large model trained with a very strong regularizer such as dropout [9]. 
- Once the cumbersome model has been trained, we can then use a different kind of training, which we call “distillation” to transfer the knowledge from the cumbersome model to a small model that is more suitable for deployment. 
- A version of this strategy has already been pioneered by Rich Caruana and his collaborators [1]. In their important paper they demonstrate convincingly that the knowledge acquired by a large ensemble of models can be transferred to a single small model.

通常训练模型的时候可以耗费较资源来提取数据中更好的特征来完成相关任务，但是部署的时候就要考虑延迟、计算量的问题，毕竟商业面向的用户群体是很庞大的。类似于昆虫的不同生命阶段：幼虫擅于从环境中吸取营养，成虫更适应迁移和繁殖。我们在训练模型的时候可以训练较大且较耗费资源的模型以提取更好的特征，但是部署的时候，我们可以采用不同的训练方法--知识蒸馏，使知识从复杂的训练模型迁移到一个较小的适合部署的模型上。

Exanple：“蝴蝶以毛毛虫的形式吃树叶积攒能量逐渐成长，最后变换成蝴蝶这一终极形态来完成繁殖。”

虽然是同一个个体，但是在面对不同环境以及不同任务时，个体的形态却是非常不同。不同的形态是为了完成特异性的任务而产生的变化，从而使个体能够更好的适应新的环境。
比如毛毛虫的形态是为了更方便的吃树叶，积攒能量，但是为了增大活动范围提高繁殖几率，毛毛虫要变成蝴蝶来完成这样的繁殖任务。蒸馏神经网络，其本质上就是要完成一个从毛毛虫到蝴蝶的转变。

- A conceptual block that may have prevented more investigation of this very promising approach is that we tend to identify the knowledge in a trained model with the learned parameter values and this makes it hard to see how we can change the form of the model but keep the same knowledge. 
- A more abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned mapping from input vectors to output vectors. 
- For cumbersome models that learn to discriminate between a large number of classes, the normal training objective is to maximize the average log probability of the correct answer, but a side-effect of the learning is that the trained model assigns probabilities to all of the incorrect answers and even when these probabilities are very small, some of them are much larger than others. 
- The relative probabilities of incorrect answers tell us a lot about how the cumbersome model tends to generalize. An image of a BMW, for example, may only have a very small chance of being mistaken for a garbage truck, but that mistake is still many times more probable than mistaking it for a carrot.

一个概念上的block阻碍了更多人进行这项研究：我们试图从已经训练好的模型的参数中发现知识，但是如何改变模型结构但保留原有的知识，这件事很难做到。抛开知识的实例化，知识可以看做为一个从输入向量到输出向量的映射。
在复杂模型判断多分类中，通常的训练目标函数是最大化正确类别的平均对数概率，但是这样学习的弊端是：训练的模型也分配了概率给了不正确的类别，尽管分配的概率很小，但是在这些错误概率之中，有一些概率也是远大于其他概率的。模型对不正确类别所分配的相对概率告诉我们这个复杂模型的泛化能力如何。比如在图像识别中，将宝马汽车判别为垃圾车的概率虽然小，但是将其误判成胡萝卜的概率更小。

- It is generally accepted that the objective function used for training should reflect the true objective of the user as closely as possible. Despite this, models are usually trained to optimize performance on the training data when the real objective is to generalize well to new data. It would clearly be better to train models to generalize well, but this requires information about the correct way to generalize and this information is not normally available. 
- When we are distilling the knowledge from a large model into a small one, however, we can train the small model to generalize in the same way as the large model. 
- If the cumbersome model generalizes well because, for example, it is the average of a large ensemble of different models, a small model trained to generalize in the same way will typically do much better on test data than a small model that is trained in the normal way on the same training set as was used to train the ensemble.

训练是的目标函数需要最大限度的反映正确目标，尽管如此模型还是通常被训练得在测试数据上表现的最优，但真实目标函数其实是需要在新的数据集上得到更好的泛化，这很难做到。

当我们从一个已经训练的复杂模型中提取知识到一个小的模型这一过程中，我们可以使这个小的模型获得与复杂模型相近的泛化能力。一个较小的模型被训练的与复杂集成模型具有相近的泛化能力将比一个正常训练的小模型表现的更好。

- An obvious way to transfer the generalization ability of the cumbersome model to a small model is
to use the class probabilities produced by the cumbersome model as “soft targets” for training the
small model.
- For this transfer stage, we could use the same training set or a separate “transfer” set. When the cumbersome model is a large ensemble of simpler models, we can use an arithmetic or geometric mean of their individual predictive distributions as the soft targets.
-  When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate.

使用复杂模型产出的各个类别概率作为小模型训练的软目标是实现泛化能力的迁移的一种方法。在这个迁移阶段，我们可以使用原来的训练集或者新的'迁移集'。当这个复杂模型是由多个简单模型集成的时候，我们可以使用它们各自预测概率分布的算术或者几何平均作为小模型训练的软目标。

- For tasks like MNIST in which the cumbersome model almost always produces the correct answer
with very high confidence, much of the information about the learned function resides in the ratios
of very small probabilities in the soft targets. For example, one version of a 2 may be given a
probability of 10−6 of being a 3 and 10−9 of being a 7 whereas for another version it may be the
other way around. This is valuable information that defines a rich similarity structure over the data
(i. e. it says which 2’s look like 3’s and which look like 7’s) but it has very little influence on the
cross-entropy cost function during the transfer stage because the probabilities are so close to zero.
- Caruana and his collaborators circumvent this problem by using the logits (the inputs to the final
softmax) rather than the probabilities produced by the softmax as the targets for learning the small
model and they minimize the squared difference between the logits produced by the cumbersome
model and the logits produced by the small model.
- Our more general solution, called “distillation”, is to raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets. We then use the same high temperature when training the small model to match these soft targets. We show later that matching the logits of the cumbersome model is actually a special case of distillation.

直接使用teacher网络的softmax的输出结果q，可能不大合适。因此，一个网络训练好之后，对于正确的答案会有一个很高的置信度。例如，在MNIST数据中，对于某个2的输入，对于2的预测概率会很高，而对于2类似的数字，例如3和7的预测概率为10<sup>-6</sup>和10<sup>-10</sup>。这样的话，teacher网络学到数据的相似信息（例如数字2和3，7很类似）很难传达给student网络。由于它们的概率值接近0。因此，文章提出了softmax-T:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123distillation.jpg)


## 2. Distillation
Neural networks typically produce class probabilities by using a “softmax” output layer that converts
the logit, z<sub>i</sub>, computed for each class into a probability, q<sub>i</sub>, by comparing z<sub>i</sub> with the other logits.where T is a temperature that is normally set to 1. Using a higher value for T produces a softer probability distribution over classes.
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123distillation.jpg)

- In the simplest form of distillation, knowledge is transferred to the distilled model by training it on a transfer set and using a soft target distribution(that is produced by using the cumbersome model with a high temperature in its softmax) for each case in the transfer set .
- The same high temperature is used when training the distilled model, but after it has been trained it uses a temperature of 1.

- When the correct labels are known for all or some of the transfer set, this method can be significantly
improved by also training the distilled model to produce the correct labels.
- One way to do this is to use the correct labels to modify the soft targets, but we found that a better way is to simply use a weighted average of two different objective functions. 
    - The first objective function is the cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model.
    - The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1. 

We found that the best results were generally obtained by using a considerably lower weight on the second
objective function. Since the magnitudes of the gradients produced by the soft targets scale as 1/T<sup>2</sup> it is important to multiply them by T<sup>2</sup> when using both hard and soft targets. This ensures that the relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters.


### 2.1 Matching logits is a special case of distillation

Each case in the transfer set contributes a cross-entropy gradient, dC/dz<sub>i</sub>, with respect to each logit, z<sub>i</sub> of the distilled model. If the cumbersome model has logits v<sub>i</sub> which produce soft target probabilities p<sub>i</sub> and the transfer training is done at a temperature of T , this gradient is given by:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123distillation2.jpg)

If the temperature is high compared with the magnitude of the logits, we can approximate:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123distillation3.jpg)

If we now assume that the logits have been zero-meaned separately for each transfer case so that
sum{z<sub>i</sub>}= sum{v<sub>i</sub>}=0  Eq. 3 simplifies to:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123distillation4.jpg)

## 3. Preliminary experiments on MNIST
- To see how well distillation works, we trained a single large neural net with two hidden layers
of 1200 rectified linear hidden units on all 60,000 training cases. The net was strongly regularized
using dropout and weight-constraints as described in [5].
- **Dropout can be viewed as a way of training an exponentially large ensemble of models that share weights.** In addition, the input images were jittered by up to two pixels in any direction.
- This net achieved 67 test errors whereas a smaller net with two hidden layers of 800 rectified linear hidden units and no regularization achieved 146 errors. 
- But if the smaller net was regularized solely by adding the additional task of matching the soft targets produced by the large net at **a temperature of 20**, it achieved 74 test errors.
- This shows that soft targets can transfer a great deal of knowledge to the distilled model, including the knowledge about how to generalize that is learned from translated training data even though the transfer set does not contain any translations.

When the distilled net had 300 or more units in each of its two hidden layers, all temperatures above
8 gave fairly similar results. But when this was radically reduced to 30 units per layer, temperatures
in the range 2.5 to 4 worked significantly better than higher or lower temperatures.

We then tried omitting all examples of the digit 3 from the transfer set. So from the perspective
of the distilled model, 3 is a mythical digit that it has never seen. Despite this, the distilled model
only makes 206 test errors of which 133 are on the 1010 threes in the test set. Most of the errors
are caused by the fact that the learned bias for the 3 class is much too low. If this bias is increased
by 3.5 (which optimizes overall performance on the test set), the distilled model makes 109 errors
of which 14 are on 3s. So with the right bias, the distilled model gets 98.6% of the test 3s correct
despite never having seen a 3 during training. If the transfer set contains only the 7s and 8s from the
training set, the distilled model makes 47.3% test errors, but when the biases for 7 and 8 are reduced
by 7.6 to optimize test performance, this falls to 13.2% test errors.

## 4. Experiments on speech recognition



### 4.1 Results

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191125experimenttable.jpg)
## 5 Training ensembles of specialists on very big datasets

### 5.1 The JFT dataset

### 5.2 Specialist Models

### 5.3 Assigning classes to specialists

### 5.4 Performing inference with ensembles of specialists

### 5.5 Results

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191125experimenttable2.jpg)
## 6 Soft Targets as Regularizers


### 6.1 Using soft targets to prevent specialists from overfitting



## 7 Relationship to Mixtures of Experts
The use of specialists that are trained on subsets of the data has some resemblance to mixtures of
experts [6] which use a gating network to compute the probability of assigning each example to each
expert. At the same time as the experts are learning to deal with the examples assigned to them, the
gating network is learning to choose which experts to assign each example to based on the relative
discriminative performance of the experts for that example. Using the discriminative performance
of the experts to determine the learned assignments is much better than simply clustering the input
vectors and assigning an expert to each cluster, but it makes the training hard to parallelize: First,
the weighted training set for each expert keeps changing in a way that depends on all the other
experts and second, the gating network needs to compare the performance of different experts on
the same example to know how to revise its assignment probabilities. These difficulties have meant
that mixtures of experts are rarely used in the regime where they might be most beneficial: tasks
with huge datasets that contain distinctly different subsets.

It is much easier to parallelize the training of multiple specialists. We first train a generalist model
and then use the confusion matrix to define the subsets that the specialists are trained on. Once these
subsets have been defined the specialists can be trained entirely independently. At test time we can
use the predictions from the generalist model to decide which specialists are relevant and only these
specialists need to be run.


## 8 Discussion
We have shown that distilling works very well for transferring knowledge from an ensemble or
from a large highly regularized model into a smaller, distilled model. On MNIST distillation works
remarkably well even when the transfer set that is used to train the distilled model lacks any examples
of one or more of the classes. For a deep acoustic model that is version of the one used by Android
voice search, we have shown that nearly all of the improvement that is achieved by training an
ensemble of deep neural nets can be distilled into a single neural net of the same size which is far
easier to deploy.

For really big neural networks, it can be infeasible even to train a full ensemble, but we have shown
that the performance of a single really big net that has been trained for a very long time can be significantly improved by learning a large number of specialist nets, each of which learns to discriminate
between the classes in a highly confusable cluster. We have not yet shown that we can distill the
knowledge in the specialists back into the single large net.



## 核心
**Dropout can be viewed as a way of training an exponentially large ensemble of models that share weights.**



### 公式
2006年的Model Compression提出的方法是直接比较logits来避免这个问题。具体地，对于每一条数据，记原模型产生的某个logits是 v<sub>i</sub> ，新模型产生的logits是z<sub>i</sub> ，我们需要最小化：

1/2(z<sub>i</sub> - v<sub>i</sub>)^2

而本文提出了一个更通用的方法Softmax-T:

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123distillation.jpg)

其中 T 是温度，这是从统计力学中的玻尔兹曼分布中借用的概念。容易证明，当温度  T  趋向于0时，softmax输出将收敛为一个one-hot向量；温度 T 趋向于无穷时，softmax的输出则更「软」。因此，在训练新模型的时候，可以使用较高的  T  使得softmax产生的分布足够软，这时让新模型的softmax输出近似原模型；在训练结束以后再使用正常的温度 1 来预测。具体地，在训练时我们需要最小化两个分布的交叉熵(Cross-entropy)，记新模型利用公式 Softmax-T 产生的分布是 q ，原模型产生的分布是 p ，则我们需要最小化:

C = -p<sup>T</sup>log q

在化学中，蒸馏是一个有效的分离沸点不同的组分的方法，大致步骤是先升温使低沸点的组分汽化，然后降温冷凝，达到分离出目标物质的目的。在前面提到的这个过程中，我们先让温度 T 升高，然后在测试阶段恢复「低温」，从而将原模型中的知识提取出来，因此将其称为是蒸馏，实在是妙。

相关公式推导见[公式](https://zhuanlan.zhihu.com/p/90049906)

### 蒸馏
蒸馏神经网络取名为蒸馏（Distill），其实是一个非常形象的过程。

我们把数据结构信息和数据本身当作一个混合物，分布信息通过概率分布被分离出来。首先，T值很大，相当于用很高的温度将关键的分布信息从原有的数据中分离，之后在同样的温度下用新模型融合蒸馏出来的数据分布，最后恢复温度，让两者充分融合。这也可以看成Prof. Hinton将这一个迁移学习过程命名为蒸馏的原因。

### 蒸馏经验

Transfer Set和Soft target

实验证实，Soft target可以起到正则化的作用（不用soft target的时候需要early stopping，用soft target后稳定收敛）数据过少的话无法完整表达teacher学到的知识，需要增加无监督数据（用teacher的预测作为标签）或进行数据增强，可以使用的方法有：1.增加[MASK]，2.用相同POS标签的词替换，3.随机n-gram采样，具体步骤参考文献

超参数T

T越大越能学到teacher模型的泛化信息。比如MNIST在对2的手写图片分类时，可能给2分配0.9的置信度，3是1e-6，7是1e-9，从这个分布可以看出2和3有一定的相似度，因此这种时候可以调大T，让概率分布更平滑，展示teacher更多的泛化能力T可以尝试1～20之间

BERT蒸馏蒸馏

单BERT：模型架构：

- 单层BiLSTM；目标函数：logits的MSE蒸馏
- Ensemble BERT：模型架构：BERT；目标函数：soft prob+hard prob；方法：MT-DNN。该论文用给每个任务训练多个MT-DNN，取soft target的平均，最后再训一个MT-DNN，效果比纯BERT好3.2%。但感觉该研究应该是刷榜的结晶，平常应该没人去训BERT ensemble吧。。
- BAM：Born-aging Multi-task。用多个任务的Single BERT，蒸馏MT BERT；目标函数：多任务loss的和；方法：在mini-batch中打乱多个任务的数据，任务采样概率为  ，防止某个任务数据过多dominate模型、teacher annealing、layerwise-learning-rate，LR由输出层到输出层递减，因为前面的层需要学习到general features。最终student在大部分任务上超过teacher，而且上面提到的tricks也提供了不少帮助。文献4还不错，推荐阅读一下。
- TinyBERT：截止201910的SOTA。利用Two-stage方法，分别对预训练阶段和精调阶段的BERT进行蒸馏，并且不同层都设计了损失函数。与其他模型的对比如下：

![](https://pic4.zhimg.com/v2-06423040ac6234d719d80cab1820adbb_b.jpg)
## 参考
[All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)

[知识蒸馏简述-Ivan Yan](https://www.zhihu.com/people/Ivan0131/posts)

[知识蒸馏是什么？](https://zhuanlan.zhihu.com/p/90049906)

[Caruana et al., Model Compression, 2006](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)

[蒸馏神经网络到底在蒸馏什么？（设计思想篇）](https://zhuanlan.zhihu.com/p/39945855)


## MNIST蒸馏实验

### 较为复杂的三层卷积池化网络，训练100个epoch得到一个较为复杂的网络
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 1000
net_name = 'conv_net_3x3'


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))  # l1a shape=(?, 28, 28, 32)
    #
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])  # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625])  # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

prec = tf.reduce_mean(tf.cast(tf.equal(predict_op, tf.argmax(Y, 1)), tf.float32))

summary_cost = tf.summary.scalar('cost', cost)
summary_prec = tf.summary.scalar('prec', prec)

train_writer = tf.summary.FileWriter("logs/" + net_name + "/train", flush_secs=5)
test_writer = tf.summary.FileWriter("logs/" + net_name + "/test", flush_secs=5)

sav = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    k = 1
    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            log_cost, log_prec, val_prec, _ = sess.run([summary_cost, summary_prec, prec, train_op],
                                                       feed_dict={X: trX[start:end], Y: trY[start:end],
                                                                  p_keep_conv: 0.8, p_keep_hidden: 0.5})
            train_writer.add_summary(log_cost, k)
            train_writer.add_summary(log_prec, k)
            print(i, k, "train prec", val_prec)
            k = k + 1

        log_prec, val_prec = sess.run([summary_prec, prec], feed_dict={X: teX[:test_size],
                                                                       Y: teY[:test_size],
                                                                       p_keep_conv: 1.0,
                                                                       p_keep_hidden: 1.0})
        test_writer.add_summary(log_prec, k)
        print(i, val_prec)
        sav.save(sess, "checkpoints/" + net_name, global_step=i)

```

### 利用复杂模型的参数，在小的模型上进行蒸馏实验

- 在这个过程中比较重要的一步是根据模型训练保存的文件恢复模型的参数：

```
# donor & acceptor networks

y_donor, donor_params = lenet4()
y_acceptor, acceptor_params = fully_connected()

...

# Donor loading
donor_saver = tf.train.Saver(donor_params)
donor_saver.restore(sess, 'checkpoints/source_model/' + donor_name)
```
- 然后在小模型上将复杂模型的logits（即softtarget）计入损失目标函数，但不对复杂模型的logits进行反向梯度传播。
```
# distillation
# stop_gradient 阻止输入的节点的loss计入梯度计算；
distill_ent = tf.reduce_mean(- tf.reduce_sum(tf.stop_gradient(y_donor) * tf.log(y_acceptor), reduction_indices=1))
truth_ent = tf.reduce_mean(- tf.reduce_sum(Y * tf.log(y_acceptor), reduction_indices=1))

distil_step = tf.train.GradientDescentOptimizer(0.1).minimize(distill_ent + truth_ent)
```

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 1000

donor_name = 'conv_net_3x3-99'
acceptor_name = 'distil_comb_L2'


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

T = 1


# lenet Model

def lenet4_model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    #  三个卷积+池化
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


def lenet4():
    w = init_weights([3, 3, 1, 32])  # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 4 * 4, 625])  # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

    return tf.nn.softmax((1.0 / T) * lenet4_model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)), [w, w2, w3, w4,
                                                                                                        w_o]


#  return softmax-T*logits_v, param{ [w, w2, w3, w4, w_o] }


# NN Model

def fc_layer(inp, size, name):
    W_layer = tf.Variable(tf.truncated_normal([inp.get_shape()[1].value, size], stddev=0.1))
    b_layer = tf.Variable(tf.constant(0, tf.float32, shape=[size]))
    res = tf.matmul(inp, W_layer) + b_layer
    tf.summary.histogram(name + "_weights", W_layer)
    return res, [W_layer, b_layer]


#  return softmax-T*logits_z, param{ [W_layer1, b_layer1,  W_layer2, b_layer2]  }
def fully_connected():
    L1, fc_params1 = fc_layer(tf.reshape(X, [-1, 784]), 100, "L1")
    L1a = tf.nn.sigmoid(L1)
    res, fc_params2 = fc_layer(L1a, 10, "L2")
    return tf.nn.softmax((1.0 / T) * res), fc_params1 + fc_params2


# donor & acceptor networks
y_donor, donor_params = lenet4()
y_acceptor, acceptor_params = fully_connected()

# distillation
# stop_gradient 阻止输入的节点的loss计入梯度计算；
distill_ent = tf.reduce_mean(- tf.reduce_sum(tf.stop_gradient(y_donor) * tf.log(y_acceptor), reduction_indices=1))
truth_ent = tf.reduce_mean(- tf.reduce_sum(Y * tf.log(y_acceptor), reduction_indices=1))

distil_step = tf.train.GradientDescentOptimizer(0.1).minimize(distill_ent + truth_ent)


# Summaries

def prec(y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1)), tf.float32))


# MNIST data(distill_ent + truth_ent)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img


# distillate knowledge from donor to acceptor
def distillate(net_name):
    acceptor_prec = prec(y_acceptor)
    donor_prec = prec(y_donor)

    distil_writer = tf.summary.FileWriter("logs/" + net_name + "/train", flush_secs=5)
    distil_test_writer = tf.summary.FileWriter("logs/" + net_name + "/test", flush_secs=5)

    tf.summary.scalar('accuracy', acceptor_prec)
    summaries = tf.summary.merge_all()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        # Donor loading
        donor_saver = tf.train.Saver(donor_params)
        donor_saver.restore(sess, 'checkpoints/source_model/' + donor_name)

        acc_saver = tf.train.Saver(acceptor_params)

        k = 1
        for i in range(10):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))
            for start, end in training_batch:
                val_prec, val_donor_prec, log_summaries, val_distil_ent, val_truth_ent, _ = sess.run(
                    [acceptor_prec, donor_prec, summaries, distill_ent, truth_ent, distil_step],
                    feed_dict={X: trX[start:end],
                               Y: trY[start:end],
                               p_keep_conv: 1,
                               p_keep_hidden: 1})

                test_val_prec, test_log_summaries = sess.run([acceptor_prec, summaries],
                                                             feed_dict={X: teX[:test_size],
                                                                        Y: teY[:test_size],
                                                                        p_keep_conv: 1.0,
                                                                        p_keep_hidden: 1.0})

                distil_writer.add_summary(log_summaries, k)
                distil_test_writer.add_summary(test_log_summaries, k)
                print(i, k, 'distillation distil_ent:', val_distil_ent, 'truth_ent', val_truth_ent, 'donor_prec: ',
                      val_donor_prec, '; train_prec', val_prec, '; test_prec', test_val_prec)

                k = k + 1

            acc_saver.save(sess, "checkpoints/" + net_name, global_step=i)


# train net using back-prop
def train_net(train_step, net_prec, net_params, net_name):
    net_prec = prec(y_acceptor)

    writer = tf.summary.FileWriter("logs/" + net_name + "/train", flush_secs=5)
    test_writer = tf.summary.FileWriter("logs/" + net_name + "/test", flush_secs=5)

    tf.scalar_summary('accuracy', net_prec)
    summaries = tf.merge_all_summaries()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        net_saver = tf.train.Saver(net_params)

        k = 1
        for i in range(10):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))
            for start, end in training_batch:
                val_prec, log_summaries, _ = sess.run([net_prec, summaries, train_step],
                                                      feed_dict={X: trX[start:end],
                                                                 Y: trY[start:end],
                                                                 p_keep_conv: 1,
                                                                 p_keep_hidden: 1})

                test_val_prec, test_log_summaries = sess.run([net_prec, summaries],
                                                             feed_dict={X: teX[:test_size],
                                                                        Y: teY[:test_size],
                                                                        p_keep_conv: 1,
                                                                        p_keep_hidden: 1})

                writer.add_summary(log_summaries, k)
                test_writer.add_summary(test_log_summaries, k)
                print("epoch:", i, k, 'train_prec', val_prec, '; test_prec', test_val_prec)
                k = k + 1
            net_saver.save(sess, "checkpoints/" + net_name, global_step=i)


train_cross_ent = tf.reduce_mean(- tf.reduce_sum(Y * tf.log(y_acceptor), reduction_indices=1))
train_prec = prec(y_acceptor)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(train_cross_ent)

distillate(acceptor_name)
# train_net(train_step, train_prec, acceptor_params, "L2")

```

## 实验结果

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191125distill_compare.jpg)

从图中我们可以看到进行知识蒸馏的模型比原来的模型收敛的更快，精度也更高
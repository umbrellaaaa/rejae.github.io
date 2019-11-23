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

- A conceptual block that may have prevented more investigation of this very promising approach is that we tend to identify the knowledge in a trained model with the learned parameter values and this makes it hard to see how we can change the form of the model but keep the same knowledge. 
- A more abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned mapping from input vectors to output vectors. 
- For cumbersome models that learn to discriminate between a large number of classes, the normal training objective is to maximize the average log probability of the correct answer, but a side-effect of the learning is that the trained model assigns probabilities to all of the incorrect answers and even when these probabilities are very small, some of them are much larger than others. 
- The relative probabilities of incorrect answers tell us a lot about how the cumbersome model tends to generalize. An image of a BMW, for example, may only have a very small chance of being mistaken for a garbage truck, but that mistake is still many times more probable than mistaking it for a carrot.

一个概念上的block阻碍了更多人进行这项研究：我们试图从已经训练好的模型的参数中发现知识，但是如何改变模型结构但保留原有的知识，这件事很难做到。抛开知识的实例化，知识可以看做为一个从输入向量到输出向量的映射。
在复杂模型判断多分类中，通常的训练目标函数是最大化正确类别的平均对数概率，但是这样学习的弊端是：训练的模型也分配了概率给了不正确的类别，尽管分配的概率很小，但是在这些错误概率之中，有一些概率也是远大于其他概率的。模型对不正确类别所分配的相对概率告诉我们这个复杂模型的泛化能力如何。比如在图像识别中，将收垃圾的车判别为宝马汽车的概率虽然小，但是将其误判成胡萝卜的概率更小。

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


## 4. Experiments on speech recognition



### 4.1 Results


## 5 Training ensembles of specialists on very big datasets

### 5.1 The JFT dataset



### 5.2 Specialist Models

### 5.3 Assigning classes to specialists


### 5.4 Performing inference with ensembles of specialists



### 5.5 Results


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




## 参考
[All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)

[知识蒸馏简述-Ivan Yan](https://www.zhihu.com/people/Ivan0131/posts)
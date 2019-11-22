---
layout:     post
title:      梯度爆炸
subtitle:   cheer up
date:       2019-6-6
author:     RJ
header-img: 
catalog: true
tags:
    - DL

---
<p id = "build"></p>
---


在神经网络中，梯度下降算法是使用非常广泛的优化算法。梯度下降算法的变体有好多，比如随机梯度下降（Stochastic gradient descent，SGD）、小批量梯度下降（Mini Batch Gradient Descent）等，但对于梯度下降算法而言，难免需要考虑梯度下降算法中遇到的梯度弥散以及梯度爆炸等问题，本文主要讲解神经网络中的梯度爆炸问题，从以下三个方面讲解：

什么是梯度爆炸，以及在训练过程中梯度爆炸会引发哪些问题；
如何知道网络模型是否存在梯度爆炸；
如何在网络模型中解决梯度爆炸问题；

梯度爆炸是什么？

误差梯度在网络训练时被用来得到网络参数更新的方向和幅度，进而在正确的方向上以合适的幅度更新网络参数。在深层网络或递归神经网络中，误差梯度在更新中累积得到一个非常大的梯度，这样的梯度会大幅度更新网络参数，进而导致网络不稳定。在极端情况下，权重的值变得特别大，以至于结果会溢出（NaN值，无穷与非数值）。当梯度爆炸发生时，网络层之间反复乘以大于1.0的梯度值使得梯度值成倍增长。

梯度爆炸会引发哪些问题？

在深度多层感知机网络中，梯度爆炸会导致网络不稳定，最好的结果是无法从训练数据中学习，最坏的结果是由于权重值为NaN而无法更新权重。

梯度爆炸会使得学习不稳定；

—— 深度学习第282页

在循环神经网络（RNN）中，梯度爆炸会导致网络不稳定，使得网络无法从训练数据中得到很好的学习，最好的结果是网络不能在长输入数据序列上学习。

梯度爆炸问题指的是训练过程中梯度大幅度增加，这是由于长期组件爆炸造成的；

——训练循环神经网络中的困难

如何知道网络中是否有梯度爆炸问题？

在网络训练过程中，如果发生梯度爆炸，那么会有一些明显的迹象表明这一点，例如：

模型无法在训练数据上收敛（比如，损失函数值非常差）；
模型不稳定，在更新的时候损失有较大的变化；
模型的损失函数值在训练过程中变成NaN值；
如果你遇到上述问题，我们就可以深入分析网络是否存在梯度爆炸问题。还有一些不太为明显的迹象可以用来确认网络中是否存在梯度爆炸问题：

模型在训练过程中，权重变化非常大；
模型在训练过程中，权重变成NaN值；
每层的每个节点在训练时，其误差梯度值一直是大于1.0；
如何解决梯度爆炸问题？

解决梯度爆炸问题的方法有很多，本部分将介绍一些有效的实践方法：

1.重新设计网络模型

在深层神经网络中，梯度爆炸问题可以通过将网络模型的层数变少来解决。此外，在训练网络时，使用较小批量也有一些好处。在循环神经网络中，训练时使用较小时间步长更新（也被称作截断反向传播）可能会降低梯度爆炸发生的概率。

2.使用修正线性激活函数

在深度多层感知机中，当激活函数选择为一些之前常用的Sigmoid或Tanh时，网络模型会发生梯度爆炸问题。而使用修正线性激活函数（ReLU）能够减少梯度爆炸发生的概率，对于隐藏层而言，使用修正线性激活函数（ReLU）是一个比较合适的激活函数，当然ReLU函数有许多变体，大家在实践过程中可以逐一使用以找到最合适的激活函数。

3.使用长短周期记忆网络

由于循环神经网络中存在的固有不稳定性，梯度爆炸可能会发生。比如，通过时间反向传播，其本质是将循环网络转变为深度多层感知神经网络。通过使用长短期记忆单元（LSTM）或相关的门控神经结构能够减少梯度爆炸发生的概率。

对于循环神经网络的时间序列预测而言，采用LSTM是新的最佳实践。

4.使用梯度裁剪

在深度多层感知网络中，当有大批量数据以及LSTM是用于很长时间序列时，梯度爆炸仍然会发生。当梯度爆炸发生时，可以在网络训练时检查并限制梯度的大小，这被称作梯度裁剪。

梯度裁剪是处理梯度爆炸问题的一个简单但非常有效的解决方案，如果梯度值大于某个阈值，我们就进行梯度裁剪。

——自然语言处理中的神经网络方法的第5.2.4节

具体而言，检查误差梯度值就是与一个阈值进行比较，若误差梯度值超过设定的阈值，则截断或设置为阈值。

在某种程度上，梯度爆炸问题可以通过梯度裁剪来缓解（在执行梯度下降步骤之前对梯度进行阈值操作）

——深度学习第294页

在Keras深度学习库中，在训练网络之前，可以对优化器的clipnorm和 clipvalue参数进行设置来使用梯度裁剪，一般而言，默认将clipnorm和 clipvalue分别设置为1和0.5.

在Keras API中使用优化器
5.使用权重正则化

如果梯度爆炸问题仍然发生，另外一个方法是对网络权重的大小进行校验，并对大权重的损失函数增添一项惩罚项，这也被称作权重正则化，常用的有L1（权重的绝对值和）正则化与L2（权重的绝对值平方和再开方）正则化。

使用L1或L2惩罚项会减少梯度爆炸的发生概率

——训练循环神经网络中的困难

在Keras深度学习库中，可以在每层上使用L1或L2正则器设置kernel_regularizer参数来完成权重的正则化操作。

在Keras API中使用正则器
进一步阅读

如果你想进一步深入研究梯度爆炸问题，本节将提供更多的资源：

书籍

深度学习；
自然语音处理中的神经网络方法；
文献

训练循环网络中的困难；
学习具有梯度下降的长期依赖是困难的；
理解梯度爆炸问题；
文章

为什么在神经网络中梯度爆炸是个问题（尤其是循环神经网络）？
在循环神经网络中，LSTM是如何防止梯度弥散和梯度爆炸问题？
线性整流函数；
Keras API

在Keras API中使用优化器；
在Keras API中使用正则化；
总结

通过本文，你将会学习到如何在训练深层神经网络时发现梯度爆炸问题，尤其是以下几点：

什么是梯度爆炸，以及在训练过程中梯度爆炸会引发哪些问题；
如何知道网络模型是否存在梯度爆炸；
如何在网络模型中解决梯度爆炸问题；
作者信息

![](https://pic3.zhimg.com/80/v2-6d4161243d0cc6cfd9f2a4f216243456_hd.jpg)
Jason Brownlee，机器学习专家，专注于机器学习的推广教育。

Linkedin: http://www.linkedin.com/in/jasonbrownlee/

本文由阿里云云栖社区组织翻译。
---
layout:     post
title:      DL function
subtitle:   
date:       2019-11-29
author:     RJ
header-img: 
catalog: true
tags:
    - DL
---
<p id = "build"></p>
---

## 前言 
深度学习中有很多地方是需要注意的，正好开始整理自己所学的文档了，今天遇到了权重初始化的问题，那么这部分知识就正式开始整理了。

## 权重初始化
[权重初始化参考](https://zhuanlan.zhihu.com/p/25110150)

初始化为0的可行性？

答案是不可行。 为什么将所有W初始化为0是错误的呢？是因为如果所有的参数都是0，那么所有神经元的输出都将是相同的，那在back propagation的时候同一层内所有神经元的行为也是相同的 --- gradient相同，weight update也相同。这显然是一个不可接受的结果。

可行的几种初始化方式：
- pre-training 
- random initialization 
- Xavier initialization ：  W = tf.Variable(np.random.randn(node_in, node_out)) / np.sqrt(node_in)

- He initialization    ：   W = tf.Variable(np.random.randn(node_in,node_out)) / np.sqrt(node_in/2)

其中    node_in = layer_sizes[i]  ; node_out = layer_sizes[i + 1]


### MNIST测试初始化效果

```
'*****************random initialize w************************'

Epoch:0001cost=8.209401439
Accuracy:0.2899
Epoch:0002cost=4.501752799
Accuracy:0.4548
Epoch:0003cost=3.084369575
Accuracy:0.56
Epoch:0004cost=2.427498819
Accuracy:0.6199
Epoch:0005cost=2.058773852
Accuracy:0.6606
Epoch:0006cost=1.821953765
Accuracy:0.6884
Epoch:0007cost=1.655686877
Accuracy:0.7084
Epoch:0008cost=1.531370343
Accuracy:0.7262
Epoch:0009cost=1.434300166
Accuracy:0.7388
Epoch:0010cost=1.355875988
Accuracy:0.751
Epoch:0011cost=1.291047658
Accuracy:0.7599
Epoch:0012cost=1.236139774
Accuracy:0.7674
Epoch:0013cost=1.189290791
Accuracy:0.7735
Epoch:0014cost=1.148372694
Accuracy:0.7801
Epoch:0015cost=1.112297407
Accuracy:0.7872
Epoch:0016cost=1.080240906
Accuracy:0.7917
Epoch:0017cost=1.051378414
Accuracy:0.7973
Epoch:0018cost=1.025336763
Accuracy:0.8006
Epoch:0019cost=1.001531084
Accuracy:0.805
Epoch:0020cost=0.979600413
Accuracy:0.8094

'*****************xavier initialize w************************'
Epoch:0001cost=1.201949758
Accuracy:0.8406
Epoch:0002cost=0.673215782
Accuracy:0.8671
Epoch:0003cost=0.557662858
Accuracy:0.8766
Epoch:0004cost=0.502294260
Accuracy:0.8844
Epoch:0005cost=0.468406819
Accuracy:0.89
Epoch:0006cost=0.444951771
Accuracy:0.8932
Epoch:0007cost=0.427721882
Accuracy:0.8961
Epoch:0008cost=0.414128707
Accuracy:0.8984
Epoch:0009cost=0.403213230
Accuracy:0.9
Epoch:0010cost=0.394078740
Accuracy:0.9025
Epoch:0011cost=0.386388999
Accuracy:0.903
Epoch:0012cost=0.379650087
Accuracy:0.9047
Epoch:0013cost=0.373825378
Accuracy:0.9058
Epoch:0014cost=0.368685644
Accuracy:0.9074
Epoch:0015cost=0.364035460
Accuracy:0.9074
Epoch:0016cost=0.359817683
Accuracy:0.9089
Epoch:0017cost=0.356041863
Accuracy:0.9093
Epoch:0018cost=0.352607728
Accuracy:0.9102
Epoch:0019cost=0.349460995
Accuracy:0.9106
Epoch:0020cost=0.346524342
Accuracy:0.911
```

经过对比发现，初始化不同，在20个epoch里，差异是巨大的。随机初始化经过11个epoch才达到xavier初始化第一个epoch的cost, 意味着整整多跑了11个无用的epoch。除此之外，xavier的第二个epoch就超过了随机初始化的精度，这说明了axvier初始化的参数更逼近真实参数。所以，以后你还用tf.random_normal吗？



## 激活函数
[机器之心](https://www.jiqizhixin.com/graph/technologies/1697e627-30e7-48a6-b799-39e2338ffab5)

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191214activation.png)

sigmoid

relu

elu FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)

gelu

在激活函数领域，大家公式的鄙视链应该是：Elus > Relu > Sigmoid ，这些激活函数都有自身的缺陷， sigmoid容易饱和，Elus与Relu缺乏随机因素。

在神经网络的建模过程中，模型很重要的性质就是非线性，同时为了模型泛化能力，需要加入随机正则，例如dropout(随机置一些输出为0,其实也是一种变相的随机非线性激活)， 而随机正则与非线性激活是分开的两个事情， 而其实模型的输入是由非线性激活与随机正则两者共同决定的。

GELUs正是在激活中引入了随机正则的思想，是一种对神经元输入的概率描述，直观上更符合自然的认识，同时实验效果要比Relus与ELUs都要好。

什么样的函数能用来做激活函数这里 ?

除了单调性有争议外，其他的目前激活函数大都满足。
大体说来需要满足：
- 非线性   
- 几乎处处可微   
- 计算简单   
- 非饱和性   
- 单调性   
- 输出范围有限   
- 接近恒等变换   
- 参数少   
- 归一化


### GELUs （区别于glue data）
高斯误差线性单元：

The GELU nonlinearity is the expected transformation of a stochastic regularizer which randomly applies the identity or zero map to a neuron’s input.

The GELU nonlinearity weights inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.

GELUs其实是 dropout、zoneout、Relus的综合，GELUs对于输入乘以一个0,1组成的mask，而该mask的生成则是依概率随机的依赖于输入。假设输入为X, mask为m，则m服从一个伯努利分布(Φ(x)\Phi(x)Φ(x), Φ(x)=P(X&lt;=x),X服从标准正太分布\Phi(x)=P(X&lt;=x), X服从标准正太分布Φ(x)=P(X<=x),X服从标准正太分布)，这么选择是因为神经元的输入趋向于正太分布，这么设定使得当输入x减小的时候，输入会有一个更高的概率被dropout掉，这样的激活变换就会随机依赖于输入了。
数学表达如下：
GELU(x)=xP(X&lt;=x)=xΦ(x)GELU(x) = xP(X&lt;=x) = x\Phi(x)
GELU(x)=xP(X<=x)=xΦ(x)

这里Φ(x)是正太分布的概率函数，可以简单采用正太分布N(0,1), 要是觉得不刺激当然可以使用参数化的正太分布N(μ,σ), 然后通过训练得到μ,σ

对于假设为标准正太分布的GELU(x)GELU(x)GELU(x), 论文中提供了近似计算的数学公式，如下：

GELU(x)=0.5x(1+tanh[2/π−−−√(x+0.044715x3)])

GELU(x) = 0.5x(1+tanh[\sqrt{2/\pi}(x+0.044715x^3)])

GELU(x)=0.5x(1+tanh[ 
2/π
​	
 (x+0.044715x 
3
 )])

bert源码中的近似计算更简单:

```python
def gelu(input_tensor):
	cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
	return input_tesnsor*cdf

```
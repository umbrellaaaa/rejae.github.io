---
layout:     post
title:      NLP 
subtitle:   summary
date:       2019-10-18
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---

<p id = "build"></p>
---

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/NLPtask20191214.jpg)

## 面试问题
1. 各种优化器(SGD,GD, adam等)
2. 各种激活函数(sigmoid,relu等)
3. 梯度消失问题(DNN,CNN,RNN中存在的，以及如何解决的)
4. 超参数（bs,lr，max_len），训练的trick等
5. 如何防止过拟合以及为什么可以 （BN droupout 早停 L1L2）
6. Transform 的结构，解码器和编码器（什么样的以及区别），self-attention是怎么计算的，kqv，以及多头是怎么实现的
7. LSTM 各个门 RNN的梯度消失问题（LSTM画出来，以及解决方式）
8. Bert的训练机制（两个训练任务，为什么会用mask，预测两个是否连续），bert的输入，三个embedding，token，segment，position
9. LR的推导
10. DNN和LR异同 
11. 网络降维的方法，1*1的卷积核的作用（降维和升维）
12. 梯度消失的两种解决方案（LSTM和resnet）
13. Word2vec 两种训练方式 和fasttext对比
14. Fasttext为什么快，两种优化训练的方式，以及fasttext为什么会快，以及能否进行无监督的任务
15. DSSM以及变体
16. NLP的传统方法，ngram 以及HMM，CRF
17. Kmeans以及kmeans++以及K的选取
18. LR为什么会选择sigmoid，处理根号dk，用一句话来概括LR
19. 信息熵 交叉熵 相对熵的概念
20. 局部最优 鞍点 (SGD)
21. 循环神经网络和递归神经网络的区别
22. LSTM结构中C<sub>t</sub>与h<sub>t</sub>的关系是什么？
23. 自编码器的理解和应用
24. 线性回归与逻辑回归以及最大熵模型
25. Batch N 与 Layer N ？




## 问题解决
1. 各种优化器：

SGD:

GD:

adam:


2. 各种激活函数

一般激活函数有如下一些性质：
- 非线性：  当激活函数是线性的，一个两层的神经网络就可以基本上逼近所有的函数。但如果激活函数是恒等激活函数的时候，即f(x)=x，就不满足这个性质，而且如果MLP使用的是恒等激活函数，那么其实整个网络跟单层神经网络是等价的；
- 可微性：  当优化方法是基于梯度的时候，就体现了该性质；
- 单调性：  当激活函数是单调的时候，单层网络能够保证是凸函数；
- f(x)≈x：  当激活函数满足这个性质的时候，如果参数的初始化是随机的较小值，那么神经网络的训练将会很高效；如果不满足这个性质，那么就需要详细地去设置初始值；
- 输出值的范围：  当激活函数输出值是有限的时候，基于梯度的优化方法会更加稳定，因为特征的表示受有限权值的影响更显著；当激活函数的输出是无限的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的Learning Rate。


sigmoid: pass

relu: pass

gelu:  

In all, the activation choice has remained a necessary architecture decision for neural networks lest the network be a deep linear classifier. 激活函数结构的必要性在于使神经网络不会成为一个深度线性分类器。即激活函数使得神经网络具有非线性特点。


3. 梯度消失/爆炸为什么出现，以及怎样解决？

()[https://zhuanlan.zhihu.com/p/28687529]


21. 循环神经网络和递归神经网络的区别
recurrent: 时间维度的展开，代表信息在时间维度从前往后的的传递和积累，可以类比markov假设，后面的信息的概率建立在前面信息的基础上，在神经网络结构上表现为后面的神经网络的隐藏层的输入是前面的神经网络的隐藏层的输出；recursive: 空间维度的展开，是一个树结构，比如nlp里某句话，用recurrent neural network来建模的话就是假设句子后面的词的信息和前面的词有关，而用recurxive neural network来建模的话，就是假设句子是一个树状结构，由几个部分(主语，谓语，宾语）组成，而每个部分又可以在分成几个小部分，即某一部分的信息由它的子树的信息组合而来，整句话的信息由组成这句话的几个部分组合而来。

22. 在LSTM中Ct是状态输出，ht是隐层结合了Ct的输出。Ct是细胞状态，用于记忆，而ht是根据当前数据x的输入以及当前的细胞状态共同决定当前隐层的输出。从公式上表达是：

<center> h<sub>t</sub> = sigmod(w<sub>x</sub>x<sub>t</sub> + w<sub>h</sub>h<sub>t-1</sub>) *tan(C<sub>t</sub>)</center> 

其中的Ct又是由Ct-1*f x C'得到，而C' = tan[xt, ht-1] x sigmoid[xt,ht-1] 得到。

所以ht 是由Ct和sigmoid[xt, ht-1]得出的结果。

23. 自编码器的理解和应用

自编码器（Autoencoder，AE），是一种利用反向传播算法使得输出值等于输入值的神经网络，它先将输入压缩成潜在空间表征，然后通过这种表征来重构输出。自编码器由两部分组成：

编码器：这部分能将输入压缩成潜在空间表征，可以用编码函数h=f(x)表示。

解码器：这部分能重构来自潜在空间表征的输入，可以用解码函数r=g(h)表示。

为何要用输入来重构输出？

如果自编码器的唯一目的是让输出值等于输入值，那这个算法将毫无用处。事实上，我们希望通过训练输出值等于输入值的自编码器，**让潜在表征h具有价值属性**。

整个自编码器可以用函数g(f(x)) = r来描述，其中输出r与原始输入x相近

自编码器的应用主要有两个方面：
- 第一是数据去噪
- 第二是为进行可视化而降维

设置合适的维度和稀疏约束，自编码器可以学习到比PCA等技术更有意思的数据投影。

种类：
- 香草自编码器
- 多层自编码器
- 卷积自编码器
- 正则自编码器


香草自编码器在这种自编码器的最简单结构中，只有三个网络层，即只有一个隐藏层的神经网络。它的输入和输出是相同的，可通过使用Adam优化器和均方误差损失函数，来学习如何重构输入。在这里，如果隐含层维数（64）小于输入维数（784），则称这个编码器是有损的。通过这个约束，来迫使神经网络来学习数据的压缩表征。

```python
input_size = 784
hidden_size = 64
output_size = 784

x = Input(shape=(input_size,))

#Encoder
h = Dense(hidden_size, activation='relu')(x)

#Decoder
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```

多层自编码器如果一个隐含层还不够，显然可以将自动编码器的隐含层数目进一步提高。在这里，实现中使用了3个隐含层，而不是只有一个。任意一个隐含层都可以作为特征表征，但是为了使网络对称，我们使用了最中间的网络层。
```python
input_size = 784
hidden_size = 128
code_size = 64

x = Input(shape=(input_size,))

#Encoder
hidden_1 = Dense(hidden_size, activation='relu')(x)
h = Dense(code_size, activation='relu')(hidden_1)

#Decoder
hidden_2 = Dense(hidden_size, activation='relu')(h)
r = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```

卷积自编码器你可能有个疑问，除了全连接层，自编码器应用到卷积层吗？答案是肯定的，原理是一样的，但是要使用3D矢量（如图像）而不是展平后的一维矢量。对输入图像进行下采样，以提供较小维度的潜在表征，来迫使自编码器从压缩后的数据进行学习。
```python
x = Input(shape=(28, 28,1)) 

# Encoder
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
h = MaxPooling2D((2, 2), padding='same')(conv1_3)

# Decoder
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```

正则自编码器除了施加一个比输入维度小的隐含层，一些其他方法也可用来约束自编码器重构，如正则自编码器。正则自编码器不需要使用浅层的编码器和解码器以及小的编码维数来限制模型容量，而是使用损失函数来鼓励模型学习其他特性（除了将输入复制到输出）。这些特性包括稀疏表征、小导数表征、以及对噪声或输入缺失的鲁棒性。即使模型容量大到足以学习一个无意义的恒等函数，非线性且过完备的正则自编码器仍然能够从数据中学到一些关于数据分布的有用信息。在实际应用中，常用到两种正则自编码器，分别是稀疏自编码器和降噪自编码器。稀疏自编码器：一般用来学习特征，以便用于像分类这样的任务。稀疏正则化的自编码器必须反映训练数据集的独特统计特征，而不是简单地充当恒等函数。以这种方式训练，执行附带稀疏惩罚的复现任务可以得到能学习有用特征的模型。还有一种用来约束自动编码器重构的方法，是对其损失函数施加约束。比如，可对损失函数添加一个正则化约束，这样能使自编码器学习到数据的稀疏表征。要注意，在隐含层中，我们还加入了L1正则化，作为优化阶段中损失函数的惩罚项。与香草自编码器相比，这样操作后的数据表征更为稀疏。
```python
input_size = 784
hidden_size = 64
output_size = 784

x = Input(shape=(input_size,))

# Encoder
h = Dense(hidden_size, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)

# Decoder
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```

降噪自编码器：这里不是通过对损失函数施加惩罚项，而是通过改变损失函数的重构误差项来学习一些有用信息。向训练数据加入噪声，并使自编码器学会去除这种噪声来获得没有被噪声污染过的真实输入。因此，这就迫使编码器学习提取最重要的特征并学习输入数据中更加鲁棒的表征，这也是它的泛化能力比一般编码器强的原因。这种结构可以通过梯度下降算法来训练。
```python
x = Input(shape=(28, 28, 1))

# Encoder
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
h = MaxPooling2D((2, 2), padding='same')(conv1_2)

# Decoder
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```
总结本文先介绍了自编码器的基本结构，还研究了许多不同类型的自编码器，如香草、多层、卷积和正则化，通过施加不同约束，包括缩小隐含层的维度和加入惩罚项，使每种自编码器都具有不同属性。

24. 线性回归与逻辑回归

线性回归解析解：

w* =  (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y

f(x<sub>i</sub>)=x<sub>i</sub><sup>T</sup>w* 

逻辑回归：

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191203173054.png)

参考：[LR](https://zhuanlan.zhihu.com/p/42087746)

最大熵模型：
参考：[ME](https://zhuanlan.zhihu.com/p/41990423)

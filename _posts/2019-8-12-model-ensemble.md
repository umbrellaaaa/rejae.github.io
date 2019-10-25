---
layout:     post
title:      模型融合
subtitle:   
date:       2019-8-12
author:     RJ
header-img: 
catalog: true
tags:
    - ML
---
## 前言
集成模型的效果总是优于单个模型的。<br>
今天就来好好谈谈模型融合。<br>
常用的方法包括：
1. 公平投票
2. 简单效果优先权重分配模型
3. Bagging有放回抽样训练模型
4. Boosting改变样本权重，效果优先权重分配模型
5. Stacking

<p id = "build"></p>
---

## 正文
前两个都很简单，就是将最后得到的submission结果作一个投票或者根据各个模型的acc，使用带权结果计算。后面三个着重整理一下。
<h2>Bagging</h2>
Bagging就是采用有放回的方式进行抽样，用抽样的样本建立子模型,对子模型进行训练，这个过程重复多次，最后进行融合。大概分为这样两步：<br>

1. 重复K次:  
    - 有放回地重复抽样建模
    - 训练子模型
2. 模型融合:  
    - 分类问题：voting
    - 回归问题：averag

Bagging算法不用我们自己实现，随机森林就是基于Bagging算法的一个典型例子，采用的基分类器是决策树。R和python都集成好了，直接调用。

<h2>Boosting</h2>
Bagging算法可以并行处理，而Boosting的思想是一种迭代的方法，每一次训练的时候都更加关心分类错误的样例，给这些分类错误的样例增加更大的权重，下一次迭代的目标就是能够更容易辨别出上一轮分类错误的样例。最终将这些弱分类器进行加权相加。

<h2>Stacking</h2>
Stacking 与 bagging 和 boosting 主要存在两方面的差异。

- 首先，Stacking 通常考虑的是异质弱学习器（不同的学习算法被组合在一起），而bagging 和 boosting 主要考虑的是同质弱学习器。

- 其次，stacking 学习用元模型组合基础模型，而bagging 和 boosting 则根据确定性算法组合弱学习器。

staking的python实现：


```python

# 模型融合 Stacking
将第i层的预测结果作为下一层的输入数据，具体的:<br>

return oof_train, oof_test  # 返回当前分类器对训练集和测试集的预测结果;

将你的每个分类器都调用get_oof函数，并把它们的结果合并，就得到了新的训练和测试数据new_train,new_test

new_train = np.concatenate(new_train, axis=1)

new_test = np.concatenate(new_test, axis=1)

________________________________________________________________
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

_N_FOLDS = 5  # 采用5折交叉验证
kf = KFold(n_splits=_N_FOLDS, random_state=42)  # sklearn的交叉验证模块，用于划分数据

def get_oof(clf, X_train, y_train, X_test):
    # X_train: 1000 * 10
    # y_train: 1 * 1000
    # X_test : 500 * 10
    oof_train = np.zeros((X_train.shape[0], 1))  # 1000 * 1  Stacking后训练数据的输出
    oof_test_skf = np.empty((_N_FOLDS, X_test.shape[0], 1))  # 5 * 500 * 1，oof_test_skf[i]代表第i折交叉验证产生的模型对测试集预测结果
    
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)): # 交叉验证划分此时的训练集和验证集
        kf_X_train = X_train[train_index]  # 800 * 10 训练集
        kf_y_train = y_train[train_index]  # 1 * 800 训练集对应的输出
        kf_X_val = X_train[test_index]  # 200 * 10  验证集

        clf.fit(kf_X_train, kf_y_train)  # 当前模型进行训练
        
        oof_train[test_index] = clf.predict(kf_X_val).reshape(-1, 1)  # 对当前验证集进行预测， 200 * 1  || 200    --->  200,1        
        oof_test_skf[i, :] = clf.predict(X_test).reshape(-1, 1)  # 对测试集预测 oof_test_skf[i, :] : 500 * 1

    oof_test = oof_test_skf.mean(axis=0)  # 对每一则交叉验证的结果取平均
    
    return oof_train, oof_test  # 返回当前分类器对训练集和测试集的预测结果
```

```python
# 将数据换成你的数据
X_train = np.random.random((1000, 10))  # 1000 * 10
y_train = np.random.random_integers(0, 1, (1000,))  # 1000
X_test = np.random.random((500, 10))  # 500 * 10

# 将你的每个分类器都调用get_oof函数，并把它们的结果合并，就得到了新的训练和测试数据new_train,new_test
new_train, new_test = [], []
for clf in [LinearRegression(), RandomForestRegressor()]:
    oof_train, oof_test = get_oof(clf, X_train, y_train, X_test)
    new_train.append(oof_train)
    new_test.append(oof_test)

new_train = np.concatenate(new_train, axis=1)
new_test = np.concatenate(new_test, axis=1)

# 用新的训练数据new_train作为新的模型的输入，stacking第二层
clf = RandomForestRegressor()
clf.fit(new_train, y_train)
prdict_result = clf.predict(new_test)
```
stacking代码总结，代码实现的重要流程分析：<br>
构建输出矩阵oof_train，shape为(train.shape,1)。构建K倍测试集大小的测试结果集矩阵，shape为(K,test.shape,1)<br>
运用KFold，在循环中用每次的（K-1）折训练clf, 用余下的1折进行predict,得到当前折的预测结果，重复K次得到与train训练集的所有预测结果。<br>
在每次循环都会训练一个clf，用这个clf预测测试集，共得到K个测试集，取均值作为测试集的结果。
最后，取上层的train,test预测结果，作为下一层的输入。<br>
## 后记

参考：<br>
[模型融合](https://www.zhihu.com/search?type=content&q=%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88%20Bagging)

[stacking模型融合](https://www.zhihu.com/search?type=content&q=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%20%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88)
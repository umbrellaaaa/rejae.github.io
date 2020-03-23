---
layout:     post
title:      深度学习模型部署
subtitle:   
date:       2020-3-23
author:     RJ
header-img: 
catalog: true
tags:
    - job

---
<p id = "build"></p>
---

## 前言
之前参考github上，基于Flask搭建了一个简单的NER和写诗模型的部署。没有考虑太多问题，做的也很简陋。

现在公司需要一个流式数据实时的处理英中机器翻译。现在我的模型是通过文件输入的，然后输出翻译结果文件。调用一次后，就结束了，这显然不行。

需要了解数据流相关的处理，模型那边应该随时在线，和循环等待输入类似。
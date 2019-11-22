---
layout:     post
title:      文本分类对比实验
subtitle:   
date:       2019-11-22
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 前言
采用统一的训练数据，进行对比实验才有意义，之前用的cnews数据和chinese glue提供的数据有些不同，还是按标准来吧。顺带跑跑它的各个baseline。

## 数据分析

```python
path = 'Documents/NLP_code/chineseGLUE-master/baselines/glue/chineseGLUEdatasets/thucnews/'

train_path = path+'train.txt'
dev_path = path+'dev.txt'
test_path = path+'test.txt'
files= [train_path,dev_path,test_path]

for file in files:
    with open(file,'r',encoding='utf-8') as f:
        sentences = f.readlines()
        print(file,len(sentences))

out:

train.txt 33437
dev.txt 4180
test.txt 4180


with open(file,'r',encoding='utf-8') as f:
    sentences = f.readlines()
    label_list = []
    for sentence in sentences:
        label_list.append(sentence.split('_!_')[1])
    label_set=set(label_list)
    print(label_set)

out:

{'财经', '彩票', '星座', '娱乐', '社会', '股票', '家居', '时政', '时尚', '体育', '科技', '房产', '游戏', '教育'}

一共是14个类别

```

## albert_tiny
```
***** Eval results /home/rejae/Documents/NLP_code/chineseGLUE-master/baselines/models/albert/thucnews_output/model.ckpt-0 *****
eval_accuracy = 0.113663554
eval_loss = 2.6364124
global_step = 0
loss = 2.6364102
***** Eval results /home/rejae/Documents/NLP_code/chineseGLUE-master/baselines/models/albert/thucnews_output/model.ckpt-1000 *****
eval_accuracy = 0.87724334
eval_loss = 0.50574416
global_step = 1000
loss = 0.505266
***** Eval results /home/rejae/Documents/NLP_code/chineseGLUE-master/baselines/models/albert/thucnews_output/model.ckpt-2000 *****
eval_accuracy = 0.8973439
eval_loss = 0.38616368
global_step = 2000
loss = 0.38576332
***** Eval results /home/rejae/Documents/NLP_code/chineseGLUE-master/baselines/models/albert/thucnews_output/model.ckpt-3000 *****
eval_accuracy = 0.90141183
eval_loss = 0.35735667
global_step = 3000
loss = 0.35697806
***** Eval results /home/rejae/Documents/NLP_code/chineseGLUE-master/baselines/models/albert/thucnews_output/model.ckpt-3134 *****
eval_accuracy = 0.90141183
eval_loss = 0.3567447
global_step = 3134
loss = 0.35636652

```
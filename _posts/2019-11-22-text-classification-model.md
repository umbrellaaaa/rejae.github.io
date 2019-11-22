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
采用统一的训练数据，进行对比实验更有意义，之前用的cnews数据和chinese glue提供的数据有些不同，还是按标准来吧。顺带跑跑它的各个baseline。

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

每个类别的数量：

from collections import Counter
label_dict = dict(Counter(labels))
label_dict

from collections import Counter
label_dict = dict(Counter(labels))
label_dict
1
from collections import Counter
2
label_dict = dict(Counter(labels))
3
label_dict
{'体育': 5248,
 '股票': 6183,
 '社会': 2041,
 '时政': 2567,
 '娱乐': 3723,
 '科技': 6478,
 '财经': 1487,
 '游戏': 978,
 '彩票': 290,
 '家居': 1305,
 '教育': 1637,
 '房产': 814,
 '时尚': 541,
 '星座': 144}

 

```
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191123dataanalyze.jpg)

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

## CNN 
```
Testing...
Test Loss:   0.41, Test Acc:  91.03%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          财经       0.81      0.82      0.81       187
          彩票       0.89      0.72      0.79        43
          星座       0.90      0.69      0.78        13
          娱乐       0.94      0.91      0.92       447
          社会       0.79      0.87      0.83       247
          股票       0.94      0.91      0.92       784
          家居       0.87      0.90      0.89       146
          时政       0.83      0.88      0.86       294
          时尚       0.88      0.88      0.88        56
          体育       0.95      0.99      0.97       688
          科技       0.93      0.91      0.92       832
          房产       1.00      0.97      0.98        97
          游戏       0.86      0.86      0.86       113
          教育       0.94      0.88      0.91       233

    accuracy                           0.91      4180
   macro avg       0.89      0.87      0.88      4180
weighted avg       0.91      0.91      0.91      4180

```

## RNN  use_pretrain = False
best train:<br>
Iter:   2300, Train Loss:   0.11, Train Acc:  96.09%, Val Loss:   0.49, Val Acc:  87.78%, Time: 0:29:23 *
test:<br>
```
Testing...
Test Loss:   0.47, Test Acc:  88.52%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          财经       0.84      0.68      0.75       187
          彩票       0.94      0.77      0.85        43
          星座       0.86      0.46      0.60        13
          娱乐       0.92      0.91      0.91       447
          社会       0.74      0.85      0.79       247
          股票       0.89      0.93      0.91       784
          家居       0.83      0.82      0.82       146
          时政       0.85      0.80      0.82       294
          时尚       0.66      0.80      0.73        56
          体育       0.95      0.98      0.97       688
          科技       0.89      0.89      0.89       832
          房产       0.97      0.95      0.96        97
          游戏       0.76      0.81      0.79       113
          教育       0.95      0.82      0.88       233

    accuracy                           0.89      4180
   macro avg       0.86      0.82      0.83      4180
weighted avg       0.89      0.89      0.88      4180

Confusion Matrix...
[[127   0   0   3   1  46   2   1   1   0   4   1   0   1]
 [  0  33   0   0   0   0   0   0   0   9   1   0   0   0]
 [  0   0   6   1   0   0   0   0   2   0   1   1   1   1]
 [  1   0   0 405  10   2   1   1   7   4  13   0   2   1]
 [  1   0   1   9 210   1   2   6   2   4   6   1   1   3]
 [ 21   0   0   0   1 731   1  13   0   0  17   0   0   0]
 [  0   0   0   2   2   3 120   2   9   1   6   0   1   0]
 [  0   0   0   3  20  11   3 234   1   7  13   0   0   2]
 [  0   0   0   3   3   0   4   0  45   0   1   0   0   0]
 [  0   1   0   5   5   0   0   2   0 672   2   0   0   1]
 [  1   1   0  10  12  19   9  10   1   5 741   0  22   1]
 [  0   0   0   0   0   2   1   0   0   1   0  92   1   0]
 [  0   0   0   1   1   2   1   0   0   0  16   0  92   0]
 [  1   0   0   0  18   5   1   7   0   1   7   0   1 192]]
Time usage: 0:00:22

```

## RNN  use_pretrain = True
best train:<br>
Iter:   1900, Train Loss:   0.37, Train Acc:  88.28%, Val Loss:   0.44, Val Acc:  89.90%, Time: 0:24:33 *
```
Testing...
Test Loss:   0.41, Test Acc:  89.57%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          财经       0.82      0.78      0.80       187
          彩票       0.79      0.86      0.82        43
          星座       0.45      0.69      0.55        13
          娱乐       0.91      0.91      0.91       447
          社会       0.79      0.86      0.83       247
          股票       0.90      0.93      0.92       784
          家居       0.86      0.82      0.84       146
          时政       0.91      0.79      0.85       294
          时尚       0.69      0.86      0.76        56
          体育       0.95      0.97      0.96       688
          科技       0.91      0.92      0.91       832
          房产       0.99      0.95      0.97        97
          游戏       0.83      0.76      0.80       113
          教育       0.92      0.82      0.86       233

    accuracy                           0.90      4180
   macro avg       0.84      0.85      0.84      4180
weighted avg       0.90      0.90      0.90      4180

Confusion Matrix...
[[146   0   1   2   2  30   2   0   0   1   3   0   0   0]
 [  0  37   0   0   0   0   0   0   0   5   0   1   0   0]
 [  0   0   9   0   0   0   1   0   2   0   1   0   0   0]
 [  1   0   0 408   9   3   3   2   4   6   8   0   2   1]
 [  1   1   1   9 213   4   0   4   1   3   4   0   1   5]
 [ 21   0   0   0   0 731   4   8   1   3  16   0   0   0]
 [  2   1   3   4   0   2 120   0   8   0   3   0   1   2]
 [  2   0   0   5  15  19   2 233   0   6   5   0   1   6]
 [  0   0   0   0   3   0   2   0  48   1   1   0   1   0]
 [  0   7   0   6   3   0   0   1   2 668   1   0   0   0]
 [  2   1   0  10  13  15   3   5   2   6 763   0  10   2]
 [  1   0   0   0   0   0   1   1   0   1   0  92   1   0]
 [  0   0   2   0   0   1   0   0   1   0  22   0  86   1]
 [  1   0   4   3  10   5   1   3   1   1  14   0   0 190]]
Time usage: 0:00:22

```

## Bilstm  use_pretrain = True
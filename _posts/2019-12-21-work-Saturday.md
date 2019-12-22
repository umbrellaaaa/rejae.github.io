---
layout:     post
title:      work saturdat 
subtitle:   
date:       2019-12-21
author:     RJ
header-img: 
catalog: true
tags:
    - Learning
---
<p id = "build"></p>
---

<h1> test transformer</h1>

调试代码，从最初修改源代码中以train代替test-----到使用test数据测试训练

## error1：
```
 the  0 th example.
'炳' is not in list
```
说明test数据中的词，超出了训练数据的vocab.

除此之外，还遇到错误，这是由于测试的时候是一条一条的数据，所以只需要将x  reshape(-1,1)即可：
```python
Cannot feed value of shape (30,) for Tensor 'Placeholder:0', which has shape '(?, ?)'

Cannot feed value of shape (23,) for Tensor 'Placeholder:0', which has shape '(?, ?)'

```
这是输入数据的padding问题。查看训练过程中的padding过程，发现只需要每一个批次padding到相同的长度，这与传统的padding所有批次为相同的长度不同。

## error2
类似的，测试集pny_list中的拼音也有不在训练集中的，所以导致出现以下错误，即打印的为上一次的结果：
```
 the  5 th example.
[356, 2269, 1972, 709, 169, 50, 335, 262, 164, 798, 520, 440, 262, 50, 165, 39, 412, 486, 27, 735, 93, 232, 412, 262, 304, 107, 739, 1207, 1414, 321, 1891, 1847]
原文汉字： 吉隆坡郊外一座十二层公寓十一日下午突然倒塌将五十多名房客困在里面
识别结果： 集哑竹元外一做时二层工育时一日下五突然导她迎五时多明防客困响仰壮

 the  6 th example.
'nei3' is not in list
[356, 2269, 1972, 709, 169, 50, 335, 262, 164, 798, 520, 440, 262, 50, 165, 39, 412, 486, 27, 735, 93, 232, 412, 262, 304, 107, 739, 1207, 1414, 321, 1891, 1847]
原文汉字： 选一本好书使你罹小恙而顿愈处逆境而不馁昌如夏花春草盛若锦缎烈火
识别结果： 集哑竹元外一做时二层工育时一日下五突然导她迎五时多明防客困响仰壮
```
所以源代码在构建词表的时候就出现了问题，应该用defaultdict(int)构建，当遇到未收录的词的时候，直接返回默认值。


具体的，在不改动源代码构建的list类型词表，用index取id得情况下，将list转换成defaultdict(int)，当遇到OOV默认返回id=0，即0对应的[PAD]。

```python
vocab_dict_w2id = defaultdict(int)
for index, item in enumerate(train_data.han_vocab):
    vocab_dict_w2id[item] = index

vocab_dict_id2w = {v: k for k, v in vocab_dict_w2id.items()}

pny_dict_w2id = defaultdict(int)
for index, item in enumerate(train_data.pny_vocab):
    pny_dict_w2id[item] = index

pny_dict_id2w = {v: k for k, v in pny_dict_w2id.items()}

```

## 度量准则
get_opcodes
```python

def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


>>> a = "qabxcd"
>>> b = "abycdf"
>>> s = SequenceMatcher(None, a, b)
>>> for tag, i1, i2, j1, j2 in s.get_opcodes():
...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
    delete a[0:1] (q) b[0:0] ()
    equal a[1:3] (ab) b[0:2] (ab)
    replace a[3:4] (x) b[2:3] (y)
    equal a[4:6] (cd) b[3:5] (cd)
    insert a[6:6] () b[5:6] (f)
```

## 第一次调试错误率：
```
...
 the  2494 th example.
[463, 551, 1929, 435, 968, 463, 551, 158, 1172, 630, 135, 2012, 594, 691, 69, 109, 109, 135, 304, 356, 1272, 324, 1568, 50, 647, 616, 199, 1349, 153, 2, 305, 302, 691, 1026, 1301]
原文汉字： 国务委员兼国务院秘书长罗干民政部部长多吉才让也一同前往延安看望人民群众
识别结果： 国物震员间国物院密书长螺干民正不不长多集材让野一同前往严安是欲人民群急
词错误率： 0.5136740654925498
```
原因分析：

由于数据集太小，导致缺词，缺拼音，OOV严重。
```python
make lm pinyin vocab...
100%|██████████| 10000/10000 [00:01<00:00, 9259.61it/s]
100%|██████████| 10000/10000 [00:01<00:00, 9275.29it/s]
pny_vocab: 1042

make lm hanzi vocab...
100%|██████████| 2495/2495 [00:00<00:00, 10509.62it/s]
 84%|████████▍ | 2096/2495 [00:00<00:00, 5242.64it/s]han_vocab: 1793
100%|██████████| 2495/2495 [00:00<00:00, 5173.93it/s]
```

## 尝试增加epoch从10-->>20

 由于代码保存最后一个epoch继续训练，所以相当于是epoch=30, 查看epoch影响如何。

 ```
epochs 1 : average loss =  1.6575595687389373
epochs 2 : average loss =  1.3558342099189757
epochs 3 : average loss =  1.3236024334430694
epochs 4 : average loss =  1.305806235599518
epochs 5 : average loss =  1.2921207308292388
epochs 6 : average loss =  1.2788871832847595
epochs 7 : average loss =  1.2661192213058472
epochs 8 : average loss =  1.2548233767986297
epochs 9 : average loss =  1.245651912212372
epochs 10 : average loss =  1.2382040247917174
 ```
 ```
epochs 1 : average loss =  1.2326828608036042
epochs 2 : average loss =  1.2288068061828614
epochs 3 : average loss =  1.2250357716560363
epochs 4 : average loss =  1.2221878454208375
epochs 5 : average loss =  1.2201760899066925
epochs 6 : average loss =  1.218393822813034
epochs 7 : average loss =  1.2171159165382386
epochs 8 : average loss =  1.2154317249298097
epochs 9 : average loss =  1.214419894504547
epochs 10 : average loss =  1.2138375066757203
epochs 11 : average loss =  1.2128349842071533
epochs 12 : average loss =  1.2121970566272735
epochs 13 : average loss =  1.211873703956604
epochs 14 : average loss =  1.2114190195083618
epochs 15 : average loss =  1.2108791604042053
epochs 16 : average loss =  1.2106108968734741
epochs 17 : average loss =  1.2100825407028197
epochs 18 : average loss =  1.209688826084137
epochs 19 : average loss =  1.2092793963432311
epochs 20 : average loss =  1.2092923361778258
 ```

得到结果：
```
 the  2494 th example.
[463, 551, 1929, 2509, 968, 463, 551, 1948, 1172, 630, 135, 191, 594, 691, 710, 109, 109, 135, 304, 1956, 1272, 324, 293, 434, 1129, 616, 199, 1537, 153, 139, 2351, 1385, 691, 1026, 243]
原文汉字： 国务委员兼国务院秘书长罗干民政部部长多吉才让也一同前往延安看望人民群众
识别结果： 国物震番间国物慰密书长要干民政不不长多隆材让也医志前往桥安家妃沫民群重
词错误率： 0.601769802437792
```

可知模型可能是过拟合了。所以后序要注意的一点是控制过拟合。

方式包括：
- 增大数据量
- earlystop
- decay_learning_rate
- dropout  
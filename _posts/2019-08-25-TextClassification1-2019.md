---
layout:     post
title:      基于LR模型的文本情感分类
subtitle:    cheer up
date:       2019-08-25
author:     RJ
header-img: img/banboo.jpg
catalog: true
tags:
    - NLP
---

> “ zzz”



## 前言

正式开始文本情感分析项目

<p id = "build"></p>
---

## 正文
准备开始情感分析的项目：
<br>https://www.datafountain.cn/competitions/350/datasets
<br>昨天查了一下基于情感词典的方法，感觉比较杂乱，情感词典适用于具体的领域，像外卖、酒店、影评，而对于这次比赛的互联网新闻情感分析就比较棘手了，没有找到合适的情感词典，而且思考了一下，这也不是未来主流的方向，所以放弃了基于情感词典的方法。
<br>
下载了比赛的数据，进行了简单的分析：<br>
1.分析id是否有误，但应该不会影响id匹配问题

2.分析title内容，将title中的nan内容替换为空串

3.分析content内容，将所有非汉字字符都过滤掉

##基于LR的新闻情感分析：
```python

import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold
import re
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('./train/Train_Dataset.csv')
train_label_df = pd.read_csv('./train/Train_Dataset_Label.csv')
train_df=pd.merge(train_df,train_label_df,on=['id'],copy=False)

test_df = pd.read_csv('./test/Test_Dataset.csv')
train_test_df = pd.concat([train_df,test_df],ignore_index=True)

train_test_df['cut_text'] = train_test_df['content'].apply(lambda x: ' '.join(jieba.cut(str(x))))

train_shape = train_df.shape

pattern=re.compile('[^\u4e00-\u9fa5]+')
train_test_df['cut_text'].apply(lambda x: ''.join(re.sub(pattern, ' ', x, count=0, flags=0)))

tf = TfidfVectorizer() 
tf_feat = tf.fit_transform(train_test_df['cut_text'].values)  #tf_feat = tf_feat.tocsr()

tf_feat.shape

X = tf_feat[:train_shape[0]]
y = train_test_df['label'][:train_shape[0]]

sub = tf_feat[train_shape[0]:]
N=10
kf = StratifiedKFold(n_splits=N,random_state=42,shuffle=True)
oof = np.zeros((X.shape[0],3))
oof_sub = np.zeros((sub.shape[0],3))

for j,(train_in,test_in) in enumerate(kf.split(X,y)):
    print('running',j)
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
    clf = LogisticRegression(C=100) #正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
    clf.fit(X_train,y_train)
    test_y = clf.predict_proba(X_test)
    oof[test_in] = test_y
    oof_sub = oof_sub + clf.predict_proba(sub)

xx_cv = f1_score(y,np.argmax(oof,axis=1),average='macro')
print(xx_cv)

result = pd.DataFrame()
result['id'] = test_df['id']
result['label'] = np.argmax(oof_sub,axis=1)
print('finish')

result[['id','label']].to_csv('./baselien_lr_tfidf_{}.csv'.format(str(np.mean(xx_cv)).split('.')[1]),index=False) 

```
最后得到了0.6331582088710596的F1 score，作为一个baseline，还是可以了。

##参考内容

tfidf计算：
```python
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
 
if __name__ == "__main__":
    corpus=["我 来到 北京 清华大学","他 来到 了 网易 杭研 大厦",
"小明 硕士 毕业 与 中国 科学院","我 爱 北京 天安门"]
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print("-------这里输出第",i,u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j],weight[i][j]) 


```

## 实验
使用清华cnews数据集，

```python
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
import re
import warnings
import os

warnings.filterwarnings('ignore')

base_path = '/data/cnews/'
train_path = base_path + 'cnews.train.txt'
val_path = base_path + 'cnews.val.txt'
test_path = base_path + 'cnews.test.txt'


def get_cut_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        m_list = f.readlines()
        train_data = []
        labels = []
        for idx, item in enumerate(m_list):
            label, data = item.split('\t')
            labels.append(label)
            sentence = ' '.join(jieba.cut(str(item)))
            train_data.append(sentence.lstrip())
    label_id = get_label_id(labels)
    return train_data, label_id


# 处理标签多分类
def get_label_id(label):
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    label_id = [cat_to_id[item_label] for item_label in label]
    return label_id


# 处理文本特征 train/valid/test三个数据集
train_data, train_label_id = get_cut_text(train_path)

val_data, val_label_id = get_cut_text(val_path)
test_data, test_label_id = get_cut_text(test_path)

train_len = len(train_data)
test_len = len(test_data)
tf = TfidfVectorizer(ngram_range=(1,4),analyzer='char')
tf_feat = tf.fit_transform(train_data+test_data)  #  train test 全部传入

# val_tf_feat = tf.fit_transform(val_data)

X = tf_feat[:train_len]
y = train_label_id
y = np.array(y)
N = 5
kf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
oof = np.zeros((X.shape[0], 10))
oof_sub = np.zeros((len(test_data), 10))

for j, (train_in, test_in) in enumerate(kf.split(X, y)):
    print('running', j)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
    clf = LogisticRegression(C=1.0)  # 正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
    clf.fit(X_train, y_train)
    test_y = clf.predict_proba(X_test)
    oof[test_in] = test_y
    test_data_predict = clf.predict_proba(tf_feat[train_len:])
    oof_sub = oof_sub + test_data_predict


xx_cv = f1_score(y, np.argmax(oof, axis=1), average='macro')
print(xx_cv)
count = 0
for idx, item in enumerate(test_label_id):
    if np.argmax(oof_sub, axis=1)[idx] == item:
        count=count+1
        
acc = count/len(test_label_id)
acc
```
 
 实验结果：
 ```
train: f1 = 0.9643000758823714
test: acc : 0.9428
```
## 后记

传统的机器学习模型具有一定的参考价值，作为一个baseline,可以对比深度学习模型的效果。







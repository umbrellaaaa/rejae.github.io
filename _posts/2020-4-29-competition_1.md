---
layout:     post
title:      招商银行信用违约比赛
subtitle:   
date:       2020-4-29
author:     RJ
header-img: 
catalog: true
tags:
    - nlp

---
<p id = "build"></p>
---

## 前言
之前有参加过比赛，但是由于基础不扎实或者比赛需要服务器，所以心不在焉，没有好好去争取。

现在是一个好机会，而且比赛可以直通招商银行的面试，那就尽力打比赛吧。


## 流程跑通基础代码

开始没有做什么其他操作，通过describe取出数值列，归一化都没有做就直接丢LR模型，跑出来有0.95577的分数

听说的是有某几个特征放出来，所以难度降低了，然后比赛还要更新数据，再说吧。

流程跑通就要思考有哪些数据处理，特征生成的方式，怎样提升模型效果，以及后面的模型融合。
```python

# EDA

一、赛题背景
在当今大数据时代，信用评分不仅仅用在办理信用卡、贷款等金融场景，类似的评分产品已经触及到我们生活的方方面面，比如借充电宝免押金、打车先用后付等，甚至在招聘、婚恋场景都有一席之地。
招行作为金融科技的先行者，APP月活用户数上亿，APP服务不仅涵盖资金交易、理财、信贷等金融场景，也延伸到饭票、影票、出行、资讯等非金融场景，可以构建用户的信用评分，基于信用评分为用户提供更优质便捷的服务。

二、课题研究要求本次大赛为参赛选手提供了两个数据集（训练数据集和评分数据集），包含用户标签数据、过去60天的交易行为数据、过去30天的APP行为数据。希望参赛选手基于训练数据集，通过有效的特征提取，构建信用违约预测模型，并将模型应用在评分数据集上，输出评分数据集中每个用户的违约概率。

三、评价指标
![](999995968_1587116931158_96DC3559A0144585FCA6C40301691517.png)

其中D^+与D^-分别为评分数据集中发生违约与未发生违约用户集合，|D^+ |与|D^- |为集合中的用户量，f(x)为参赛者对于评分数据集中用户发生违约的概率估计值，I为逻辑函数。

四、数据说明 
- 1.训练数据集_tag.csv，评分数据集_tag.csv提供了训练数据集和评分数据集的用户标签数据；
- 2.训练数据集_trd.csv，评分数据集_trd.csv提供了训练数据集和评分数据集的用户60天交易行为数据；
- 3.训练数据集_beh.csv，评分数据集_ beh.csv提供了训练数据集和评分数据集的用户30天APP行为数据；
- 4.数据说明.xlsx为数据集字段说明和数据示例；
- 5.提交样例：
    - 5.1采⽤UTF-8⽆BOM编码的txt⽂件提交，⼀共提交⼀份txt⽂件。 
    - 5.2输出评分数据集中每个用户违约的预测概率，输出字段为：用户标识和违约预测概率，用\t分割，每个用户的预测结果为一行，所有数据按用户id从小到大排序，注意不能有遗漏的数据或多出的数据。

## 训练数据
- 训练数据集_beh.csv
- 训练数据集_trd.csv
- 训练数据集_tag.csv

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# read data
train_tag = pd.read_csv('../data/训练数据集_tag.csv')
train_beh = pd.read_csv('../data/训练数据集_beh.csv')
train_trd = pd.read_csv('../data/训练数据集_trd.csv')

test_tag = pd.read_csv('../data/评分数据集_tag.csv')
test_beh = pd.read_csv('../data/评分数据集_beh.csv')
test_trd = pd.read_csv('../data/评分数据集_trd.csv')


# count wy's people numbers
def count_wy(df):
    df = df.drop_duplicates(subset=['id'], keep='first')
    wy_ids = df[df['flag'] == 1]
    print('people num:', len(df))
    print('wy people num:', len(wy_ids))
    print('比例：', len(wy_ids) / len(df), '\n')
    return wy_ids['id']


#  process missing value and process object data to int data
class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                # .astype(str)
                output[col] = LabelEncoder().fit_transform(output[col].astype(str))
        else:
            for colname, col in output.iteritems():
                output[str(colname)] = LabelEncoder().fit_transform(str(col))
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


#  first process beh table
def process_beh_table(df, ids):
    data_df = DataFrame(columns=(
        ['id', 'JJD', 'MSG', 'AAO', 'XAI', 'CQC', 'CQA', 'LCT', 'ZY1', 'EGB', 'BWA', 'SZA', 'MTA', 'JF2', 'JJK', 'TRN',
         'CQB', 'EGA', 'SZD', 'SYK', 'CQE', 'FDA', 'LC0', 'FTR', 'FLS', 'XAG', 'GBA', 'CTR', 'CQD', 'BWE']))

    for m_id in tqdm(ids):

        row_dict = {'id': m_id, 'JJD': 0, 'MSG': 0, 'AAO': 0, 'XAI': 0, 'CQC': 0, 'CQA': 0, 'LCT': 0, 'ZY1': 0,
                    'EGB': 0, 'BWA': 0,
                    'SZA': 0, 'MTA': 0, 'JF2': 0, 'JJK': 0, 'TRN': 0, 'CQB': 0,
                    'EGA': 0, 'SZD': 0, 'SYK': 0, 'CQE': 0, 'FDA': 0, 'LC0': 0, 'FTR': 0, 'FLS': 0, 'XAG': 0, 'GBA': 0,
                    'CTR': 0, 'CQD': 0, 'BWE': 0}

        for i in df[df['id'] == m_id]['page_no'].values:
            row_dict[i] += 1
        data_df = data_df.append(row_dict, ignore_index=True)

    return data_df


#  the oricess trd table
def process_trd_table(tag_beh_table, train_trd, ids):
    values = [0] * len(ids)
    BA_trd_dict = dict(zip(ids, values))
    BB_trd_dict = dict(zip(ids, values))
    BC_trd_dict = dict(zip(ids, values))
    CA_trd_dict = dict(zip(ids, values))
    CB_trd_dict = dict(zip(ids, values))
    CC_trd_dict = dict(zip(ids, values))

    for indexs in tqdm(train_trd.index):
        if train_trd.loc[indexs].values[2] == 'B':
            if train_trd.loc[indexs].values[3] == 'A':
                #             BA_trd_dict[train_trd.loc[indexs].values[0]] += train_trd.loc[indexs].values[-1]
                BA_trd_dict[train_trd.loc[indexs].values[0]] += 1
            elif train_trd.loc[indexs].values[3] == 'B':
                #             BB_trd_dict[train_trd.loc[indexs].values[0]] += train_trd.loc[indexs].values[-1]
                BB_trd_dict[train_trd.loc[indexs].values[0]] += 1
            else:
                #             BC_trd_dict[train_trd.loc[indexs].values[0]] += train_trd.loc[indexs].values[-1]
                BC_trd_dict[train_trd.loc[indexs].values[0]] += 1
        else:
            if train_trd.loc[indexs].values[3] == 'A':
                #             CA_trd_dict[train_trd.loc[indexs].values[0]] += train_trd.loc[indexs].values[-1]
                CA_trd_dict[train_trd.loc[indexs].values[0]] += 1
            elif train_trd.loc[indexs].values[3] == 'B':
                #             CB_trd_dict[train_trd.loc[indexs].values[0]] += train_trd.loc[indexs].values[-1]
                CB_trd_dict[train_trd.loc[indexs].values[0]] += 1
            else:
                #             CC_trd_dict[train_trd.loc[indexs].values[0]] += train_trd.loc[indexs].values[-1]
                CC_trd_dict[train_trd.loc[indexs].values[0]] += 1

    tag_beh_table['BA_trd'] = list(BA_trd_dict.values())
    tag_beh_table['BB_trd'] = list(BB_trd_dict.values())
    tag_beh_table['BC_trd'] = list(BC_trd_dict.values())
    tag_beh_table['CA_trd'] = list(CA_trd_dict.values())
    tag_beh_table['CB_trd'] = list(CB_trd_dict.values())
    tag_beh_table['CC_trd'] = list(CC_trd_dict.values())
    return tag_beh_table


print('len train_tag_df', len(train_tag['id']))
train_tag_df_wy_ids = count_wy(train_tag)
print('len train_beh_df', len(set(train_beh['id'])))
train_beh_df_wy_ids = count_wy(train_beh)
print('len train_trd_df', len(set(train_trd['id'])))
train_trd_df_wy_ids = count_wy(train_trd)

train_tag_ids = list(train_tag.iloc[:, 0].values)

test_tag_ids = list(test_tag.iloc[:, 0].values)
print('test_tag_ids length', len(test_tag_ids))
test_beh_ids = set(test_beh['id'])
print('test_beh_ids length', len(test_beh_ids))
test_trd_ids = set(test_trd['id'])
print('test_trd_ids length', len(test_trd_ids))

#  先处理trd，然后dummy，最后merge beh表
beh_table = process_beh_table(train_beh, set(train_beh['id']))
tag_trd_table = process_trd_table(train_tag, train_trd, train_tag_ids)

test_beh_table = process_beh_table(test_beh, set(test_beh['id']))
test_tag_trd_table = process_trd_table(test_tag, test_trd, test_tag_ids)

#  MultiColumnLabelEncoder(fruit_data.columns.tolist()).fit_transform(fruit_data)
train_id = tag_trd_table['id']
train_flag = tag_trd_table['flag']
tag_trd_table = tag_trd_table.drop(['id', 'flag'], axis=1)
num_tag_trd_table = MultiColumnLabelEncoder(tag_trd_table).fit_transform(tag_trd_table)

test_id = test_tag_trd_table['id']
test_tag_trd_table = test_tag_trd_table.drop(['id'], axis=1)
test_num_tag_trd_table = MultiColumnLabelEncoder(test_tag_trd_table).fit_transform(test_tag_trd_table)


#  process low catagory feature to onehot
def get_final_train_data(num_final_table):
    dummy_feature = []
    for col in num_final_table.columns:
        # print(col,':',set(num_df[col].values))
        if len(set(num_final_table[col].values)) < 20:
            dummy_feature.append(col)
        # print(col,':',set(train[col].values),'\n')

    for item in dummy_feature:
        num_final_table[item] = num_final_table[item].apply(str)

    dummy_df = pd.get_dummies(num_final_table)

    return dummy_df


dummy_df = get_final_train_data(num_tag_trd_table)
dummy_df['id'] = train_id
train_data = pd.merge(dummy_df, beh_table, on="id", how="left")
train_data['flag'] = train_flag
train_data.to_csv('../data/train.csv')

test_dummy_df = get_final_train_data(test_num_tag_trd_table)
test_dummy_df['id'] = test_id
test_data = pd.merge(dummy_df, test_beh_table, on="id", how="left")
test_data.to_csv('../data/test.csv')

```



## 多表融合
注意到tag和beh表能很好的merge，但是trd表，有很多交易记录，而每个人的交易笔数又全相同，所以没办法构成一个列数相同的DataFrame

```python
Dat_Flg1_Cd	交易方向	"收支交易方向， B：支出；C：收入"	
Dat_Flg3_Cd	支付方式		
Trx_Cod1_Cd	收支一级分类代码		
Trx_Cod2_Cd	收支二级分类代码		
trx_tm	交易时间		
cny_trx_amt	交易金额		折人民币交易金额
```
注意到Dat_Flg1_Cd，Dat_Flg3_Cd，Trx_Cod1_Cd，Trx_Cod2_Cd 四个特征比较固定，变的是交易时间和交易金额，支付方向和收支可以通过组合特征统计其频次起到压缩的效果。


注意上面的tag先融合了trd之后再做的dummy，这样不太好，后面得做修改。









## competition2 TX 广告计算
```
本届算法大赛的题目来源于一个重要且有趣的问题。众所周知，像用户年龄和性别这样的人
口统计学特征是各类推荐系统的重要输入特征，其中自然也包括了广告平台。这背后的假设
是，用户对广告的偏好会随着其年龄和性别的不同而有所区别。许多行业的实践者已经多次
验证了这一假设。然而，大多数验证所采用的方式都是以人口统计学属性作为输入来产生推
荐结果，然后离线或者在线地对比用与不用这些输入的情况下的推荐性能。本届大赛的题目
尝试从另一个方向来验证这个假设，即以用户在广告系统中的交互行为作为输入来预测用户
的人口统计学属性。

我们认为这一赛题的“逆向思考”本身具有其研究价值和趣味性，此外也有实用价值和挑战
性。例如，对于缺乏用户信息的实践者来说，基于其自有系统的数据来推断用户属性，可以
帮助其在更广的人群上实现智能定向或者受众保护。与此同时，参赛者需要综合运用机器学
习领域的各种技术来实现更准确的预估。

具体而言，在比赛期间，我们将为参赛者提供一组用户在长度为 91 天（3 个月）的时间窗
口内的广告点击历史记录作为训练数据集。每条记录中包含了日期（从 1 到 91）、用户信息
（年龄，性别），被点击的广告的信息（素材 id、广告 id、产品 id、产品类目 id、广告主
id、广告主行业 id 等），以及该用户当天点击该广告的次数。测试数据集将会是另一组用户
的广告点击历史记录。提供给参赛者的测试数据集中不会包含这些用户的年龄和性别信息。
本赛题要求参赛者预测测试数据集中出现的用户的年龄和性别，并以约定的格式提交预测结
果。大赛官网后台将会自动计算得分并排名。详情参见【评估方式】和【提交方式】。
```


## lightGBM
```python
data = np.random.rand(500, 10)  # 500 个样本, 每一个包含 10 个特征
label = np.random.randint(2, size=500)  # 二元目标变量,  0 和 1
train_data = lgb.Dataset(data, label=label)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size=0.2)
xgtrain = lgb.Dataset(X_train, y_train)
xgvalid = lgb.Dataset(X_val, y_val)

train_data = lgb.Dataset('train.svm.txt')
train_data.save_binary('train.bin')

w = np.random.rand(500, )
train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], 
                   categorical_feature=['c3'],weight=w)

train_data = lgb.Dataset(data, label=label)
w = np.random.rand(500, )
train_data.set_weight(w)

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary', #xentlambda
    'metric': 'auc',
    'silent':0,
    'learning_rate': 0.05,
    'is_unbalance': 'true',  #当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
    'num_leaves': 64,  # 一般设为少于2^(max_depth)
    'max_depth': -1,  #最大的树深，设为-1时表示不限制树的深度
    'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
    'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    #'scale_pos_weight':100,
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 2.99,  # L1 regularization term on weights
    'reg_lambda': 1.9,  # L2 regularization term on weights
    'nthread': 10,
    'verbose': 0,
}


def feval_spec(preds, train_data):
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(train_data.get_label(), preds)
    tpr0001 = tpr[fpr <= 0.0005].max()
    tpr001 = tpr[fpr <= 0.001].max()
    tpr005 = tpr[fpr <= 0.005].max()
    #tpr01 = tpr[fpr.values <= 0.01].max()
    tprcal = 0.4 * tpr0001 + 0.3 * tpr001 + 0.3 * tpr005
    return 'spec_cal',tprcal,True



num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

num_round = 10
lgb.cv(param, train_data, num_round, nfold=5)

bst = lgb.train(param, train_data, num_round, valid_sets=valid_sets, 
      early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)


data = np.random.rand(7, 10)
ypred = bst.predict(data)

ypred = bst.predict(data, num_iteration=bst.best_iteration)
```




## word2vec

将每个模块（序号）看成一个词，一个用户的所有操作就成了一篇文档

将用户的点击历史按时间顺序排序后，成为一个sentence，运用word2vec。

可解释性：

- 用户的行为具有一定规律性
- 文字的表达具有一定规律性

```python
creation = pd.read_csv('creative_id_df.csv')
creation.head()

sentences=[]
for item in creation['sentence']:
    #sent = ''.join(item[1:-1].split(','))
    sent = list(item[1:-1].replace("'",'').split(','))
    sentences.append(sent)


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

import multiprocessing
from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=1,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.corpus_count

t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=5, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


w2v_model.init_sims(replace=True)

w2v_model.wv.most_similar(positive=["1223"])
```

model.wv.save_word2vec_format('model.bin', binary=True)


现在我们有了序号对应的向量就要使用深度学习模型来训练和预测了。

### 加载词向量
[word2vec into tensorlfow](https://stackoverflow.com/questions/53353978/how-to-project-my-word2vec-model-in-tensorflow)
[参考](https://www.xuqingtang.top/2019/05/15/%E5%9C%A8Keras%E7%9A%84Embedding%E5%B1%82%E4%B8%AD%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9A%84word2vec%E8%AF%8D%E5%90%91%E9%87%8F/)
[keras word2vec](https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/)
[keras w2v kaggle demo](https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings)
[stackoverflow w2v ](https://stackoverflow.com/questions/53353978/how-to-project-my-word2vec-model-in-tensorflow)
```python
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import gensim
import multiprocessing
from gensim.models import Word2Vec

## 1 导入 预训练的词向量
Word2VecModel = gensim.models.Word2Vec.load('w2v.model') # 读取词向量

Word2VecModel.wv['260644']  # 词语的向量，是numpy格式
```








## 参考

[pandas 速查](https://zhuanlan.zhihu.com/p/29665562)
[3Top/word2vec-api](https://github.com/3Top/word2vec-api)
[kaggle gensim-word2vec-tutorial](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial)
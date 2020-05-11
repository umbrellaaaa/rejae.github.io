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

### 1. beh_df 共 4个feature

import pandas as pd
beh_df = pd.read_csv('训练数据集_beh.csv')
print(list(beh_df.columns))

### 3. tag 共 43 个Feature

import pandas as pd
tag_df = pd.read_csv('训练数据集_tag.csv')
feature_list = ['id', 'gdr_cd', 'age', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'job_year', 'ic_ind', 'fr_or_sh_ind', 'dnl_mbl_bnk_ind', 'dnl_bind_cmb_lif_ind', 'hav_car_grp_ind', 'hav_hou_grp_ind', 'l6mon_agn_ind', 'frs_agn_dt_cnt', 'vld_rsk_ases_ind', 'fin_rsk_ases_grd_cd', 'confirm_rsk_ases_lvl_typ_cd', 'cust_inv_rsk_endu_lvl_cd', 'l6mon_daim_aum_cd', 'tot_ast_lvl_cd', 'pot_ast_lvl_cd', 'bk1_cur_year_mon_avg_agn_amt_cd', 'l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms', 'loan_act_ind', 'pl_crd_lmt_cd', 'ovd_30d_loan_tot_cnt', 'his_lng_ovd_day', 'hld_crd_card_grd_cd', 'crd_card_act_ind', 'l1y_crd_card_csm_amt_dlm_cd', 'atdd_type', 'perm_crd_lmt_cd', 'cur_debit_cnt', 'cur_credit_cnt', 'cur_debit_min_opn_dt_cnt', 'cur_credit_min_opn_dt_cnt', 'cur_debit_crd_lvl']

tag_df=tag_df[feature_list]
print(len(tag_df.columns))
print(list(tag_df.columns))

# 融合 to_csv

beh_df = beh_df.merge(tag_df,on='id',how='left')

# 检查有无null
#num_beh_df[beh_df.describe().isnull().values]
features = list(beh_df.describe())
num_beh_df = beh_df[features]
num_beh_df.to_csv('num_beh_df.csv',index=False)



### 2. trd_df 共 8 个feature
- ['id', 'flag', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd', 'trx_tm', 'cny_trx_amt']
- [用户标识,目标变量,交易方向,支付方式,收支一级分类代码,收支二级分类代码,交易时间,交易金额]

import pandas as pd
trd_df = pd.read_csv('训练数据集_trd.csv')
print(list(trd_df.columns))

# 融合 to_csv
trd_df = trd_df.merge(tag_df,on='id',how='left')

# 检查有无Nan
#num_trd_df[trd_df.describe().isnull().values]

num_trd_df=trd_df[list(trd_df.describe())]
num_trd_df.to_csv('num_trd_df.csv',index=False)



# 评分数据
- 评分数据集_beh_a.csv
- 评分数据集_trd_a.csv
- 评分数据集_tag_a.csv

## 测试数据集
- 评分数据集_beh_a.csv
- 评分数据集_trd_a.csv
- 评分数据集_tag_a.csv

import pandas as pd
pf_beh_df = pd.read_csv('评分数据集_beh_a.csv')
pf_trd_df = pd.read_csv('评分数据集_trd_a.csv')
pf_tag_df = pd.read_csv('评分数据集_tag_a.csv')
print(len(pf_beh_df))

print(len(set(pf_beh_df['id'])))

feature_list = ['id', 'gdr_cd', 'age', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'job_year', 'ic_ind', 'fr_or_sh_ind', 'dnl_mbl_bnk_ind', 'dnl_bind_cmb_lif_ind', 'hav_car_grp_ind', 'hav_hou_grp_ind', 'l6mon_agn_ind', 'frs_agn_dt_cnt', 'vld_rsk_ases_ind', 'fin_rsk_ases_grd_cd', 'confirm_rsk_ases_lvl_typ_cd', 'cust_inv_rsk_endu_lvl_cd', 'l6mon_daim_aum_cd', 'tot_ast_lvl_cd', 'pot_ast_lvl_cd', 'bk1_cur_year_mon_avg_agn_amt_cd', 'l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms', 'loan_act_ind', 'pl_crd_lmt_cd', 'ovd_30d_loan_tot_cnt', 'his_lng_ovd_day', 'hld_crd_card_grd_cd', 'crd_card_act_ind', 'l1y_crd_card_csm_amt_dlm_cd', 'atdd_type', 'perm_crd_lmt_cd', 'cur_debit_cnt', 'cur_credit_cnt', 'cur_debit_min_opn_dt_cnt', 'cur_credit_min_opn_dt_cnt', 'cur_debit_crd_lvl']

pf_tag_df=pf_tag_df[feature_list]
print(len(pf_tag_df.columns))
print(list(pf_tag_df.columns))

# 融合 to_csv

pf_beh_df = pf_beh_df.merge(pf_tag_df,on='id',how='left')

# 检查有无null
#num_beh_df[beh_df.describe().isnull().values]
features = list(pf_beh_df.describe())

num_pf_beh_df = pf_beh_df[features]
num_pf_beh_df.to_csv('num_pf_beh_df.csv',index=False)

# 融合 to_csv
pf_trd_df = pf_trd_df.merge(pf_tag_df,on='id',how='left')
# 检查有无Nan
#num_trd_df[trd_df.describe().isnull().values]

num_pf_trd_df=pf_trd_df[list(pf_trd_df.describe())]
num_pf_trd_df.to_csv('num_pf_trd_df.csv',index=False)

num_pf_trd_df.head()

```

```python
import pandas as pd

num_beh_df = pd.read_csv('num_beh_df.csv')
num_beh_df.head()

num_trd_df = pd.read_csv('num_trd_df.csv')
num_trd_df.head()

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
f,ax=plt.subplots(1,2,figsize=(10,4))
num_beh_df['flag'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('flag')
ax[0].set_ylabel('')
sns.countplot('flag',data=num_beh_df,ax=ax[1])
ax[1].set_title('flag')
plt.show()

# 模型

#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

data = num_trd_df

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['flag'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['flag']

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))

## 预测

- predict(X)	Predict class labels for samples in X.
- predict_log_proba(X)	Log of probability estimates.
- predict_proba(X)
-----------------------
+ num_pf_beh_df.csv
+ num_pf_trd_df.csv

import pandas as pd
pf_beh_df = pd.read_csv('评分数据集_beh_a.csv')
pf_trd_df = pd.read_csv('评分数据集_trd_a.csv')
pf_tag_df = pd.read_csv('评分数据集_tag_a.csv')
print(len(pf_trd_df))

pf_trd_df_id_list = list(pf_trd_df['id'])
num_pf_trd_df = pd.read_csv('num_pf_trd_df.csv')
num_pf_trd_df['id']=pf_trd_df_id_list

num_pf_trd_df.head()

data = num_pf_trd_df

print(len(data))
X=data[data.columns[:-1]]

# 直接预测 0,1

prediction=model.predict_proba(X)
prediction_list = list(prediction[:,1])

count = 1
for i in prediction[:,1]:
    if i>0.5:
        count+=1
print(count)

from pandas import DataFrame
result_pre_df = DataFrame({'id':pf_trd_df_id_list,'prob':prediction_list})
print(len(result_pre_df))
result_pre_df.head()

result_pre_df.to_csv('result_pre.csv')

count_wy_dict=dict()
for item in set(data['id']):
    count_wy_dict.update({item[1:]:[]})
    


for ids,prob in zip(result_pre_df['id'],result_pre_df['prob']):
    count_wy_dict[ids[1:]].append(prob)



keys = list(count_wy_dict.keys())

keys = map(lambda x : int(x) ,keys)

with open('upload_1.txt','w',encoding='utf-8') as f:
    for k in sorted(list(keys)):

        pred = sum(count_wy_dict[str(k)])/len(count_wy_dict[str(k)])
        f.write('U'+str(k)+'\t'+ str(pred)+'\n')



result_pre_df.to_csv('result_pre.csv')
```

## Pandas

1. 计算频率
```python
import pandas as pd
data1 = {"a":[1.,3.,5.,2.],
         "b":[4.,8.,3.,7.],
         "c":[5.,45.,67.,34]}

df1 = pd.DataFrame(data1)
df2=df1.sum(axis=1)# 按行相加
print(df1.div(df2, axis='rows'))#每一行除以对应的向量
```
2. 

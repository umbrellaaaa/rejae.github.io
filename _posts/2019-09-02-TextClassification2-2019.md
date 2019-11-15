---
layout:     post
title:      文本分类任务
subtitle:    "\"传统与深度学习方式\""
date:       2019-09-02
author:     RJ
header-img: img/banboo.jpg
catalog: true
tags:
    - NLP
---

> “ 纸上得来终觉浅，觉知此事须躬行。”



## 前言
用一周时间完成一个文本分类任务：<br>
1. 回顾一下数据预处理方法
2. 传统方式用LR, Bays, XGBoost
3. 深度学习用CNN，RNN，Transformer

<p id = "build"></p>
---

## 正文

### 数据预处理方法：
数据集介绍：
1. cnews.train.txt  50000
2. cnews.val.txt  5000
3. cnews.test.txt  10000

查看一下数据集：
```python
train_df=pd.read_table(train_dir,header=None)
print(train_df.shape)
train_df.head()
【out:】
(50000, 2)

	0 	1
0 	体育 	马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有...
1 	体育 	商瑞华首战复仇心切 中国玫瑰要用美国方式攻克瑞典多曼来了，瑞典来了，商瑞华首战求3分的信心也...
2 	体育 	冠军球队迎新欢乐派对 黄旭获大奖张军赢下PK赛新浪体育讯12月27日晚，“冠军高尔夫球队迎新...
3 	体育 	辽足签约危机引注册难关 高层威逼利诱合同笑里藏刀新浪体育讯2月24日，辽足爆发了集体拒签风波...
4 	体育 	揭秘谢亚龙被带走：总局电话骗局 复制南杨轨迹体坛周报特约记者张锐北京报道  谢亚龙已经被公安...
----------------------------------------------------------------------

train_df=pd.read_table(val_dir,header=None)
print(train_df.shape)
train_df.head()

(5000, 2)

	0 	1
0 	体育 	黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛...
1 	体育 	1.7秒神之一击救马刺王朝于危难 这个新秀有点牛！新浪体育讯在刚刚结束的比赛中，回到主场的马...
2 	体育 	1人灭掘金！神般杜兰特！ 他想要分的时候没人能挡新浪体育讯在NBA的世界里，真的猛男，敢于直...
3 	体育 	韩国国奥20人名单：朴周永领衔 两世界杯国脚入选新浪体育讯据韩联社首尔9月17日电 韩国国奥...
4 	体育 	天才中锋崇拜王治郅 周琦：球员最终是靠实力说话2月14日从土耳其男篮邀请赛回到北京之后，周琦...
----------------------------------------------------------------------
train_df=pd.read_table(test_dir,header=None)
print(train_df.shape)
train_df.head()

(10000, 2)

	0 	1
0 	体育 	鲍勃库西奖归谁属？ NCAA最强控卫是坎巴还是弗神新浪体育讯如今，本赛季的NCAA进入到了末...
1 	体育 	麦基砍28+18+5却充满寂寞 纪录之夜他的痛阿联最懂新浪体育讯上天对每个人都是公平的，贾维...
2 	体育 	黄蜂vs湖人首发：科比冲击七连胜 火箭两旧将登场新浪体育讯北京时间3月28日，NBA常规赛洛...
3 	体育 	双面谢亚龙作秀终成做作 谁来为低劣行政能力埋单是谁任命了谢亚龙？谁放纵了谢亚龙？谁又该为谢亚...
4 	体育 	兔年首战山西换帅后有虎胆 张学文用乔丹名言励志今晚客场挑战浙江稠州银行队，是山西汾酒男篮的兔...

```
这个数据是已经切分好了的，但是当我们只拿到一个数据集的时候，需要自己切分成train,val，这个时候我们可以借助sklearn的train_test_split方法切分：
```python
from sklearn.model_selection import train_test_split  # 更新
X=list(train_df[1])
y=list(train_df[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
#### 读取数据；构建词表；读取词表；读取类别；将文件转为id表示
```python
def open_file(filename, mode='r'):
    
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except Exception as e:
                print(e)
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content) #将多个list融合用extend方法

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')

    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad
```
#### 生成batch数据
最后一个batch取min(data_len, 最后一个batch的)
```python
def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))#shuffle改变原数组，permutation重新创建
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
```

### TextCNN
cnn_model.py
```python
# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    vocab_size = 5000  # 词汇表达小
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    hidden_dim = 128  # 全连接层神经元
    num_classes = 10  # 类别数

    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸


    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

```
run_cnn.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)
    train()
    if sys.argv[1] == 'train':
        train()
    else:
        test()

```

### CNN模型小结：<br>

--------

CNN，RNN文本分类项目，作者的编码非常规范，tf的命名域处理的也很好，配置文件也处理的杠杠的。<br>
回顾一下cnn模型需要的参数：<br>
字的维度，句子长度，词汇表大小<br>
卷积核大小，卷积核数目，隐藏层大小<br>
dropout, learning_rate<br>
batch_size, num_epochs<br>
print_per_batch, save_per_batch<br>

-----------
接下来是模型相关的组件：
input_x, input_y, keep_prob三个待输入的placeholder<br>
在cpu上执行：embedding_lookup<br>
**在cnn的name_scope中**执行conv1d, 并且最大池化 reduce_max<br>
**在score的name_scope中**，将cnn的最大池化输出接入全连接层tf.layers.dense， 经过dropout, relu后再全连接输出类别层，此时得到的是类别数量大小的一个数组，其中dense的返回说明是： <br>
  Returns:
    Output tensor the same shape as `inputs` except the last dimension is of size `units`. 即输出张量和输入张量形状相同，除了最后一维度的大小是 units，类比一下矩阵相乘就好理解了。<br>
接下来将输出层的值进行softmax, 随后取argmax，得到预测的类别。
**在optimize的name_scope中**，采用交叉熵损失函数，然后reduce_mean后得到loss，再将loss传入到AdamOptimizer中。<br>
**在accuracy的name_scope中**，          
correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)<br>
self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
____
在模型训练的时候，配置好summary的FileWriter(tensorboard_dir)，以及tf.train.saver, 创建好session，初始化参数后，就进入epoch循环中训练。<br>
writer.add_summary(session.run(merged_summary, feed_dict=feed_dict), total_batch) <br>
loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)<br>
session.run(model.optim, feed_dict=feed_dict)#模型训练
### 对比CNN与RNN模型：
```python
    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
```
------------
```python
    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

```
### RNN张量变化分析
对比发现RNN的构建较CNN的麻烦一些，除了LSTM与GRU的cell选择，多层cell的堆叠外，还需要注意_output的输出。<br>
  tf.nn.dynamic_rnn ,    Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

打开pycharm的debug模式，跟踪一下张量的变换：<br>
input: [batch_size, seq_len, embedding_size]<br>
由于是两层的MultiRNNCell, embedding_size传入后，经过两层的RNN，输出两个64维的隐状态，然后拼接成为一个128维的隐状态。动态RNN经过seq_len=600次递归调用，输出output的shape是：[batch_size, max_len, 2*embedding_size]。<br>
我们取的last=_outputs[:,-1,:]即我们只取最后一个单元的输出(即第600个字位置处)。<br>
至于为什么采取concat而不是取最后一层的输出值，根据我的理解是因为第一层抽取浅层特征，高层抽取深层特征，拼接后能兼顾。<br>
之后再接入两个全连接层，就很简单了。
### CNN张量变化分析
input: [batch_size, seq_len, embedding_size]<br>
conv:[batch_size, 600-5+1, 256]，其中600-5+1=596是seq_len用尺寸为5的卷积核卷积后的结果，256是一共256个卷积核。<br>
gmp是由conv采取reduce_max生成:[batch_size, 256]

##参考内容
https://github.com/rejae/text-classification-cnn-rnn


## 后记
本文是基于CNN，RNN的文本分类，后面会接着分析传统的LR,Bays,XGBoost传统文本分类，以及基于transformer的文本分类。


## 实验结果
```
TextRNN train
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:   9.38%, Val Loss:    2.3, Val Acc:  10.78%, Time: 0:00:14 *
Iter:    100, Train Loss:   0.92, Train Acc:  68.75%, Val Loss:    1.1, Val Acc:  63.58%, Time: 0:01:30 *
Iter:    200, Train Loss:   0.57, Train Acc:  83.59%, Val Loss:   0.77, Val Acc:  76.48%, Time: 0:02:45 *
Iter:    300, Train Loss:   0.39, Train Acc:  86.72%, Val Loss:   0.77, Val Acc:  79.44%, Time: 0:04:00 *
Epoch: 2
Iter:    400, Train Loss:   0.26, Train Acc:  93.75%, Val Loss:   0.63, Val Acc:  82.34%, Time: 0:05:16 *
Iter:    500, Train Loss:    0.3, Train Acc:  91.41%, Val Loss:   0.64, Val Acc:  82.12%, Time: 0:06:31
Iter:    600, Train Loss:   0.43, Train Acc:  88.28%, Val Loss:   0.55, Val Acc:  83.50%, Time: 0:07:45 *
Iter:    700, Train Loss:   0.21, Train Acc:  94.53%, Val Loss:   0.54, Val Acc:  84.32%, Time: 0:09:00 *
Epoch: 3
Iter:    800, Train Loss:    0.2, Train Acc:  92.97%, Val Loss:   0.49, Val Acc:  86.74%, Time: 0:10:15 *
Iter:    900, Train Loss:   0.14, Train Acc:  93.75%, Val Loss:   0.46, Val Acc:  88.32%, Time: 0:11:29 *
Iter:   1000, Train Loss:   0.16, Train Acc:  97.66%, Val Loss:   0.42, Val Acc:  88.34%, Time: 0:12:44 *
Iter:   1100, Train Loss:   0.19, Train Acc:  93.75%, Val Loss:   0.41, Val Acc:  89.12%, Time: 0:14:00 *
Epoch: 4
Iter:   1200, Train Loss:   0.13, Train Acc:  97.66%, Val Loss:   0.38, Val Acc:  90.56%, Time: 0:15:15 *
Iter:   1300, Train Loss:  0.051, Train Acc:  97.66%, Val Loss:   0.37, Val Acc:  89.70%, Time: 0:16:29
Iter:   1400, Train Loss:  0.045, Train Acc: 100.00%, Val Loss:   0.34, Val Acc:  91.04%, Time: 0:17:44 *
Iter:   1500, Train Loss:   0.19, Train Acc:  95.31%, Val Loss:   0.36, Val Acc:  90.16%, Time: 0:18:58
Epoch: 5
Iter:   1600, Train Loss:   0.12, Train Acc:  96.09%, Val Loss:   0.35, Val Acc:  90.86%, Time: 0:20:14
Iter:   1700, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:   0.32, Val Acc:  91.86%, Time: 0:21:29 *
Iter:   1800, Train Loss:   0.12, Train Acc:  96.09%, Val Loss:   0.38, Val Acc:  90.30%, Time: 0:22:44
Iter:   1900, Train Loss:  0.055, Train Acc:  97.66%, Val Loss:   0.34, Val Acc:  91.32%, Time: 0:24:00
Epoch: 6
Iter:   2000, Train Loss:  0.045, Train Acc:  98.44%, Val Loss:   0.45, Val Acc:  88.80%, Time: 0:25:15
Iter:   2100, Train Loss:  0.056, Train Acc:  98.44%, Val Loss:   0.38, Val Acc:  90.14%, Time: 0:26:30
Iter:   2200, Train Loss:  0.025, Train Acc: 100.00%, Val Loss:   0.37, Val Acc:  90.98%, Time: 0:27:44
Iter:   2300, Train Loss:  0.094, Train Acc:  96.88%, Val Loss:   0.41, Val Acc:  89.30%, Time: 0:28:59
Epoch: 7
Iter:   2400, Train Loss:  0.028, Train Acc:  99.22%, Val Loss:   0.35, Val Acc:  91.50%, Time: 0:30:13
Iter:   2500, Train Loss:  0.055, Train Acc:  99.22%, Val Loss:   0.33, Val Acc:  91.54%, Time: 0:31:28
Iter:   2600, Train Loss:  0.073, Train Acc:  97.66%, Val Loss:   0.35, Val Acc:  91.46%, Time: 0:32:42
Iter:   2700, Train Loss:   0.13, Train Acc:  96.09%, Val Loss:   0.32, Val Acc:  91.62%, Time: 0:33:57
No optimization for a long time, auto-stopping...

Test
Testing...
Test Loss:    0.2, Test Acc:  94.62%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       0.98      0.98      0.98      1000
          财经       0.93      0.98      0.96      1000
          房产       1.00      0.99      1.00      1000
          家居       0.96      0.80      0.87      1000
          教育       0.92      0.92      0.92      1000
          科技       0.94      0.96      0.95      1000
          时尚       0.91      0.97      0.94      1000
          时政       0.92      0.94      0.93      1000
          游戏       0.96      0.96      0.96      1000
          娱乐       0.95      0.95      0.95      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000

Confusion Matrix...
[[983   0   0   0   2   0   0   0   1  14]
 [  0 983   1   0   4   1   1  10   0   0]
 [  0   1 995   2   1   0   0   1   0   0]
 [  0  29   0 795  32  24  54  52   7   7]
 [ 15  18   0   4 919  16   2   8   6  12]
 [  0   1   0   8   7 964   3   2  14   1]
 [  0   1   0  12   5   1 966   2   5   8]
 [  0  18   0   2  23  11   0 942   3   1]
 [  4   2   0   0   5   6  19   0 960   4]
 [  3   1   1   8   5   5  13   4   5 955]]
Time usage: 0:00:53

```

```
Training and evaluating...
Text_CNN train
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  15.62%, Val Loss:    2.3, Val Acc:  10.40%, Time: 0:00:04 *
Iter:    100, Train Loss:   0.59, Train Acc:  82.81%, Val Loss:    1.1, Val Acc:  69.80%, Time: 0:00:07 *
Iter:    200, Train Loss:   0.22, Train Acc:  93.75%, Val Loss:   0.58, Val Acc:  81.86%, Time: 0:00:09 *
Iter:    300, Train Loss:   0.22, Train Acc:  90.62%, Val Loss:   0.39, Val Acc:  89.04%, Time: 0:00:12 *
Iter:    400, Train Loss:   0.19, Train Acc:  95.31%, Val Loss:   0.33, Val Acc:  90.94%, Time: 0:00:14 *
Iter:    500, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:    0.3, Val Acc:  91.62%, Time: 0:00:16 *
Iter:    600, Train Loss:   0.39, Train Acc:  90.62%, Val Loss:   0.35, Val Acc:  90.12%, Time: 0:00:18
Iter:    700, Train Loss:  0.068, Train Acc:  98.44%, Val Loss:   0.27, Val Acc:  92.84%, Time: 0:00:20 *
Epoch: 2
Iter:    800, Train Loss:   0.18, Train Acc:  90.62%, Val Loss:   0.23, Val Acc:  92.92%, Time: 0:00:22 *
Iter:    900, Train Loss:   0.23, Train Acc:  90.62%, Val Loss:   0.22, Val Acc:  93.30%, Time: 0:00:24 *
Iter:   1000, Train Loss:   0.11, Train Acc:  98.44%, Val Loss:   0.27, Val Acc:  92.04%, Time: 0:00:26
Iter:   1100, Train Loss:   0.14, Train Acc:  95.31%, Val Loss:   0.23, Val Acc:  92.90%, Time: 0:00:28
Iter:   1200, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:    0.2, Val Acc:  93.96%, Time: 0:00:30 *
Iter:   1300, Train Loss:  0.083, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  93.82%, Time: 0:00:32
Iter:   1400, Train Loss:   0.19, Train Acc:  93.75%, Val Loss:   0.24, Val Acc:  93.26%, Time: 0:00:34
Iter:   1500, Train Loss:  0.046, Train Acc:  98.44%, Val Loss:    0.2, Val Acc:  94.16%, Time: 0:00:36 *
Epoch: 3
Iter:   1600, Train Loss:  0.034, Train Acc:  98.44%, Val Loss:    0.2, Val Acc:  94.14%, Time: 0:00:38
Iter:   1700, Train Loss:  0.044, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.50%, Time: 0:00:40 *
Iter:   1800, Train Loss:  0.025, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  94.94%, Time: 0:00:42
Iter:   1900, Train Loss:  0.093, Train Acc:  96.88%, Val Loss:   0.19, Val Acc:  94.26%, Time: 0:00:44
Iter:   2000, Train Loss:  0.089, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.70%, Time: 0:00:46 *
Iter:   2100, Train Loss:  0.063, Train Acc:  95.31%, Val Loss:   0.16, Val Acc:  95.08%, Time: 0:00:48
Iter:   2200, Train Loss:  0.046, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  94.18%, Time: 0:00:50
Iter:   2300, Train Loss: 0.0093, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.32%, Time: 0:00:52
Epoch: 4
Iter:   2400, Train Loss:  0.016, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.20%, Time: 0:00:54
Iter:   2500, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  94.54%, Time: 0:00:56
Iter:   2600, Train Loss: 0.0055, Train Acc: 100.00%, Val Loss:   0.18, Val Acc:  94.86%, Time: 0:00:58
Iter:   2700, Train Loss:   0.14, Train Acc:  95.31%, Val Loss:   0.21, Val Acc:  94.28%, Time: 0:01:00
Iter:   2800, Train Loss: 0.0078, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  94.72%, Time: 0:01:02
Iter:   2900, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  95.00%, Time: 0:01:04
Iter:   3000, Train Loss: 0.0038, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  94.08%, Time: 0:01:06
No optimization for a long time, auto-stopping...

Test:
Testing...
Test Loss:   0.13, Test Acc:  96.06%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       0.99      0.99      0.99      1000
          财经       0.93      0.99      0.96      1000
          房产       1.00      0.99      1.00      1000
          家居       0.99      0.84      0.91      1000
          教育       0.92      0.94      0.93      1000
          科技       0.94      0.99      0.96      1000
          时尚       0.97      0.97      0.97      1000
          时政       0.93      0.95      0.94      1000
          游戏       0.99      0.96      0.98      1000
          娱乐       0.95      0.98      0.97      1000

    accuracy                           0.96     10000
   macro avg       0.96      0.96      0.96     10000
weighted avg       0.96      0.96      0.96     10000

Confusion Matrix...
[[991   0   0   0   3   2   0   3   0   1]
 [  0 992   0   0   1   1   0   6   0   0]
 [  0   2 994   0   2   0   0   0   0   2]
 [  1  42   3 836  22  28  18  37   1  12]
 [  1   9   0   1 942  14   7  15   1  10]
 [  0   2   0   1   3 988   2   2   1   1]
 [  1   2   0   1   8   4 966   1   3  14]
 [  0  16   0   1  27   6   0 948   1   1]
 [  1   6   0   0  10   3   4   1 964  11]
 [  1   1   0   2   5   3   2   1   0 985]]
Time usage: 0:00:07

```
对比结果发现CNN的效果居然大于RNN的效果，考虑embedding_size=64太小，所以我打算增大到100进行实验：<br>
CNN训练后的测试结果：
```
Testing...
Test Loss:   0.13, Test Acc:  96.84%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       0.99      0.99      0.99      1000
          财经       0.96      0.99      0.97      1000
          房产       1.00      1.00      1.00      1000
          家居       0.99      0.90      0.94      1000
          教育       0.93      0.94      0.93      1000
          科技       0.95      0.98      0.97      1000
          时尚       0.97      0.97      0.97      1000
          时政       0.95      0.96      0.95      1000
          游戏       0.97      0.98      0.98      1000
          娱乐       0.97      0.98      0.98      1000

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000

Confusion Matrix...
[[993   0   0   0   3   1   0   0   2   1]
 [  0 989   0   0   5   2   0   4   0   0]
 [  0   1 998   0   1   0   0   0   0   0]
 [  0  20   2 898  20  20  10  22   3   5]
 [  1   2   0   4 938  15   6  21   6   7]
 [  0   1   0   2   3 982   1   2   8   1]
 [  1   1   0   3   7   3 970   1   3  11]
 [  1  13   0   1  20   8   0 956   1   0]
 [  1   1   0   0   8   1   6   0 982   1]
 [  1   1   0   3   7   3   4   0   3 978]]
Time usage: 0:00:07

```
测试精度从96.06%提高到了96.84%。
```
RNN Test
Test Loss:   0.24, Test Acc:  94.63%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       0.99      0.99      0.99      1000
          财经       0.93      0.98      0.96      1000
          房产       0.99      0.99      0.99      1000
          家居       0.99      0.77      0.86      1000
          教育       0.93      0.89      0.91      1000
          科技       0.96      0.97      0.96      1000
          时尚       0.89      0.97      0.93      1000
          时政       0.89      0.95      0.92      1000
          游戏       0.95      0.97      0.96      1000
          娱乐       0.97      0.96      0.97      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000

Confusion Matrix...
[[992   0   0   0   4   1   0   0   3   0]
 [  0 983   2   0   2   1   0  12   0   0]
 [  0   2 995   1   1   0   1   0   0   0]
 [  5  32  11 765  32  10  87  46   7   5]
 [  1   6   0   1 892  18   8  55  13   6]
 [  0   3   0   2   2 974   4   1  14   0]
 [  0   0   0   5   6   2 974   1   5   7]
 [  0  19   0   0  16   7   0 952   4   2]
 [  0   1   0   1   2   2  13   3 972   6]
 [  3   8   1   1   4   4  10   0   5 964]]
Time usage: 0:00:54

```
RNN测试从Test Loss:    0.2, Test Acc:  94.62%---->>Test Loss:   0.24, Test Acc:  94.63%
对比CNN  Test Loss:   0.13, Test Acc:  96.06%---->>Test Loss:   0.13, Test Acc:  96.84%
效果提升了近1个百分点，而RNN几乎没有变化，查看代码，发现作者的CNN设置的dropout=0.5而rnn的却是0.8，可能是为了方便RNN快速训练吧，所以我将embedding_size从64->100并没有什么效果，而且训练到10个epoch的时候结束的，并没有早停，查看训练结果，发现train的acc都快逼近100%了，而valid只在90%徘徊，估计是过拟合了吧。于是我先调整dropout=0.5试试。
设置dropout=0.5的测试结果为：Test Loss:   0.19, Test Acc:  95.29%，对比Test Acc:  94.63%，嗯，有了0.66%的提升。
## 使用word2vec100维词嵌入测试模型：
使用wiki_100.utf词向量，进行训练，通过如下方法接入模型：
1. 将model中的embedding设为类变量以便访问：self.embedding, 在model中添加emb_file路径, use_pretrained的flag
2. 在run_rnn中添加load_word2vec方法如下：
```python
def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split() ##取出pretrain_file的一行
        if len(line) == word_dim + 1: ##char+嵌入维度
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32) ## 形成pre_trained字典，格式为'char':array([100维])
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):  # KeyError: 17 是' '
        print(i)
        print(id_to_word[i])
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
              c_found + c_lower + c_zeros, n_words,
              100. * (c_found + c_lower + c_zeros) / n_words)
          )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
              c_found, c_lower, c_zeros
          ))
    return new_weights
```
在模型session init之后，加入：
```python

    # use pre_trained
    if config.use_pre_trained:
        id_to_char = {value : key for key, value in word_to_id.items()}
        emb_weights = session.run(model.embedding.read_value())
        emb_weights = load_word2vec(config.emb_file, id_to_char, config.embedding_dim, emb_weights)
        session.run(model.embedding.assign(emb_weights))
```
即可将预训练词向量传入模型。替换预训练向量，测试结果如下：

CNN Testing...
Test Loss:   0.14, Test Acc:  95.91%

LSTM Testing...
Test Loss:   0.15, Test Acc:  95.72%


对比之前的结果：
<br>
```
embedding_size=64 --->>  100
RNN  Test Loss:    0.2, Test Acc:  94.62%---->>Test Loss:   0.24, Test Acc:  94.63%
CNN  Test Loss:   0.13, Test Acc:  96.06%---->>Test Loss:   0.13, Test Acc:  96.84%
```
<br>
发现CNN Testing有所下降为0.93%，而RNN的提升也很明显为1.11%。个人思考应该是使用wiki_data预训练词向量对LSTM的前后记忆联系有较大的帮助，而CNN更只是滑动窗口似的进行卷积池化，语义联系不如RNN大，所以这个好处get不到。

## 为RNN加入attention机制
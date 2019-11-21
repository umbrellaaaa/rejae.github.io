---
layout:     post
title:      Project Structure
subtitle:   summary
date:       2019-11-14
author:     RJ
header-img: 
catalog: true
tags:
    - DL

---
<p id = "build"></p>
---

## 前言
在学习深度学习模型的时候，书中为了更简单的向学者呈现知识，往往代码很简洁、零碎，总结一下需要注意的地方，和我需要进一步学习的地方：
- 代码的模块化
- 数据处理流程规范，处理结果的保存
- 实验参数和结果的保存
- tensorboard可视化参数的设置
- 控制好过拟合问题（模型训练停止的策略，正则化参数的设定，dropout参数的设定）

## 一.模块化 
- 数据加载
- 模型构建
- 模型训练

### 数据加载基类

```python
class DataBase(object):
    def __init__(self, config):
        self.config = config

    def read_data(self):
        raise NotImplementedError

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        raise NotImplementedError

    def gen_data(self):
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        raise NotImplementedError


class TrainDataBase(DataBase):
    def __init__(self, config):
        super(TrainDataBase, self).__init__(config)

    def read_data(self):
        raise NotImplementedError

    def remove_stop_word(self, inputs):
        raise NotImplementedError

    def get_word_vectors(self, words):
        raise NotImplementedError

    def gen_vocab(self, words, labels):
        raise NotImplementedError

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        raise NotImplementedError

    def gen_data(self):
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        raise NotImplementedError


class EvalPredictDataBase(DataBase):
    def __init__(self, config):
        super(EvalPredictDataBase, self).__init__(config)

    def read_data(self):
        raise NotImplementedError

    def load_vocab(self):
        raise NotImplementedError

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        raise NotImplementedError

    def padding(self, inputs, sequence_length):
        raise NotImplementedError

    def gen_data(self):
        raise NotImplementedError

    def next_batch(self, x, y, batch_size):
        raise NotImplementedError


```

### 模型构建父类
基类初始化参数： config, vocab_size=None, word_vectors=None

其中config是模型的配置参数是配置参数的路径，以config/bilstm_atten_config.json为例，包括以下参数：
```
{
  "model_name": "bilstm_atten",
  "epochs": 2,
  "checkpoint_every": 100,
  "eval_every": 100,
  "learning_rate": 1e-3,
  "optimization": "adam",
  "embedding_size": 200,
  "hidden_sizes": [256,128],
  "sequence_length": 100,
  "batch_size": 128,
  "vocab_size": 10000,
  "num_classes": 1,
  "keep_prob": 0.5,
  "l2_reg_lambda": 0.0,
  "max_grad_norm": 5.0,
  "train_data": "data/imdb/train_data.txt",
  "eval_data": "data/imdb/eval_data.txt",
  "stop_word": "data/english",
  "output_path": "outputs/imdb/bilstm_atten",
  "word_vectors_path": null,
  "ckpt_model_path": "ckpt_model/imdb/bilstm_atten",
  "pb_model_path": "pb_model/imdb/bilstm_atten"
}
```
```python

基类具有以下几类方法：
init_saver()
build_model()： 空实现
train()
eval()
infer()
get_train_op()
get_optimizer()
cal_loss()
get_predictions()

import tensorflow as tf
class BaseModel(object):
    def __init__(self, config, vocab_size=None, word_vectors=None):
        """
        文本分类的基类，提供了各种属性和训练，验证，测试的方法
        :param config: 模型的配置参数
        :param vocab_size: 当不提供词向量的时候需要vocab_size来初始化词向量
        :param word_vectors：预训练的词向量，word_vectors 和 vocab_size必须有一个不为None
        """
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")  # 数据输入
        self.labels = tf.placeholder(tf.float32, [None], name="labels")  # 标签
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.l2_loss = tf.constant(0.0)  # 定义l2损失
        self.loss = 0.0  # 损失
        self.train_op = None  # 训练入口
        self.summary_op = None
        self.logits = None  # 模型最后一层的输出
        self.predictions = None  # 预测结果
        self.saver = None  # 保存为ckpt模型的对象

    def cal_loss(self):
        """
        计算损失，支持二分类和多分类
        :return:
        """
        with tf.name_scope("loss"):
            losses = 0.0
            if self.config["num_classes"] == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.reshape(self.labels, [-1, 1]))
            elif self.config["num_classes"] > 1:
                self.labels = tf.cast(self.labels, dtype=tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.labels)
            loss = tf.reduce_mean(losses)
            return loss

    def get_optimizer(self):
        """
        获得优化器
        :return:
        """
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.config["learning_rate"])
        return optimizer

    def get_train_op(self):
        """
        获得训练的入口
        :return:
        """
        # 定义优化器
        optimizer = self.get_optimizer()

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        tf.summary.scalar("loss", self.loss)
        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def get_predictions(self):
        """
        得到预测结果
        :return:
        """
        predictions = None
        if self.config["num_classes"] == 1:
            predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
        elif self.config["num_classes"] > 1:
            predictions = tf.argmax(self.logits, axis=-1, name="predictions")
        return predictions

    def build_model(self):
        """
        创建模型
        :return:
        """
        raise NotImplementedError

    def init_saver(self):
        """
        初始化saver对象
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :param dropout_prob: dropout比例
        :return: 损失和预测结果
        """

        feed_dict = {self.inputs: batch["x"],
                     self.labels: batch["y"],
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, summary, loss, predictions = sess.run([self.train_op, self.summary_op, self.loss, self.predictions],
                                                 feed_dict=feed_dict)
        return summary, loss, predictions

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.inputs: batch["x"],
                     self.labels: batch["y"],
                     self.keep_prob: 1.0}

        summary, loss, predictions = sess.run([self.summary_op, self.loss, self.predictions], feed_dict=feed_dict)
        return summary, loss, predictions

    def infer(self, sess, inputs):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param inputs: batch数据
        :return: 预测结果
        """
        feed_dict = {self.inputs: inputs,
                     self.keep_prob: 1.0}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
```

### BiLstmAttenModel继承基类并实现build_model
```python
import tensorflow as tf
from .base import BaseModel


class BiLstmAttenModel(BaseModel):
    def __init__(self, config, vocab_size, word_vectors):
        super(BiLstmAttenModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.contrib.layers.xavier_initializer())
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)

            # 定义两层双向LSTM的模型结构
            with tf.name_scope("Bi-LSTM"):
                for idx, hidden_size in enumerate(self.config["hidden_sizes"]):
                    with tf.name_scope("Bi-LSTM" + str(idx)):
                        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                            output_keep_prob=self.keep_prob)
                        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                            output_keep_prob=self.keep_prob)

                        # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                        outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                                 embedded_words, dtype=tf.float32,
                                                                                 scope="bi-lstm" + str(idx))

                        # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2] 再送入下一层
                        embedded_words = tf.concat(outputs, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(embedded_words, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)
            output_size = self.config["hidden_sizes"][-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hidden_size = self.config["hidden_sizes"][-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，M=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config["sequence_length"]])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作[batch, hidden, time]  [batch, time, 1]  --->>>[batch, hidden, 1]
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config["sequence_length"], 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.keep_prob)

        return output

```

### 模型训练基类
训练基类需要实现：
1. 加载数据
2. 创建模型
3. 训练模型

共三个基础方法
```python
# train_base.py
class TrainerBase(object):
    def __init__(self):
        self.model = None  # 模型的初始化对象
        self.config = None  # 模型的配置参数
        self.current_step = 0

    def load_data(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
```
### train继承train_base

```python
import json
import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import tensorflow as tf
from data_helpers import TrainData, EvalData
from trainers.train_base import TrainerBase
from models import TextCnnModel, BiLstmModel, BiLstmAttenModel, RcnnModel, TransformerModel
from utils.metrics import get_binary_metrics, get_multi_metrics, mean


class Trainer(TrainerBase):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), args.config_path), "r") as fr:
            self.config = json.load(fr)

        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        # self.builder = tf.saved_model.builder.SavedModelBuilder("../pb_model/weibo/bilstm/savedModel")

        # 加载数据集
        self.load_data()
        self.train_inputs, self.train_labels, label_to_idx = self.train_data_obj.gen_data()
        print("train data size: {}".format(len(self.train_labels)))
        self.vocab_size = self.train_data_obj.vocab_size
        print("vocab size: {}".format(self.vocab_size))
        self.word_vectors = self.train_data_obj.word_vectors
        self.label_list = [value for key, value in label_to_idx.items()]

        self.eval_inputs, self.eval_labels = self.eval_data_obj.gen_data()
        print("eval data size: {}".format(len(self.eval_labels)))

        # 初始化模型对象
        self.create_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        self.train_data_obj = TrainData(self.config)

        # 生成验证集对象和验证集数据
        self.eval_data_obj = EvalData(self.config)

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "textcnn":
            self.model = TextCnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "bilstm":
            self.model = BiLstmModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "bilstm_atten":
            self.model = BiLstmAttenModel(config=self.config, vocab_size=self.vocab_size,
                                          word_vectors=self.word_vectors)
        elif self.config["model_name"] == "rcnn":
            self.model = RcnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "transformer":
            self.model = TransformerModel(config=self.config, vocab_size=self.vocab_size,
                                          word_vectors=self.word_vectors)

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                              self.config["output_path"] + "/summary/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                             self.config["output_path"] + "/summary/eval")
            #result 存储
            result_name = self.config['output_result']
            with open(result_name, 'a', encoding='utf-8') as f:
                f.write(str(self.config))
                f.write('\n')
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.train_data_obj.next_batch(self.train_inputs, self.train_labels,
                                                            self.config["batch_size"]):
                    summary, loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])
                    train_summary_writer.add_summary(summary)

                    if self.config["num_classes"] == 1:
                        acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch["y"])
                        print(
                            "train: step: {}, loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                                current_step, loss, acc, auc, recall, prec, f_beta))
                    elif self.config["num_classes"] > 1:
                        acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch["y"],
                                                                      labels=self.label_list)
                        print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            current_step, loss, acc, recall, prec, f_beta))

                    current_step += 1
                    if self.eval_data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.eval_data_obj.next_batch(self.eval_inputs, self.eval_labels,
                                                                        self.config["batch_size"]):
                            eval_summary, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_summary_writer.add_summary(eval_summary)

                            eval_losses.append(eval_loss)
                            if self.config["num_classes"] == 1:
                                acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=eval_predictions,
                                                                                    true_y=eval_batch["y"])
                                eval_accs.append(acc)
                                eval_aucs.append(auc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                            elif self.config["num_classes"] > 1:
                                acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                              true_y=eval_batch["y"],
                                                                              labels=self.label_list)
                                eval_accs.append(acc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                        print("\n")
                        print("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                        print("\n")

                        with open(result_name,'a',encoding='utf-8') as f:
                            f.write("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                            f.write('\n')

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model", default="config/bilstm_atten_config.json")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
```


## 二. 数据处理流程规范，处理结果的保存
```python
        # 加载数据集
        self.load_data()
        self.train_inputs, self.train_labels, label_to_idx = self.train_data_obj.gen_data()
        print("train data size: {}".format(len(self.train_labels)))
        self.vocab_size = self.train_data_obj.vocab_size
        print("vocab size: {}".format(self.vocab_size))
        self.word_vectors = self.train_data_obj.word_vectors
        self.label_list = [value for key, value in label_to_idx.items()]

        self.eval_inputs, self.eval_labels = self.eval_data_obj.gen_data()
        print("eval data size: {}".format(len(self.eval_labels)))
```
其中load_data():
```
    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        self.train_data_obj = TrainData(self.config)

        # 生成验证集对象和验证集数据
        self.eval_data_obj = EvalData(self.config)
```
其中gen_data():
```
    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 如果不是第一次数据预处理，则直接读取
        if os.path.exists(os.path.join(self._output_path, "train_data.pkl")) and \
                os.path.exists(os.path.join(self._output_path, "label_to_index.pkl")) and \
                os.path.exists(os.path.join(self._output_path, "word_to_index.pkl")):
            print("load existed train data")
            with open(os.path.join(self._output_path, "train_data.pkl"), "rb") as f:
                train_data = pickle.load(f)

            with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
                word_to_index = pickle.load(f)

            self.vocab_size = len(word_to_index)

            with open(os.path.join(self._output_path, "label_to_index.pkl"), "rb") as f:
                label_to_index = pickle.load(f)

            if os.path.exists(os.path.join(self._output_path, "word_vectors.npy")):
                self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))

            return np.array(train_data["inputs_idx"]), np.array(train_data["labels_idx"]), label_to_index

        # 1，读取原始数据
        inputs, labels = self.read_data()
        print("read finished")

        # 2，得到去除低频词和停用词的词汇表
        words = self.remove_stop_word(inputs)
        print("word process finished")

        # 3，得到词汇表
        word_to_index, label_to_index = self.gen_vocab(words, labels)
        print("vocab process finished")

        # 4，输入转索引
        inputs_idx = self.trans_to_index(inputs, word_to_index)
        print("index transform finished")

        # 5，对输入做padding
        inputs_idx = self.padding(inputs_idx, self._sequence_length)
        print("padding finished")

        # 6，标签转索引
        labels_idx = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
        with open(os.path.join(self._output_path, "train_data.pkl"), "wb") as fw:
            pickle.dump(train_data, fw)

        return np.array(inputs_idx), np.array(labels_idx), label_to_index
```
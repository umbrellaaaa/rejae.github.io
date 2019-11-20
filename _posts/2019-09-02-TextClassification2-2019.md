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
用一段时间完成一个文本分类任务：<br>
1. 回顾一下数据预处理方法
2. 传统方式用LR, Bays
3. 深度学习用CNN，RNN(多层、双向、attention)，Transformer

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

val_df=pd.read_table(val_dir,header=None)
print(val_df.shape)
val_df.head()

(5000, 2)

	0 	1
0 	体育 	黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛...
1 	体育 	1.7秒神之一击救马刺王朝于危难 这个新秀有点牛！新浪体育讯在刚刚结束的比赛中，回到主场的马...
2 	体育 	1人灭掘金！神般杜兰特！ 他想要分的时候没人能挡新浪体育讯在NBA的世界里，真的猛男，敢于直...
3 	体育 	韩国国奥20人名单：朴周永领衔 两世界杯国脚入选新浪体育讯据韩联社首尔9月17日电 韩国国奥...
4 	体育 	天才中锋崇拜王治郅 周琦：球员最终是靠实力说话2月14日从土耳其男篮邀请赛回到北京之后，周琦...
----------------------------------------------------------------------
test_df=pd.read_table(test_dir,header=None)
print(test_df.shape)
test_df.head()

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
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

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
**在score的name_scope中**将cnn的最大池化输出接入全连接层tf.layers.dense, 经过dropout, relu后再全连接输出类别层，此时得到的是类别数量大小的一个数组，其中dense的返回说明是： <br>
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
### RNN模型小结
embedding_size = 64维，两层GRU layer都是128维， time_step=600, 两层layer后取最后一个time_step结果传入projection_layer，再传入分类Layer.



## 实验
1. 实验1 采用原始参数训练，几个关键参数：
    - GRU 64维自训练词向量、单向、单层
    - GRU 64维自训练词向量、单向、双层[128,128]（原始）
    - GRU 64维自训练词向量、单向、双层[256,128]
    
2. 实验2 双层[256,128]效果更好，对比增加wiki预训练词向量对比**实验1**：
    - GRU 100维自训练词向量、单向、双层[256,128]
    - GRU 100维wiki预训练词向量、单向、双层[256,128]

3. 实验3 增加wiki预训练词向量效果更好，增加attention对比**实验2**：
    - GRU 100维wiki预训练词向量、单向、双层[256,128]、有attention

4. 实验4 增加双向GRU，对比**实验3**：
    - GRU 双向、双层[256,128], 无attention
    - GRU 双向、双层[256,128], 有attention

5. 实验5 使用LSTM_Cell,对比GRU_Cell**实验4**



实验1： 调整config.hidden_dim=[]
```
- Test1 【128】    Loss:    0.2, Test Acc:  94.71%
- Test2 【128,128】Loss:    0.2, Test Acc:  94.62%
- Test3 【256,128】Loss:   0.22, Test Acc:  95.23%
```
实验2 是否使用wiki_100.utf8：  use_pre_trained = True/False
```
- Test Loss:【256,128】否   0.19, Test Acc:  94.86%
- Test Loss:【256,128】是   0.17, Test Acc:  95.47%
```

实验3  wiki_100.utf8：【256,128】  +  attention： 调整 last = self._attention(_outputs)
```
Testing...  
Test Loss:   0.16, Test Acc:  96.18%
```

实验4 GRU双向+  有无 attention对比：  双向tf.nn.dynamic_rnn--->>---tf.nn.bidirectional_dynamic_rnn
```
Test Loss:无attention    0.2, Test Acc:  95.79%
Test Loss:有attention   0.13, Test Acc:  96.39%
```

实验5 LSTM双向+attention VS GRU双向+attention：<br>
```
Test Loss:   0.14, Test Acc:  95.99%


```




## 具体实验细节
### 使用word2vec100维词嵌入测试模型：
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

### 为RNN加入attention机制
原来代码只是利用了LSTM的最后一个time_step的输出[batch_size, hidden_dim]传给project_layer进行计算，这里通过_attention方法利用每个time_step的LSTM输出，进行加权计算：
```python
    def _attention(self, _outputs):
        output_reshape = tf.reshape(_outputs, [-1, self.config.hidden_dim])
        w = tf.Variable(tf.random_normal([self.config.hidden_dim], stddev=0.1))
        w_reshape = tf.reshape(w, [-1, 1])
        M = tf.matmul(output_reshape, w_reshape)
        M_shape = tf.reshape(M, [-1, self.config.seq_length])
        self.alpha = tf.nn.softmax(M_shape)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(_outputs, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.seq_length, 1]))
        sequeezeR = tf.squeeze(r)
        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, self.keep_prob)

        return tf.reshape(sentenceRepren, [-1, self.config.hidden_dim])
```
测试结果
```
Testing...   双层LSTM 256   128
Test Loss:   0.21, Test Acc:  94.28%

Testing...   双层LSTM 256   128  +  attention
Test Loss:   0.16, Test Acc:  96.18%
```
提升效果还不错。

### 单向LSTM+attention  VS  双向LSTM+attention
双向LSTM模型
```
    # 定义两层双向LSTM的模型结构
    with tf.name_scope("Bi-LSTM"):
        for idx, hidden_size in enumerate(self.config["hidden_sizes"]):
            with tf.name_scope("Bi-LSTM" + str(idx)):
                # 定义前向LSTM结构
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                    output_keep_prob=self.keep_prob)
                # 定义反向LSTM结构
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                    output_keep_prob=self.keep_prob)

                # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                # fw和bw的hidden_size一样
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

```



## 后记  1

测试tf.reshape的用法：
```python
# 相当于batch == 3， time_step=2,  hidden = 4
import numpy as np
a = np.array(
[
    [
        [2,3,4,5],
        [5,6,7,8]
    ],
    [
        [1,2,3,4],
        [4,5,6,7]
    ],
    [
        [6,7,8,9],
        [4,5,6,7]
    ]
])
out:
a = array([[[2, 3, 4, 5],
        [5, 6, 7, 8]],

       [[1, 2, 3, 4],
        [4, 5, 6, 7]],

       [[6, 7, 8, 9],
        [4, 5, 6, 7]]])

import tensorflow as tf

op = tf.reshape(a,[-1,4])
sess = tf.InteractiveSession()
sess.run(op)

out:
array([[2, 3, 4, 5],
       [5, 6, 7, 8],
       [1, 2, 3, 4],
       [4, 5, 6, 7],
       [6, 7, 8, 9],
       [4, 5, 6, 7]])
```
即，reshape是从内到外，先从括号最里面开始展开，文本分类中的张量[batch_size,time_step,hidden_size]在attention那一层中使用了reshape(H,[-1,hidden_size])
本文是基于CNN，RNN的文本分类，后面会接着分析传统的LR,Bays,XGBoost传统文本分类，以及基于transformer的文本分类。

## 后记  2
- batch数据生成,使用np.random.permutation取得随机index得到shuffle数据。
注意到这个函数是个生成器，每次yield一批数据。
```python
def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
```

- 训练日志、可视化参数与模型提前终止

```python
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
```

## 参考内容
[github-CNN-RNN for text classification](https://github.com/rejae/text-classification-cnn-rnn)

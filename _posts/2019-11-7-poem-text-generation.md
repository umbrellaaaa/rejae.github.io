---
layout:     post
title:      poem text generation
subtitle:   summary
date:       2019-11-7
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## 相关文献阅读
[理解dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)
[Chinese Poetry Generation with a Working Memory Model](https://arxiv.org/pdf/1809.04306.pdf)
[GITHUB Code](https://github.com/xiaoyuanYi/WMPoetry)

## char_rnn_model.py

- 初始化配置参数：
    - is_training
    - w2v_model
    - infer 

- 初始化模型参数：
    - batch_size
    - num_unrollings
    - embedding_size
    - vocab_size
    - hidden_size
    - max_grad_norm
    - cell_type
    - num_layers
    - learning_rate
    - input_dropout
    - dropout

```python
    def __init__(self,
                 is_training, batch_size, num_unrollings, vocab_size, w2v_model,
                 hidden_size, max_grad_norm, embedding_size, num_layers, learning_rate,
                 cell_type, dropout=0.0, input_dropout=0.0, infer=False):

        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        if infer:
            self.batch_size = 1
            self.num_unrollings = 1

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.w2v_model = w2v_model

        if embedding_size <= 0:
            self.input_size = vocab_size
            self.input_dropout = 0.0
        else:
            self.input_size = embedding_size

        self.input_data = tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='inputs')
        self.targets = tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='targets')

        if self.cell_type == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif self.cell_type == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        elif self.cell_type == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell

        params = dict()
        # params['num_units'] = self.hidden_size

        if self.cell_type == 'lstm':
            params['forget_bias'] = 1.0  # 1.0 is default value
        cell = cell_fn(self.hidden_size, **params)

        cells = [cell]
        # params['input_size'] = self.hidden_size
        for i in range(self.num_layers - 1):
            higher_layer_cell = cell_fn(self.hidden_size, **params)
            cells.append(higher_layer_cell)

        if is_training and self.dropout > 0:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout) for cell in cells]

        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        with tf.name_scope('initial_state'):
            self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)
            if self.cell_type == 'rnn' or self.cell_type == 'gru':
                self.initial_state = tuple(
                    [tf.placeholder(tf.float32,
                                    [self.batch_size, multi_cell.state_size[idx]],
                                    'initial_state_' + str(idx + 1)) for idx in range(self.num_layers)])
            elif self.cell_type == 'lstm':
                self.initial_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(
                        tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size[idx][0]],
                                       'initial_lstm_state_' + str(idx + 1)),
                        tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size[idx][1]],
                                       'initial_lstm_state_' + str(idx + 1)))
                        for idx in range(self.num_layers)])

        with tf.name_scope('embedding_layer'):
            if embedding_size > 0:
                # self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
                self.embedding = tf.get_variable("word_embeddings",
                                                 initializer=self.w2v_model.vectors.astype(np.float32))

            else:
                self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)

            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            if is_training and self.input_dropout > 0:
                inputs = tf.nn.dropout(inputs, 1 - self.input_dropout)

        with tf.name_scope('slice_inputs'):
            # num_unrollings * (batch_size, embedding_size), the format of rnn inputs.
            sliced_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(
                axis=1, num_or_size_splits=self.num_unrollings, value=inputs)]

        # sliced_inputs: list of shape xx
        # inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size]
        # initial_state: An initial state for the RNN.
        #                If cell.state_size is an integer, this must be a Tensor of appropriate
        #                type and shape [batch_size, cell.state_size]
        # outputs: a length T list of outputs (one for each input), or a nested tuple of such elements.
        # state: the final state
        outputs, final_state = tf.nn.static_rnn(
            cell=multi_cell,
            inputs=sliced_inputs,
            initial_state=self.initial_state)
        self.final_state = final_state

        with tf.name_scope('flatten_outputs'):
            flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

        with tf.name_scope('flatten_targets'):
            flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])

        with tf.variable_scope('softmax') as sm_vs:
            softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size])
            softmax_b = tf.get_variable('softmax_b', [vocab_size])
            self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=flat_targets)
            self.mean_loss = tf.reduce_mean(loss)

        with tf.name_scope('loss_montor'):
            count = tf.Variable(1.0, name='count')
            sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')

            self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                               count.assign(0.0), name='reset_loss_monitor')
            self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss + self.mean_loss),
                                                count.assign(count + 1), name='update_loss_monitor')

            with tf.control_dependencies([self.update_loss_monitor]):
                self.average_loss = sum_mean_loss / count
                self.ppl = tf.exp(self.average_loss)

            average_loss_summary = tf.summary.scalar(
                name='average loss', tensor=self.average_loss)
            ppl_summary = tf.summary.scalar(
                name='perplexity', tensor=self.ppl)

        self.summaries = tf.summary.merge(
            inputs=[average_loss_summary, ppl_summary], name='loss_monitor')

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))

        # self.learning_rate = tf.constant(learning_rate)
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        if is_training:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

```

## 总结
- 之前再调试github上的文本分类的代码的时候，明显感觉到代码清晰，模块化作的特别好，但是看这个poem text generation的代码的时候，感觉很糟心，一个init里面能写三页代码，很多操作都没有进行封装，embedding_layer, lstm_layer, loss_layer全都写在了init中，让人看得头痛，其中还不乏冗余的操作，如static_rnn的input，张量时序输入调整，直接用tf.unstack(input,tiem_step,axit=1)就可以了。
- flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])这段代码将最后一层的outputs在time_step维度进行拼接，展开的结果是[16*64,128]与<br>softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size])相乘得到矩阵[1024,vocab_size], 在这1024维中，0-16-32...是样本1的time_step输入。
- rnn,gru的state初始化比较简单，lstm的有state初始化不同：

```python
#part 1
with tf.name_scope('initial_state'):
    self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)
    if self.cell_type == 'rnn' or self.cell_type == 'gru':
        self.initial_state = tuple(
            [tf.placeholder(tf.float32,
                            [self.batch_size, multi_cell.state_size[idx]],
                            'initial_state_' + str(idx + 1)) for idx in range(self.num_layers)])
    elif self.cell_type == 'lstm':
        self.initial_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(
                tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size[idx][0]],
                                'initial_lstm_state_' + str(idx + 1)),
                tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size[idx][1]],
                                'initial_lstm_state_' + str(idx + 1)))
                for idx in range(self.num_layers)])

#part 2


if self.cell_type in ['rnn', 'gru']:
    state = self.zero_state.eval()
else:
    state = tuple([(
        np.zeros((self.batch_size, self.hidden_size)),
        np.zeros((self.batch_size, self.hidden_size))
        )for _ in range(self.num_layers)])


#part 3                   
    feed_dict = {self.input_data: x, self.targets: y, self.initial_state: state,
                    self.learning_rate: learning_rate}
```
- 
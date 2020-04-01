---
layout:     post
title:      Tidy Tensorflow
subtitle:   
date:       2020-12-31
author:     RJ
header-img: 
catalog: true
tags:
    - NLP
---
<p id = "build"></p>
---

## For what? 
记录Tensorflow的用法, 从源代码学习编程方法。

global_step:
 refers to the number of batches seen by the graph. Every time a batch is provided, the weights are updated in the direction that minimizes the loss. global_step just keeps track of the number of batches seen so far. When it is passed in the minimize() argument list, the variable is increased by one. 


## 参数解析

>parser = argparse.ArgumentParser()

>parser.add_argument('--do_train', action='store_true')

store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False。反之亦然，store_false也就是默认为True，一旦命令中有此参数，其值则变为False。


## tf.reduce_sum 维度缩减

```python

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()
a = [
    [1, 2, 3],
    [4, 5, 6]
]
b = tf.reduce_sum(a, axis=-1)
print(b)
得到:
[6,15]
```

## tf.tile 维度扩展

```python
a: [[1,2],[3,4],[5,6]]  #即3*2维度

tf.tile(a,[2,2])得到：

[   
    [1,2,1,2],[3,4,3,4],[5,6,5,6]

    [1,2,1,2],[3,4,3,4],[5,6,5,6]

]

#即tf.tile按照第二个参数，对原输入作相应维度的扩张
#应用场景： transformer 在对q,k,v进行split后，输出input的batch维度扩展成原来的num_heads倍，所以对相应mask需要扩倍
#利用tf.tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度100
mask = tf.tile(inputs, [num_heads, 1])
#将mask扩展成Q_的大小，需要在axis=1即time_step那个维度扩展，且那个维度也取与q,k,v相同的维度
keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

```

## tf.where
```python
#维度[batch_size * numHeads, queries_len, key_len]
maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,scaledSimilary)  
```

## 静态和动态维度
[reference](https://www.jianshu.com/p/75a903a44cf2)
- 使用tf.get_shape()获取静态维度
- 使用tf.shape获取动态维度

如果你的placeholder输入的维度都是固定的情况下，使用get_shape()。但是很多情况下，我们希望想训练得到的网络可以用于任意大小的图像，这时你的placeholder就的输入维度都是[None,None,None,color_dim]这样的，在这种情况下，后续网络中如果需要得到tensor的维度，则需要使用tf.shape。

## 模型保存相关

初始化saver
```python

xxxxxx weight histgram summary  xxxxxxxxx

    with tf.variable_scope("encode_fc-layer%d" % (l + 1)):
        input_data = tf.layers.dense(inputs=input_data, units=args.layers[l], activation=tf.nn.tanh
                                        , kernel_initializer=xavier_initializer()
                                        , kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight),
                                        name="layer%d" % (l + 1), reuse=tf.AUTO_REUSE)
        if self.flag:
            Weights = tf.get_default_graph().get_tensor_by_name(
                os.path.split(input_data.name)[0] + '/kernel:0')
            Biases = tf.get_default_graph().get_tensor_by_name(
                os.path.split(input_data.name)[0] + '/bias:0')
            #  写进日志
            tf.summary.histogram(os.path.split(input_data.name)[0] + "/weights",
                                    Weights)  # name命名，Weights赋值
            tf.summary.histogram(os.path.split(input_data.name)[0] + "/bias", Biases)
xxxxxx weight histgram summary  xxxxxxxxx

xxxxxx loss scalar summary  xxxxxxxxx

      tf.summary.scalar('loss_with_jacobian', self.loss_with_jacobian)
        self.train_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.train_optimizer_with_jacobian = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_with_jacobian)
xxxxxx loss scalar summary  xxxxxxxxx

------------------------------------------------------------------------

init = tf.global_variables_initializer()

self.merged = tf.summary.merge_all()
self.saver = tf.train.Saver(max_to_keep=50)
self.save_path = ""

self.sess.run(init)
self.summary_writer = tf.summary.FileWriter("logs/", graph=self.sess.graph)

------------------------------------------------------------------------

for it in range(args.num_jacobian_iters):
    start = time.time()

    feed_dict = {}
    feed_dict[self.x] = np.reshape(x_data, (args.trj_nums, (args.seq_length + 1), args.state_dim))
    if args.control_input: feed_dict[self.u] = u_data[:, :args.seq_length]

    # Find loss and execude training operation
    fetches = [self._jaco_bian_loss, self.loss_with_jacobian, self.loss, self.loss_reconstruction,
                self.train_optimizer_with_jacobian]
    try:
        _jaco_bian_loss, total_loss_with_jacobian, loss, rec_loss, _ = self.sess.run(fetches=fetches,
                                                                                        feed_dict=feed_dict)
        summary_str = self.sess.run(fetches=self.merged, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, it)


```


获取checkpoint的模型
```python
def get_checkpoint_id_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()[1:]
        d_list = [item.split(':')[1].strip().replace('\"', '') for item in sentences]
        print(id_list)
        return id_list

```

循环调用保存的模型查看参数和图像关联，分析原因：
```python
for item in id_list:
    sess = deep_koopman.sess
    deep_koopman.saver.restore(sess, 'models/'+item)
```

## Transformer代码

注意Q,K,V都是输入input_x， 三者经过dense返回经过权重矩阵x*W 的结果，Q_, K_, V_, 我们使用这Q_,K_矩阵计算相似度, 再与V_计算。也即是输入经过同维线性变换后，进行相似度计算。
```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

num_heads = 8
a = [
    [[0, 0], [2, 2], [3, 3]],
    [[4, 4], [5, 5], [6, 6]]
]
b = tf.reduce_sum(a, axis=-1)
print(b)
c = tf.abs(tf.reduce_sum(a, axis=-1))
print(c)
d = tf.sign(tf.abs(tf.reduce_sum(a, axis=-1)))
print(d)


key_masks = tf.tile(d, [num_heads, 1]) # (h*N, T_k)

key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(a)[1], 1]) # (h*N, T_q, T_k)

print(key_masks)

# outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
# paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
```


## activation

```python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Built-in activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export

# b/123041942
# In TF 2.x, if the `tf.nn.softmax` is used as an activation function in Keras
# layers, it gets serialized as 'softmax_v2' instead of 'softmax' as the
# internal method name is returned in serialization. This results in errors in
# model exporting and loading as Keras can't find any activation function with
# the name of `softmax_v2`.

# This dict maps the activation function name from its v2 version to its
# canonical name.
_TF_ACTIVATIONS_V2 = {
    'softmax_v2': 'softmax',
}


@keras_export('keras.activations.softmax')
def softmax(x, axis=-1):
  """The softmax activation function transforms the outputs so that all values are in

  range (0, 1) and sum to 1. It is often used as the activation for the last
  layer of a classification network because the result could be interpreted as
  a probability distribution. The softmax of x is calculated by
  exp(x)/tf.reduce_sum(exp(x)).

  Arguments:
      x : Input tensor.
      axis: Integer, axis along which the softmax normalization is applied.

  Returns:
      Tensor, output of softmax transformation (all values are non-negative
        and sum to 1).

  Raises:
      ValueError: In case `dim(x) == 1`.
  """
  ndim = K.ndim(x)
  if ndim == 2:
    return nn.softmax(x)
  elif ndim > 2:
    e = math_ops.exp(x - math_ops.reduce_max(x, axis=axis, keepdims=True))
    s = math_ops.reduce_sum(e, axis=axis, keepdims=True)
    return e / s
  else:
    raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                     'Received input: %s' % (x,))


@keras_export('keras.activations.elu')
def elu(x, alpha=1.0):
  """Exponential linear unit.

  Arguments:
      x: Input tensor.
      alpha: A scalar, slope of negative section.

  Returns:
      The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.

  Reference:
      - [Fast and Accurate Deep Network Learning by Exponential
        Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
  """
  return K.elu(x, alpha)


@keras_export('keras.activations.selu')
def selu(x):
  """Scaled Exponential Linear Unit (SELU).

  The Scaled Exponential Linear Unit (SELU) activation function is:
  `scale * x` if `x > 0` and `scale * alpha * (exp(x) - 1)` if `x < 0`
  where `alpha` and `scale` are pre-defined constants
  (`alpha = 1.67326324`
  and `scale = 1.05070098`).
  The SELU activation function multiplies  `scale` > 1 with the
  `[elu](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/elu)`
  (Exponential Linear Unit (ELU)) to ensure a slope larger than one
  for positive net inputs.

  The values of `alpha` and `scale` are
  chosen so that the mean and variance of the inputs are preserved
  between two consecutive layers as long as the weights are initialized
  correctly (see [`lecun_normal` initialization]
  (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_normal))
  and the number of inputs is "large enough"
  (see references for more information).

  ![](https://cdn-images-1.medium.com/max/1600/1*m0e8lZU_Zrkh4ESfQkY2Pw.png)
  (Courtesy: Blog on Towards DataScience at
  https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9)

  Example Usage:
  ```python3
  n_classes = 10 #10-class problem
  model = models.Sequential()
  model.add(Dense(64, kernel_initializer='lecun_normal', activation='selu',
  input_shape=(28, 28, 1))))
  model.add(Dense(32, kernel_initializer='lecun_normal', activation='selu'))
  model.add(Dense(16, kernel_initializer='lecun_normal', activation='selu'))
  model.add(Dense(n_classes, activation='softmax'))
  ```

  Arguments:
      x: A tensor or variable to compute the activation function for.

  Returns:
      The scaled exponential unit activation: `scale * elu(x, alpha)`.

  # Note
      - To be used together with the initialization "[lecun_normal]
      (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_normal)".
      - To be used together with the dropout variant "[AlphaDropout]
      (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout)".

  References:
      [Self-Normalizing Neural Networks (Klambauer et al, 2017)]
      (https://arxiv.org/abs/1706.02515)
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * K.elu(x, alpha)


@keras_export('keras.activations.softplus')
def softplus(x):
  """Softplus activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The softplus activation: `log(exp(x) + 1)`.
  """
  return nn.softplus(x)


@keras_export('keras.activations.softsign')
def softsign(x):
  """Softsign activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The softplus activation: `x / (abs(x) + 1)`.
  """
  return nn.softsign(x)


@keras_export('keras.activations.relu')
def relu(x, alpha=0., max_value=None, threshold=0):
  """Rectified Linear Unit.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = alpha * (x - threshold)` otherwise.

  Arguments:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

  Returns:
      A tensor.
  """
  return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)


@keras_export('keras.activations.tanh')
def tanh(x):
  """Hyperbolic Tangent activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The tanh activation: `tanh(x) = sinh(x)/cosh(x) = ((exp(x) -
      exp(-x))/(exp(x) + exp(-x)))`.
  """
  return nn.tanh(x)


@keras_export('keras.activations.sigmoid')
def sigmoid(x):
  """Sigmoid.

  Applies the sigmoid activation function. The sigmoid function is defined as
  1 divided by (1 + exp(-x)). It's curve is like an "S" and is like a smoothed
  version of the Heaviside (Unit Step Function) function. For small values
  (<-5) the sigmoid returns a value close to zero and for larger values (>5)
  the result of the function gets close to 1.
  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  Sigmoid activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The sigmoid activation: `(1.0 / (1.0 + exp(-x)))`.
  """
  return nn.sigmoid(x)


@keras_export('keras.activations.exponential')
def exponential(x):
  """Exponential activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The exponential activation: `exp(x)`.
  """
  return math_ops.exp(x)


@keras_export('keras.activations.hard_sigmoid')
def hard_sigmoid(x):
  """Hard sigmoid activation function.

  Faster to compute than sigmoid activation.

  Arguments:
      x: Input tensor.

  Returns:
      Hard sigmoid activation:
      - `0` if `x < -2.5`
      - `1` if `x > 2.5`
      - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
  """
  return K.hard_sigmoid(x)


@keras_export('keras.activations.linear')
def linear(x):
  """Linear activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The linear activation: `x`.
  """
  return x


@keras_export('keras.activations.serialize')
def serialize(activation):
  if activation.__name__ in _TF_ACTIVATIONS_V2:
    return _TF_ACTIVATIONS_V2[activation.__name__]
  return activation.__name__


@keras_export('keras.activations.deserialize')
def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='activation function')


@keras_export('keras.activations.get')
def get(identifier):
  if identifier is None:
    return linear
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'activation function identifier:', identifier)

```
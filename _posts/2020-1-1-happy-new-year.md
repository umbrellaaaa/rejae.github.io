---
layout:     post
title:      happy new year
subtitle:   
date:       2020-1-1
author:     RJ
header-img: 
catalog: true
tags:
    - life

---
<p id = "build"></p>
---

## 前言
告别了2019，全身心的投入2020吧！

今年应该是腾飞的一年，收货也将是最大的一年，这一年的时间当然应该更有价值和意义。

今天应该就换新手机华为Mate30了，所以那些游戏也离我远去了，我应该拿这个手机去做更有价值和意义的事情，我的自然语言处理专业，我要全心全意的投入这个领域。

- 词法分析
- 句法分析
- 语义分析
- 篇章分析
- NER
- SRL
- 文本纠错
- **文本生成**
- KG
- seq2seq
- bert&albert
- xlnet
- pytorch
- 图像处理
- 搜索算法
- 推荐算法

看看，还有那么多事情要做呢！

## 今日学习

[Tencent cloud tensorflow](https://cloud.tencent.com/developer/labs/lab/10324/console)

```python
import tensorflow as tf

a = tf.constant([1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0],dtype=tf.float32,shape=[1,5,5,1])
b = tf.constant([1,0,1,0,1,0,1,0,1],dtype=tf.float32,shape=[3,3,1,1])
c = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='VALID')
d = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    print ("c shape:")
    print (c.shape)
    print ("c value:")
    print (sess.run(c))
    print ("d shape:")
    print (d.shape)
    print ("d value:")
    print (sess.run(d))
```


```python
import tensorflow as tf

a = tf.constant([1,-2,0,4,-5,6])
b = tf.nn.relu(a)
with tf.Session() as sess:
    print (sess.run(b))
```

```python
import tensorflow as tf

a = tf.constant([1,3,2,1,2,9,1,1,1,3,2,3,5,6,1,2],dtype=tf.float32,shape=[1,4,4,1])
b = tf.nn.max_pool(a,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
c = tf.nn.max_pool(a,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    print ("b shape:")
    print (b.shape)
    print ("b value:")
    print (sess.run(b))
    print ("c shape:")
    print (c.shape)
    print ("c value:")
    print (sess.run(c))
```

```python
import tensorflow as tf

a = tf.constant([1,2,3,4,5,6],shape=[2,3],dtype=tf.float32)
b = tf.placeholder(tf.float32)
c = tf.nn.dropout(a,b,[2,1],1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(c,feed_dict={b:0.75}))
```

```python
import tensorflow as tf
x = tf.constant([1,2,3,4,5,6,7],dtype=tf.float64)
y = tf.constant([1,1,1,0,0,1,0],dtype=tf.float64)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y,logits = x)
with tf.Session() as sess:
    print (sess.run(loss))
```

```python
#产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]。
import tensorflow as tf
initial = tf.truncated_normal(shape=[3,3], mean=0, stddev=1)
print(tf.Session().run(initial))
```

```python
#!/usr/bin/python

import tensorflow as tf
import numpy as np
a = tf.constant([1,2,3,4,5,6],shape=[2,3])
b = tf.constant(-1,shape=[3,2])
c = tf.matmul(a,b)

e = tf.constant(np.arange(1,13,dtype=np.int32),shape=[2,2,3])
f = tf.constant(np.arange(13,25,dtype=np.int32),shape=[2,3,2])
g = tf.matmul(e,f)
with tf.Session() as sess:
    print (sess.run(a))
    print ("##################################")
    print (sess.run(b))
    print ("##################################")
    print (sess.run(c))
    print ("##################################")
    print (sess.run(e))
    print ("##################################")
    print (sess.run(f))
    print ("##################################")
    print (sess.run(g))
```

```python
#!/usr/bin/python

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[None,3])
y = tf.matmul(x,x)
with tf.Session() as sess:
    rand_array = np.random.rand(3,3)
    print(sess.run(y,feed_dict={x:rand_array}))
```

```python
#!/usr/bin/python

import tensorflow as tf
import numpy as np

a = tf.constant([[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]])
b = tf.constant([2.0,1.0])
c = tf.constant([1.0])
sess = tf.Session()
print (sess.run(tf.nn.bias_add(a, b)))
#print (sess.run(tf.nn.bias_add(a,c))) error
print ("##################################")
print (sess.run(tf.add(a, b)))
print ("##################################")
print (sess.run(tf.add(a, c)))
```

```python
#!/usr/bin/python

import tensorflow as tf
import numpy as np

initial = [[1.,1.],[2.,2.]]
x = tf.Variable(initial,dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_mean(x)))
    print(sess.run(tf.reduce_mean(x,0))) #Column
    print(sess.run(tf.reduce_mean(x,1))) #row
```

```python
#!/usr/bin/python

import tensorflow as tf
import numpy as np

initial_x = [[1.,1.],[2.,2.]]
x = tf.Variable(initial_x,dtype=tf.float32)
initial_y = [[3.,3.],[4.,4.]]
y = tf.Variable(initial_y,dtype=tf.float32)
diff = tf.squared_difference(x,y)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(diff))
```

```python
#!/usr/bin/python

import tensorflow as tf
initial = tf.truncated_normal(shape=[10,10],mean=0,stddev=1)
W=tf.Variable(initial)
list = [[1.,1.],[2.,2.]]
X = tf.Variable(list,dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print ("##################(1)################")
    print (sess.run(W))
    print ("##################(2)################")
    print (sess.run(W[:2,:2]))
    op = W[:2,:2].assign(22.*tf.ones((2,2)))
    print ("###################(3)###############")
    print (sess.run(op))
    print ("###################(4)###############")
    print (W.eval(sess)) #computes and returns the value of this variable
    print ("####################(5)##############")
    print (W.eval())  #Usage with the default session
    print ("#####################(6)#############")
    print (W.dtype)
    print (sess.run(W.initial_value))
    print (sess.run(W.op))
    print (W.shape)
    print ("###################(7)###############")
    print (sess.run(X))
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
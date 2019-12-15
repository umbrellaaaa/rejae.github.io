---
layout:     post
title:      机器学习模型的部署
subtitle:   summary
date:       2019-11-6
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

<h1> 机器学习模型的部署</h1>
采用Flask-RESTful构建

Flask-RESTful 是一个 Flask 扩展，它添加了快速构建 REST APIs 的支持。它当然也是一个能够跟你现有的ORM/库协同工作的轻量级的扩展。Flask-RESTful 鼓励以最小设置的最佳实践。如果你熟悉 Flask 的话，Flask-RESTful 应该很容易上手。

## 示例代码1
```python
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/')


if __name__ == '__main__':
    app.run(debug=True)

out:
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
在浏览器中，显示：
{
    "hello": "world"
}



## 示例代码2

```python
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

TODOS = {
    'todo1': {'task': 'build an API'},
    'todo2': {'task': '?????'},
    'todo3': {'task': 'profit!'},
}


def abort_if_todo_doesnt_exist(todo_id):
    """Abort request if todo_id does not exist in TODOS"""
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('task')


# Todo
# shows a single todo item and lets you updatae or delete a todo item
class Todo(Resource):
    def get(self, todo_id):
        """Return the specified todo item given the todo_id

        Example:
            # In the terminal
            $ curl http://localhost:5000/todos/todo1

            OR

            # Python
            requests.get('http://localhost:5000/todos/todo1').json()
        """
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, todo_id):
        """Deletes an existing task

        Example:
            # In the terminal
            $ curl http://localhost:5000/todos/todo1 -X DELETE -v

            OR

            # Python
            requests.delete('http://localhost:5000/todos/todo4')
        """
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        # 204: SUCCESS; NO FURTHER CONTENT
        return '', 204

    def put(self, todo_id):
        """Updates existing task

        Example:
            # In the terminal
            $ curl http://localhost:5000/todos/todo1 -d "task=Remember the milk" -X PUT -v

            OR

            # Python
            requests.put('http://localhost:5000/todos/todo1',
                         data={'task': 'Remember the milk'}).json()
        """

        # parser
        abort_if_todo_doesnt_exist(todo_id)
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[todo_id] = task
        # 201: CREATED
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        """Return the current TODO dictionary

        Example:
            # In the terminal
            $ curl http://localhost:5000/todos

            OR

            # Python
            requests.get('http://localhost:5000/todos').json()
        """
        return TODOS

    def post(self):
        """Adds task to TODO

        Example:
            # In the terminal
            $ curl http://localhost:5000/todos -d "task=Remember the milk" -X POST -v

            OR

            # Python
            requests.post('http://localhost:5000/todos',
                         data={'task': 'Remember the milk'}).json()
        """
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        # 201: CREATED
        return TODOS[todo_id], 201


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')

if __name__ == '__main__':
    app.run(debug=True)

```

http://localhost:5000/todos 为Index页面，返回TODOS操作选项：

```python
{
    "todo1": {
        "task": "build an API"
    },
    "todo2": {
        "task": "?????"
    },
    "todo3": {
        "task": "profit!"
    }
}
```

http://localhost:5000/todos/todo1:

```python
{
    "task": "build an API"
}
```

## 部署模型实战

这里我们部署一个NLP情感分类项目：

### model.py
```python
# ML imports
from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.ensemble import RandomForestClassifier
import pickle

from util import plot_roc
# spacy_tok


class NLPModel(object):

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = MultinomialNB()
        # self.vectorizer = TfidfVectorizer(tokenizer=spacy_tok)
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        """Fits a TFIDF vectorizer to the text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='lib/models/TFIDFVectorizer.pkl'):
        """Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='lib/models/SentimentClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

    def plot_roc(self, X, y, size_x, size_y):
        """Plot the ROC curve for X_test and y_test.
        """
        plot_roc(self.clf, X, y, size_x, size_y)

```

### util.py

```python
# import spacys
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# nlp = spacy.load('en')


# def spacy_tok(text, lemmatize=False):
#     doc = nlp(text)
#     if lemmatize:
#         tokens = [tok.lemma_ for tok in doc]
#     else:
#         tokens = [tok.text for tok in doc]
#     return tokens


def plot_roc(model, x_columns, y_true, size_x=12, size_y=12):
    """Returns a ROC plot

    Forked from Matt Drury.
    """

    y_pred = model.predict_proba(x_columns)

    fpr, tpr, threshold = roc_curve(y_true, y_pred[:, 1])
    area_under_curve = auc(fpr, tpr)

    # method I: plt
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    model_name = str(type(model)).split('.')[-1].strip(">\'")
    plt.title(f'{model_name} ROC')
    ax.plot(fpr, tpr, 'k', label='AUC = %0.3f' % area_under_curve)

    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

```

### build_model.py
```python
from model import NLPModel
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_model():
    model = NLPModel()

    with open('./lib/data/train.tsv') as f:
        data = pd.read_csv(f, sep='\t')

    pos_neg = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]

    pos_neg['Binary'] = pos_neg.apply(
        lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1)

    model.vectorizer_fit(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer fit complete')

    X = model.vectorizer_transform(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer transform complete')
    y = pos_neg.loc[:, 'Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()

    model.plot_roc(X_test, y_test, size_x=12, size_y=12)


if __name__ == "__main__":
    build_model()


```

### app.py

```python
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel

app = Flask(__name__)
api = Api(app)

model = NLPModel()

clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)

```

访问网页：http://127.0.0.1:5000/?query=that+movie+was+nice <br>
返回：{
    "prediction": "Positive",
    "confidence": 0.71
}

调用：
```python
import requests
url = 'http://127.0.0.1:5000/'
params ={'query': 'that movie was boring'}
response = requests.get(url, params)
print(response.url)
print(response.json())
```

## 总结
1. 模型加载：
```python
clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)
```
2. 输入数据的处理：
```python
        user_query = args['query']
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
```
3. 调用模型方法：
```python
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)
```
4. 返回相应结果：
```python
        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output
```


## AI算法基本流程：

1. 数据准备 & 预处理  ==>>  trian  val  test  文件
2. 配置文件 hparams.py
3. 模型组件 modules.py 相应核心方法
4. 模型构建 models.py Transformer/CNN/RNN
5. 工具方法 utils.py
6. 模型训练 train.py 

--------------------------------------

配置文件 hparams.py
```python
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train1', default='iwslt2016/segmented/train.de.bpe',
                             help="german training segmented data")
    parser.add_argument('--train2', default='iwslt2016/segmented/train.en.bpe',
                             help="english training segmented data")
    parser.add_argument('--eval1', default='iwslt2016/segmented/eval.de.bpe',
                             help="german evaluation segmented data")
    parser.add_argument('--eval2', default='iwslt2016/segmented/eval.en.bpe',
                             help="english evaluation segmented data")
    parser.add_argument('--eval3', default='iwslt2016/prepro/eval.en',
                             help="english evaluation unsegmented data")

    ## vocabulary
    parser.add_argument('--vocab', default='iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")
```

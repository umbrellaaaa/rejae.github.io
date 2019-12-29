---
layout:     post
title:      python-summary
subtitle:   summary
date:       2019-11-4
author:     RJ
header-img: 
catalog: true
tags:
    - Python

---
<p id = "build"></p>
---


## 基础数据类型的常用操作

### dict
dict2 = dict.update(dict1)

```python
with open('vocab/han_vocab.json', "r", encoding='utf-8') as f:
    han_dict_w2id = json.load(f)
    han_dict_w2id = defaultdict_from_dict(han_dict_w2id)
han_dict_id2w = {v: k for k, v in han_dict_w2id.items()}
```

### list



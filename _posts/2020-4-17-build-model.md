---
layout:     post
title:      Build Model
subtitle:   
date:       2020-4-17
author:     RJ
header-img: 
catalog: true
tags:
    - nlp

---
<p id = "build"></p>
---










## 配置config文件

```python
import argparse
def config_poem_train(args=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default='./data/poem/',help='data path')
    args = parser.parse_args(args.split()) # 
    return args

args = '--output_dir output_poem --data_path ./data/poem/  --hidden_size 128 --embedding_size 128 --cell_type lstm'

```

```python
if __name__ == '__main__':
    args = '--output_dir output_poem --data_path ./data/poem/  --hidden_size 128 --embedding_size 128 --cell_type lstm'
    main(args)
```


其中add_argparse种的参数包括：
- '--name'
- type=
- default=
- help=
- action='store_true' #表示sh中有这个参数就为 true, 否则为false
- dest='debug' 

## 
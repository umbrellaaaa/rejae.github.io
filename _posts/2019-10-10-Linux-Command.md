---
layout:     post
title:      Linux Command
subtitle:   cheer up
date:       2019-10-10
author:     RJ
header-img: 
catalog: true
tags:
    - Linux

---
<p id = "build"></p>
---

download sh

mkdir iwslt2016 | 
wget -qO- --show-progress https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz | 
tar xz; mv de-en iwslt2016



## 系统相关

uname -a 查看系统版本  cat /proc/version

ps -le 查看所有进程

kill -9 具体的PID
 
nvidia-smi

watch -n 10 nvidia-smi

- yum -y install wget
- wget -c http://www.openslr.org/resources/18/data_thchs30.tgz


## 文件操作

unzip filename.zip

filename.tar.gz的解压:   tar -zxvf filename.tar.gz

z: 　　gzip  　　　　　　　　    压缩格式

x: 　　extract　　　　　　　　  解压

v:　　 verbose　　　　　　　　详细信息

f: 　　file(file=archieve)　　　　文件

filename.tar.bz2的解压:  tar -jxvf filename.tar.bz2

j: 　　bzip2　　　　　　　　　 压缩格式


rm -rf 目录名字  -r 就是向下递归，不管有多少级目录，一并删除   -f 就是直接强行删除，不作任何提示的意思

[file operation](https://blog.csdn.net/linjpg/article/details/77854671)

## 环境变量

vim /etc/profile

export PATH=/root/anaconda3/bin:$PATH

source /etc/profile  生效
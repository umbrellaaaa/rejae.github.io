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

- yum -y install wget
- wget -c http://www.openslr.org/resources/18/data_thchs30.tgz
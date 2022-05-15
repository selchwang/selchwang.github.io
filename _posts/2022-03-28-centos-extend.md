---
title: "CentOS根目录扩容"
date: 2022-03-28T08:14:00
excerpt_separator: "<!--more-->"
share: false
categories:
  - developer
tags:
  - CentOS
  - Linux
---

我大概是把自己活成了运维boy，周一挤早高峰到实验室后发现常挂的服务果然挂了，mongodb的docker起不来，`df -Th`一下发现`/`目录满了……当时内心是崩溃的。所幸查到了[解决方案](https://blog.csdn.net/qq_24871519/article/details/86243571)，姑且简单记录一下以备将来需要。建议还是看详细原文，这个仅供我本人参考……

1. 创建新分区

   这里我的`/dev/sdb`是已经分区过的，若是新的大概还需要`mklabel`创建gpt label。

   ```bash
   # bash
   fdisk /dev/sdb
   ```

   ```
   # fdisk
   Command (m for help): n
   Partition number (2-128, default 2):
   First sector (34-125015424990, default 1953124352):
   Last sector, +sectors or +size{K,M,G,T,P} (1953124352-125015424990, default 125015424990): +500G
   Created partition 2
   ```

   小插曲是，`fdisk`里直接`n`之后`w`，提示错误，`WARNING: Re-reading the partition table failed with error 16: Device or resource busy`，运行命令`partprobe`即可解决，`lsblk`可以看到新分区已经创建好。

   ```bash
   # bash
   partprobe
   lsblk
   ```

2. 扩容

   ```bash
   # bash
   lvm
   ```

   ```
   # lvm
   vgextend centos_server11 /dev/sdb2
   lvextend -l +100%FREE /dev/centos_server11/root
   ```

   我这里名称是`centos_server11`，参考的博客里说的是`centos`。

3. 同步到文件系统

   ```bash
   # bash
   xfs_growfs /dev/centos_server11/root
   ```

如此便解决了本周第一个恶心事😅


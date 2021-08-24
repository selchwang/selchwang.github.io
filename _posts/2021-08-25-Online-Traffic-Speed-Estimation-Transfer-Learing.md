---
layout:     post
title:      "论文浅尝 | Online Traffic Speed Estimation for Urban Road Networks with Few Data: A Transfer Learning Approach"
date:       2021-08-25 1:28:00
author:     "Selch"
header-style: text
catalog: true
tags:
  - 交通速度预测
  - 迁移学习
  - 图卷积网络
---

# Online Traffic Speed Estimation for Urban Road Networks with Few Data: A Transfer Learning Approach

- 问题：基于机器学习的预测方法依赖大量数据，数据不易得；
- 方案：迁移学习
  - 图卷积生成自动编码器（graph convolutional generative
    autoencoder，GCGA），修改其计算图以减少参数规模；
  - 用足量数据训练出预训练模型，将其复用到数据缺乏的网络中，只需要调整少量参数。
  - 实验：预测效果好，训练时间短。

## Introduction

### 逻辑

1. 交通速度预测很重要。
2. 已有许多解决方案，**工业界和学术界**。
   1. 工业界如谷歌、优步，汇聚实时数据，构建交通速度地图。
   2. 学术界用基于模型的，或黑箱统计的方法，基于相对缺乏的数据；模型透明、可解释，黑箱抗干扰，**已可以与工业方法媲美**。
   3. **【转折】**上述方法集中于少量道路，在大规模网络中计算低效；未能有效利用路网结构信息。
   4. **【承接】**GCGA：图卷积操作，利用了路网结构；生成式对抗设计，一次生成整个路网的速度估计。
3. **【转折】**黑箱方法依赖大量数据，不易得。
4. 本文提出**基于GCGA**的**迁移学习驱动**的城市道路网络在线交通速度估计方法。考虑历史数据特征，从GCGA中识别出拓扑特征参数。先利用充足的其他城市的数据预训练；再选取参数，利用目标城市数据，进一步调优。

------

## 一些英语

- plethora n. 过多，过量，过剩 Their accomplishments are established on **a plethora of** historical GPS records.
- much effort has **been devoted to developing** solutions for providing such information.

- scarce adj. 缺乏的，不足的，稀少的

- interpret v. 诠释，说明
- perturbation n. 扰动，摄动
- hinder vt. 阻碍 vi. 成为阻碍

---
layout:     post
title:      "论文浅尝 | Online Traffic Speed Estimation for Urban Road Networks with Few Data: A Transfer Learning Approach"
date:       2021-08-25 1:28:00
author:     "Selch"
header-style: text
mathjax: true
catalog: true
tags:
  - 论文浅尝
  - 交通速度预测
  - 迁移学习
  - 图卷积网络
---

- 问题：基于机器学习的预测方法依赖大量数据，数据不易得；
- 方案：迁移学习
  - 图卷积生成自动编码器（graph convolutional generative
    autoencoder，GCGA），修改其计算图以减少参数规模；
  - 用足量数据训练出预训练模型，将其复用到数据缺乏的网络中，只需要调整少量参数。
  - 实验：预测效果好，训练时间短。

## I. Introduction

1. 交通速度预测很重要。
2. 已有许多解决方案，**工业界和学术界**。
   1. 工业界如谷歌、优步，汇聚实时数据，构建交通速度地图。
   2. 学术界用基于模型的，或黑箱统计的方法，基于相对缺乏的数据；模型透明、可解释，黑箱抗干扰，**已可以与工业方法媲美**。
   3. **【转折】**上述方法集中于少量道路，在大规模网络中计算低效；未能有效利用路网结构信息。
   4. **【承接】**GCGA：图卷积操作，利用了路网结构；生成式对抗设计，一次生成整个路网的速度估计。
3. **【转折】**黑箱方法依赖大量数据，不易得。
4. 本文提出**基于GCGA**的**迁移学习驱动**的城市道路网络在线交通速度估计方法。考虑历史数据特征，从GCGA中识别出拓扑特征参数。先利用充足的其他城市的数据预训练；再选取参数，利用目标城市数据，进一步调优。

## II. Traffic Speed Estimation for Urban Road Networks （建模）

### 路网模型

有向图G表示路网，路口集合N，构成路口的道路集合E：
$$
\mathcal{G}=(\mathcal{N}, \mathcal{E})
$$
序列T表示过去的时刻，t=0时即为当前时刻：
$$
\mathcal{T}=\{\ldots, -2, -1, 0\}
$$
速度用v表示，某路口e在t时刻的平均交通速度：$\mathcal{v}_{e,t}$；其他非时变量用P表示，某路口e的非时变属性：$\mathcal{P}_e$。

### 问题

获取城市路网的速度地图：
$$
\mathcal{V}_0=\{\mathcal{v}_{e,0}|\forall e\in \mathcal{E}\}
$$
速度值来源于两部分，其一是道路上的测速传感器，其二是根据车辆的GPS数据构建出来的；将有传感器的路口表示为$\mathcal{E}^\mathcal{S}$；将无传感器的，可以由GPS数据计算出t时刻速度的路口表示为$\mathcal{E}^+_t$；则在t时刻未知的是：
$$
\mathcal{E}^-_t=\mathcal{E} \setminus (\mathcal{E}^S \cup \mathcal{E}^+_t)
$$
城市路网交通速度估计问题表示为：
$$
\mathbf {minimize \ MAPE}={1\over|\mathcal{E}^-_t|}\sum_{e\in \mathcal{E}^-_t} {|\hat v _ {e,0} - \mathcal{v_{e,0}}| \over \mathcal{v_{e,0}+\varepsilon}}
$$

其中$\varepsilon$为一个小正数以避免分母为0的情况，文中取值为0.01km/h；$\hat v _ {e,0}$为估计速度：
$$
\mathcal{\hat V}_0=\mathbf {EST}(\mathcal{V}_t^+,\{\mathcal{P_e}|\forall e\in \mathcal{E}\})
\\
\mathcal{\hat V}_0=\{\hat v_{e,0}|\forall e \in \mathcal{E} \}
\\
\mathcal{V}^+_t=\{ \mathcal{v}_{e,t}|\forall e \in \mathcal{E}^S \cup \mathcal{E}^+_t\}
$$

其中EST为速度估计函数（speed estimator）。

## III. Transferable Graph Convolution Generative Autoencoder

1. 引入一种基于图的卷积生成神经网络 ➡ 生成城市道路网络的速度估计数据；

2. 研究神经网络中潜在的信息传递机制 ➡ 减少计算量；

3. 介绍模型训练和测试。

### A. 图卷积操作

#### 基础性介绍：

1. 引出对图卷积的一个基本认识：对某节点，与其相邻的节点对其影响更大，距离越远影响越小且在传播过程中迅速衰减 ➡ 输出的图中，任意节点反映了其周围的特征；

2. 对多序列的图卷积操作，由于自乘邻居矩阵，可以有更大的数据融合范围。

#### 具体的做法：

这里先提了一下文献12，说是采用了其中的图卷积操作，区别在于更关注道路的情况，也就是边的情况（文献12应该是关注于节点，也就是路口）。所以先做了一个图的转化，转为了**道路连通图（road-connectivity graph）**，顶点是道路，边是连通性。新的无向图$\mathcal{R}(\mathcal{E},\mathcal{A})$定义为：

$$
\mathcal{A} = \{ (e_1, e_2) | \big | \{ {\rm fr} (e_1), {\rm to} (e_1) \} \cap \{ {\rm fr} (e_2), {\rm to} (e_2) \} \big | \neq0 , \forall e_1, e_2 \in \mathcal{E} \}
$$

理解一下就是说，两个路的连通与否取决于他们是否有公共路口： ${\rm fr}(e)$ 和 ${\rm to}(e)$ 分别表示道路e的起始、终止路口，分别取两个道路两端的路口集合，如果有公共路口，则认为他们相连通。

随后是图卷积操作。输入为$\mathcal{E}$条道路的$|\mathcal{P}|+1$个特征（属性和速度），然后根据下面的传播规则得到F个潜在的数据特征，其中A是$\mathcal{R}$的邻接矩阵：

$$
Z={\rm GC}(X,A)=\sigma (\hat D^{-{1\over2}} \hat A \hat D^{-{1\over2}} XW + b)
$$

其中
$$
X\in \mathbb R ^ {|\mathcal{E}| \times (|\mathcal{P}|+1)}
\\
Z\in \mathbb R ^ {|\mathcal{E}| \times F}
\\
\hat A\in \mathbb B ^ {|\mathcal{E}| \times |\mathcal{E}|} = \{\hat a_{ij}\} = A + I
\\
\hat D = diag\{\hat a_{11}, \hat a_{22}, \dots \}
$$

非线性激活函数$\sigma (\cdot)$，优化出的参数是$W\in \mathbb R ^ {(|\mathcal{P}|+1) \times F}$，以及$b \in \mathbb R ^{|\mathcal{E}| \times F}$。

然后提了一句：

> This propagation rule is motivated by the first-order approximation of Chebyshev polynomials of eigenvalues in the spectral domain [14]. It has been shown that such approximation can result in highly competitive performance for graph feature learning [12], [13].

### B. 图卷积生成自动编码器



## IV. Case Studies

1. 调研潜在信息转移机制（latent information transfer mechanism）的性能和计算速度的提升；

2. 评估控制参数的影响，估计GCGA模型结构对转移效果的影响。

### A. Data Sets and Simulation Configurations

选取北上广深四个城市的数据集

- 北京（1386条路）2018全年数据作为预训练数据集

- 上广深（1522、1024、400条路）2018年11月数据作为独立训练数据集（直接训练，作为对照组）

- 取$\alpha=15\%$的数据，模拟数据量很少的实际情况

- 每个数据集在**时间维度**上划分三个不相互覆盖的子集：

  - 50% 网络参数调优

  - 25% 训练进程终止？（training process termination）

  - 25% MAPE性能评估，在被移除的$\alpha=15\%$的道路上

### B. Estimation Accuracy and Training Time

- 小样本城市独立训练 vs 有北京数据预训练模型辅助的训练

- 将30天（北京的是365天）的数据取平均，得到一天中每五分钟数据点

- 效果还是比较显著的，MAPE明显降低很多，而且越是小样本下越显著

![](/img/online-traffic-speed-transfer-1.PNG)

- 同时带来训练时间的减少

![](/img/online-traffic-speed-transfer-2.PNG)

### C. Parameter Sensitivity Test and Model Structure

这里就对比了一下模型参数不同取值的MAPE和训练时间，说了下为什么之前的选择是最好的。

## V. Conclusions

就没说啥有价值的东西了…

------

## 一些英语

- plethora n. 过多，过量，过剩 Their accomplishments are established on **a plethora of** historical GPS records.
- much effort has **been devoted to developing** solutions for providing such information.
- scarce adj. 缺乏的，不足的，稀少的
- interpret v. 诠释，说明
- perturbation n. 扰动，摄动
- hinder vt. 阻碍 vi. 成为阻碍
- prominent adj. 突出的，显著的，杰出的 the most prominent objective
- latent adj. 潜在的，隐藏的，潜伏的
- decay v. 衰减
- diffuse v. 散布，传播 diffuse over the network
- manipulation n. 操作 **manipulation agent**
- two sets of tunable parameters are adopted

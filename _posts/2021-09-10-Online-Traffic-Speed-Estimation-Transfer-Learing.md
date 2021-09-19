---
layout:     post
title:      "论文浅尝 | Online Traffic Speed Estimation for Urban Road Networks with Few Data: A Transfer Learning Approach"
date:       2021-09-10 1:28:00
author:     "Selch"
header-style: text
mathjax: true
catalog: true
tags:
  - 论文浅尝
  - 交通速度预测
  - 迁移学习
  - 图卷积网络
  - 生成式对抗网络
  - GCN
  - GAN
---

## Abstract

- 问题：基于机器学习的预测方法依赖大量数据，数据不易得；
- 方案：迁移学习
  - 图卷积生成自动编码器（graph convolutional generative autoencoder，GCGA），修改其计算图以减少参数规模；
  - 用足量数据训练出预训练模型，将其复用到数据缺乏的网络中，只需要调整少量参数；
  - 实验：预测效果好，训练时间短。

## I. Introduction

1. 交通速度预测很重要。
2. 已有许多解决方案，工业界和学术界。
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

#### 基础性介绍

1. 引出对图卷积的一个基本认识：对某节点，与其相邻的节点对其影响更大，距离越远影响越小且在传播过程中迅速衰减 ➡ 输出的图中，任意节点反映了其周围的特征；

2. 对多序列的图卷积操作，由于自乘邻居矩阵，可以有更大的数据融合范围。

#### 具体的做法

这里先提了一下文献12，说是采用了其中的图卷积操作，区别在于更关注道路的情况，也就是边的情况（文献12应该是关注于节点，也就是路口）。所以先做了一个图的转化，转为了**道路连通图（road-connectivity graph）**，顶点是道路，边是连通性。新的无向图$\mathcal{R}(\mathcal{E},\mathcal{A})$定义为：

$$
\mathcal{A} = \{ (e_1, e_2) \vert \big \vert \{ {\rm fr} (e_1), {\rm to} (e_1) \} \cap \{ {\rm fr} (e_2), {\rm to} (e_2) \} \big \vert \neq 0 , \forall e_1, e_2 \in \mathcal{E} \}
$$

理解一下就是说，两个路的连通与否取决于他们是否有公共路口： ${\rm fr}(e)$ 和 ${\rm to}(e)$ 分别表示道路e的起始、终止路口，分别取两个道路两端的路口集合，如果有公共路口，则认为他们相连通。

随后是图卷积操作。输入为$\mathcal{E}$条道路的$\vert\mathcal{P}\vert+1$个特征（属性和速度），然后根据下面的传播规则得到F个潜在的数据特征，其中A是$\mathcal{R}$的邻接矩阵：

$$
Z={\rm GC}(X,A)= \sigma (\hat D ^ {-{1 \over 2}} \hat A \hat D ^ {-{1 \over 2}} XW + b)
$$

其中$X\in \mathbb R ^ {\vert\mathcal{E}\vert \times (\vert\mathcal{P}\vert+1)}$，$Z\in \mathbb R ^ {\vert\mathcal{E}\vert \times F}$，$\hat A\in \mathbb B ^ {\vert\mathcal{E}\vert \times \vert\mathcal{E}\vert} = \{\hat a_{ij}\} = A + I$，$\hat D = diag\{\hat a_{11}, \hat a_{22}, \dots \}$。非线性激活函数$\sigma (\cdot)$，优化出的参数是$W\in \mathbb R ^ {(\vert\mathcal{P}\vert+1) \times F}$，以及$b \in \mathbb R ^{\vert\mathcal{E}\vert \times F}$。

然后提了一句：

> This propagation rule is motivated by the first-order approximation of Chebyshev polynomials of eigenvalues in the spectral domain [14]. It has been shown that such approximation can result in highly competitive performance for graph feature learning [12], [13].

### B. 图卷积生成自动编码器

模型是这样式儿的：

![](/img/online-traffic-speed-transfer-3.PNG)

#### 基于自动编码器的图特征生成器

基于当前可获取的数据$\mathcal{V}^+_0$和$\{\mathcal{P}_e \vert \forall e \in \mathcal{E}\}$，产出$\hat {\mathcal{V}}_0$。这里具体的特征选取包括：道路长、宽、限速值、车道数量、沿路POI数量。

初始状态将缺失的数据置零。先是三个连续的上采样，对每条路产生128、256、512个特征值；然后连续三个下采样，将特征值减少到256、128、1。最后输出的是归一化的速度值。

#### 图特征鉴别器

鉴别器评估生成的速度值的真实性，基于速度数据$\mathcal{V}_t^+, \forall t \in \mathcal{T}$的特征先验知识。

这个网络是一组128特征图卷积层，以及三个全链接层（1024、128、1）。最后输出true or false。

#### 目标函数

$$
\min_{\theta ^ G} \max_{\theta ^ D} L = \mathbb E [\log D(\mathcal{V_0^+}, \theta ^ D)] + \mathbb E [\log (1 - D(G((\mathcal{V_0^+}, \theta ^ D)), \theta ^ D))]
$$

其中$\theta ^ G$和$\theta ^ D$分别是生成器和鉴别器的参数集，$G(\cdot, \theta ^ G)$和$D(\cdot, \theta ^ D)$分别是损失函数。

初始化将$\theta ^ G$设置为随机值，使鉴别器很容易拒绝，导致L值较大。训练算法调整两个参数集生成、鉴别“真实数据”，重复训练过程直到生成器可以成功骗过鉴别器。

经过训练后，$\theta ^G$可以被用于估计城市路网速度数据。

### C. 潜在信息迁移

首先在说为什么$\theta ^G$可以迁移。

一方面，上述传播规则中得到的$W\in \mathbb R ^ {(\vert\mathcal{P}\vert+1) \times F}$是与道路规模$\vert \mathcal{E} \vert$无关的。

另一方面，$b \in \mathbb R ^{\vert\mathcal{E}\vert \times F}$与道路结构相关。为减轻二次训练的负担，将参数b分解：

$$
b \equiv B \kappa \mu
$$

其中$B \in \mathbb R ^ {\vert \mathcal{E} \vert \times K}$，$\kappa \in \mathbb R ^ {K \times F}$，$\mu^{F \times F}$。与F类似，此处的K是用户定义的超参数，用于定义子参数集的大小。这样与路网相关的就只有参数B，在小K值下训练更加高效。图卷积操作中的传播规则被修正为：

$$
Z={\rm GC}(X,A)= \sigma (\hat D ^ {-{1 \over 2}} \hat A \hat D ^ {-{1 \over 2}} XW + B \kappa \mu)
$$

### D. 模型训练和迁移学习

定义：

- 预训练数据：稠密的数据

- 训练数据：稀疏的数据

数据集采样操作：随机移除一些道路的数据。从$\mathcal{V}_t^+$中采样M次，得到$\mathcal{V}_{t,m}^-$，其中m是数据集的序号索引。

随后用预训练数据对生成器、鉴别器进行训练。每次迭代中，输入$\mathcal{V}_{t,m}^{-}$到生成器，得到$\hat {\mathcal{V}}_{t,m}^{-}$，作为对$\mathcal{V}_{t}^{+}$的一次估计。

**生成器的损失函数采用平均MSE：**

$$
L^G = {1 \over \vert \mathcal{T} \vert M} \sum_{t \in \mathcal{T}} \sum_{m=1}^M {\rm MSE}_{t,m}
$$

**鉴别器的损失函数采用二值交叉熵：**

$$
L^D = - \sum_{n=1}^N [\mathcal{y}_n \log \hat {\mathcal{y}}_n + (1 - \mathcal{y}_n) \log (1 - \hat {\mathcal{y}}_n)]
$$

其中N是测试样本总量，$\mathcal{y}_n$和$\hat {\mathcal{y}}_n$是事实和鉴别器的评估值。

网络的参数用Adam算法优化。

最后训练结果的参数包括**网络相关的**和**网络无关的**，在用到其他数据集时只需要重新调优网络相关的参数B，这样就达到了迁移的目的。

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

  - 25% 训练进程终止（training process termination）

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

## 我的感想

这篇论文断断续续地读了两周，虽然期间干了很多开发的杂活儿吧，但感觉效率还是太低了。

之前只断断续续地不成系统地看过一些AI的东西，不过大概能感觉到本文最大的创新就是参数分解$b \equiv B \kappa \mu$那步。

> We ground this work on a graph convolutional generative autoencoder that can generate the estimations for an entire transportation network in one go, and modify its internal computation graph to reduce the size of network topology-dependent model parameters. Subsequently, pre-trained models from road networks with massive historical data can be re-used in other networks with few data, which are only employed to adjust a small number of parameters.

摘要里说的*modify its internal computation graph to reduce the size of network topology-dependent model parameters* 和 *adjust a small number of parameters* 基本都是在说这个事。

模型的设计方面，我没仔细看另一篇论文，不过感觉没太大区别。

![](/img/online-traffic-speed-transfer-3.PNG)

![](/img/online-traffic-speed-transfer-4.PNG)

收获的话大概是：

- 了解了这类论文的格式，先定义一些符号描述问题，接着介绍算法模型，最后描述实验；

- 学到了一些英语的表达；

- 学到了Latex公式的写法，还挺简单的；

- 感觉在一些不太重要的地方读的太细了，下一篇要抓住重点。

然后还有个感觉就是，这么长篇大论下来基本就在说一个很简单的矩阵分解……虽然每个环节都是必要的吧，但效率是不是低了点。

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
- hypothesis n. 假设 is established on the hypothesis that
- with respect to 考虑到
- enclose v. 把…围住，封入，（随函）附入 parameter b encloses road network topological information
- amend v. 修正
- comprise v. 包括，由…组成
- agnostic adj. 不可知的

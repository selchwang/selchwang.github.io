---
title: "变分推断"
date: 2022-05-16T11:36:00
excerpt_separator: "<!--more-->"
share: false
mathjax: true
categories:
  - blog
tags:
  - 变分推断
related: false
gallery1:
  - url: /assets/images/Margaret1.png
    image_path: /assets/images/Margaret1.png
    alt: "Margaret1"
    title: "Margaret1"
gallery2:
  - url: /assets/images/Margaret2.png
    image_path: /assets/images/Margaret2.png
    alt: "Margaret2"
    title: "Margaret2"
---


假设模型是联合概率分布$p(x,z)$，其中$x$是观测变量，$z$是隐变量，包括参数。目标是学习后验概率分布$p(z \vert x)$，用模型进行概率推理。

但直接估计这个复杂分布是参数比较困难，故用一个概率分布$q(z)$来近似条件概率分布$p(z \vert x)$，用KL散度$D(q(z) \Vert p(z \vert x))$计算两者的相似度，$q(z)$称为变分分布（variational distribution）。如果能找到与$p(z \vert x)$在KL散度意义下最近的分布$q^*(z)$，则可以用其近似原分布。

KL散度可以写成：

$$
\begin{split}
D(q(z) \Vert p(z \vert x)) &= E_q[\log q(z)] - E_q[\log p(z  \vert  x)] \\
&= E_q[\log q(z)] - E_q[\log p(z,x)] + \log p(x) \\
&= \log p(x) - \{ E_q[\log p(z,x)] - E_q[\log q(z)] \}
\end{split}
$$

KL散度大于等于0，有：

$$
\log p(x) \ge \{ E_q[\log p(z,x)] - E_q[\log q(z)] \}
$$

左端称为证据（evidence），右端称为证据下界（evidence lower bound，ELBO），证据下界记作：

$$
L(q) = E_q[\log p(x,z)] - E_q[\log q(z)]
$$

在$\log p(x)$已知的情况下，求$q(z)$使KL散度最小化，可以通过最大化证据下界实现。

对$q(z)$的要求，自然是假设为对各个分量独立：

$$
q(z) = q(z_1)q(z_2) \dots q(z_n)
$$

此时的变分分布$q(z)$称为平均场（mean field）。KL散度的最小化是在平均场的集合中求解。

**变分法的优点在于，不需要设定$q(z)$的参数化形式，只要给出分解形式并求解即可。**

---
title: "Graph Diffusion"
date: 2022-05-07T15:57:00
excerpt_separator: "<!--more-->"
share: false
mathjax: true
categories:
  - sci
tags:
  - graph
  - graph diffusion
  - contrastive learning
  - augmentation
---

图扩散（diffusion）是图增广（augmentation）的常用形式，由此增广出的图更好地汇聚了全局特征，在图对比学习中常作为一种全局视图（global view）使用。

文献[1]指出：增广出多于2个的视图，并不能有效提升性能；最有效的是`first order neighbors` + `graph diffusion`。

## 公式

公式描述`graph diffusion`的过程：

$$
\mathbf{S} = \sum_{k=0}^{\infty} \theta_k {\mathbf{T}}^k
$$

其中，$\theta_k \in [0, 1]$是权重系数，用于控制结果中局部与全局结构信息的比例，$ {\textstyle \sum_{k=0}^{\infty}} = 1$；$\mathbf{T}\in \mathbb{R}^{N \times  N}$，是转换邻接矩阵的转移矩阵。

## 实例

文献[1]经验性地证明了在大多数任务中`Personalized PageRank (PPR)`、`heat kernel`这两种diffusion的实例可以达到最佳效果。文献[2]从图频谱角度分析对此进行了印证。

$$
\begin{array}{l}
\mathbf{S}^{\mathrm{heat}} &= &\mathrm{exp}(t\mathbf{AD}^{-1}-t) \\
\mathbf{S}^{\mathrm{PPR}} &= &\alpha (\mathbf{I}_n - (1-\alpha) \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2})^{-1}
\end{array}
$$

其中，$\alpha$为随机游走中的传送（teleport）概率；$t$为扩散时间（diffusion time）。

## 实验

```python
import torch

A = torch.as_tensor([
    [1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0]
])

def diffusion_heat(A, t=5):
    A = A.to(torch.float32)
    D = (torch.sum(A, 1) + 1e-5) ** -1
    D = torch.diag(D)
    return torch.exp(t * torch.mm(A, D) - t)

print(A)
print(diffusion_heat(A, t=1))
print(diffusion_heat(A, t=5))
```
输出的结果是：
```
tensor([[1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]])
tensor([[0.5134, 0.5134, 0.3679, 0.3679, 0.5134, 0.3679],
        [0.5134, 0.3679, 0.6065, 0.3679, 0.5134, 0.3679],
        [0.3679, 0.5134, 0.3679, 0.5134, 0.3679, 0.3679],
        [0.3679, 0.3679, 0.6065, 0.3679, 0.5134, 1.0000],
        [0.5134, 0.5134, 0.3679, 0.5134, 0.3679, 0.3679],
        [0.3679, 0.3679, 0.3679, 0.5134, 0.3679, 0.3679]])
tensor([[0.0357, 0.0357, 0.0067, 0.0067, 0.0357, 0.0067],
        [0.0357, 0.0067, 0.0821, 0.0067, 0.0357, 0.0067],
        [0.0067, 0.0357, 0.0067, 0.0357, 0.0067, 0.0067],
        [0.0067, 0.0067, 0.0821, 0.0067, 0.0357, 0.9999],
        [0.0357, 0.0357, 0.0067, 0.0357, 0.0067, 0.0067],
        [0.0067, 0.0067, 0.0067, 0.0357, 0.0067, 0.0067]])
```

本来想做个可视化但我懒得写了… 就待续吧。

## References

1. Hassani, K. & Khasahmadi, A. H. Contrastive Multi-View Representation Learning on Graphs. Preprint at http://arxiv.org/abs/2006.05582 (2020).

2. Liu, N., Wang, X., Bo, D., Shi, C. & Pei, J. Revisiting Graph Contrastive Learning from the Perspective of Graph Spectrum. Preprint at http://arxiv.org/abs/2210.02330 (2022).

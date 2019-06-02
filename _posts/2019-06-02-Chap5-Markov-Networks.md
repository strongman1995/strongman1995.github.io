---
layout: post
title: "Chap5 Markov Networks"
category: PGM
tags: []
date:   2019-06-02 13:25:35 +0200
---



Bayesian Network：$$P\Leftrightarrow \{P, G\}$$ G:有向图

Markov Network: $$P\stackrel{HC}{\Leftrightarrow}  \{P, H\}$$ H: 无向图

为什么需要无向图？当然就是因为有一些独立关系无法用有向图表达，所以才弄了个无向图。不然能表达为什么不统一用有向图。

> G is a perfect-map(P-Map) for P if I(P)=I(G)

不是所有的分布都能找到对应的 P-map 的，即找到一个有向图 G 所表达的独立关系和分布 P 相同。就比如分布 P 同时满足以下两个独立性 Ind(A;D\|B,C)，Ind(B;C\|A,D)，无法在 Bayesian Network 中表达，但是这两个独立关系，在 Markov Network A，B，C，D四个节点连成一个环就能表达出来。

Bayesian Network 是否能和 Markov Network 互相等价转换呢？

举两个反例：G —> H, G 中如果有 v structure 就无法转成 H。H—> G, 刚才独立性 Ind(A;D\|B,C)，Ind(B;C\|A,D)的例子就是一个反例。

## **Markov Network Factors**

Local factors are attached to sets of nodes. $$\pi_i[node_1, node_2, ...]$$ 

factor 是一个函数，随机变量（node）的集合的赋值映射到正实数集

在相同 factor 中的随机变量是互相依赖的，即两两都有条边连着。是一个完全子图（full-connected）。即是一个 clique(图论中的定义)

> $$C\subseteq V$$ is a clique, iff $$C \subseteq \{X, Neighbor(X)\} for \forall X \in C$$

在 MN 中基本结构就是 clique，在物理学中叫 energy potential

在 BN 中基本结构就是 CPD

其实CPD 就是 factor 的一种形式。比如A，B，C 节点都指向 D 节点，这是一个 v structure，给定 D，那么 A，B，C 三者两两都是依赖关系，所以 ABC 形成了一个 factor。

能否用以边来定义 的factor 来表达联合分布呢？不行！以边定义有 n 个节点全连接的图，有 4$$\tbinom{2}{n}$$个参数，而用节点定义的会有$$2^n-1$$个参数，以边定义会丢失一些高阶的独立信息。

## **Gibbs Distribution**

表示一个联合分布

- Un-normalized factor product: $$F(a, b, c, d) = \pi_1[a, b]\pi_2[a, c]\pi_3[b, d]\pi_4[c, d]$$
- Partition function(normalization function): $$Z=\sum_{a, b, c, d} F(a, b, c, d)$$
- Probability/Gibbs Distribution: $$P(a, b, c, d)=\frac{1}{Z} F(a, b, c, d)$$

这么一看Gibbs Distribution 就是把 H 中的 所有 factor 都连乘再进行归一化。

## Hammersley-Clifford Theorem——MN 的分解定理

根据Hammersley Clifford定理，一个无向图模型的概率可以表示为定义在图上所有最大团上的势函数的乘积

> I-map: I(H)$$\subseteq$$ I(P)
>
> $$D_i$$ encodes the cliques in the graph H
>
> **Factorization to I-map**: Given a undirected graph H, if P canbe factorized as $$P(X)=\frac{1}{Z} \prod \pi_i [D_i]$$, H is an I-Map of P
>
> **I-Map to Factorization**: Given a undirected graph H, if H is an I-map of P, P can be factorized as $$P(X)=\frac{1}{Z} \prod \pi_i[D_i]$$

所以分布 P 根据无向图 H 分解，分布 P 的分解式子和在 H 上的 Gibbs distribution 是一样的

Maximal Clique 就是合并一些小 clique，直到 clique 不能继续扩展。

####  log-transformation

$$P(X)=\frac{1}{Z} \prod \pi_i[D_i]$$ 做 log-transformation，把连乘转化为相加：$$P(X_1, …, X_n)=\frac{1}{Z} \exp[-\sum_{i=1}^m \epsilon_i[D_i]]$$

其中$$\pi_i=\exp(-\epsilon[D])$$, H中有 n 个随机变量 X, 有 m 个 clique

------

定义 Q 函数：$$Q(x\|NB(x))=log[\frac{P(x\|NB(x))}{P(x=0\|NB(x))}]$$, P(x=0\|NB(x)) 是 ground state，基态。

因为只需要求在总能量的比例，所以 Q 函数描述的是关于基态的相对概率。

Z 在通常情况下很难计算。

下面也是 HC theorem，1974 年由 besag 给出了构造性证明

> H is an I-map of positive distribution P $$\Leftrightarrow$$ This exists a unique expansion of Q: 
>
> $$Q(x)=\sum_i x_i \psi_i(x_i) + \sum_{i, j}x_ix_j\psi_{i, j}(x_i, x_j)+…+x_1x_2…x_n\psi_{1, 2, …, n}(x_1, x_2, …, x_n)$$ 
>
> 其中$$\psi_{S} > 0 , \forall v \in S $$ form a clique in H， $$\psi_S$$其实就是Clique S 的参数

所以根据HC 定理，一个 MN 的概率（Gibbs Distribution）可以表示为$$P(X)=\frac{1}{Z} \exp(-Q)$$

Potential function Q: $$Q(X=x)=-\sum_i \beta_i \prod_{x_j \in C_i}x_j$$

Partitiion Function Z: $$Z=\sum_X \exp(-Q(X))$$ 

现在这个 Q 和刚才的$$\epsilon_i[D_i]$$的区别，本质是一样的，只不过 Q 是枚举出来了每一种规格的 clique

### Ising Models

$$Q=-\sum_{i<j} w_{i, j}x_ix_j -\sum_i u_ix_i$$

$$P(Q)=\frac{1}{Z}\exp(-Q)$$

在图像分割中有应用

### Boltzmann Machines

$$Q=-\sum_{i, j}w_{i, j}b_ic_j-\sum_iv_ib_i-\sum_ih_ic_i$$

## BN 和 MN 之间的转化

### BN—>MN

moralize:

1. 在 G 中 X 和 Y 直接相连，在 H 中X 和 Y 也有一条直接相连的边
2. 在 G 中 X 和 Y 有共同孩子，在 H 中 X 和 Y 加一条相连的边

第二条规则会产生一个 full clique。

moralize 的转化操作会损失独立性假设，因为在 BN 中v-structure 的独立性Ind（A;B\|C）和 MN 中转化的 v-structure 独立性(没有刚才的独立性)

### MN—>BN

这个转化更难，而且会生成比 MN 大得多的 BN

Triangulate:

1. 将 MN 作为模板
2. 固定节点的顺序
3. 按照 2的顺序根据分布的独立性假设与最小的父亲节点集添加边

> Chordal Graph 弦图就是只要节点数超过 3 的loop 都有一根弦

如果有BN 是弦图，那对应的 MN 也是弦图
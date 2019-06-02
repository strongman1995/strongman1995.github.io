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

不是所有的分布都能找到对应的 P-map 的，即找到一个有向图 G 所表达的独立关系和分布 P 相同。就比如分布 P 同时满足以下两个独立性 Ind(A;D|B,C)，Ind(B;C|A,D)，无法在 Bayesian Network 中表达，但是这两个独立关系，在 Markov Network A，B，C，D四个节点连成一个环就能表达出来。

Bayesian Network 是否能和 Markov Network 互相等价转换呢？

举两个反例：G —> H, G 中如果有 v structure 就无法转成 H。H—> G, 刚才独立性 Ind(A;D|B,C)，Ind(B;C|A,D)的例子就是一个反例。

## **Markov Network Factors**

Local factors are attached to sets of nodes. $$\pi_i[node_1, node_2, ...]$$ 

factor 是一个函数，随机变量（node）的集合的赋值映射到正实数集

在相同 factor 中的随机变量是互相依赖的，即两两都有条边连着。是一个完全子图（full-connected）。即是一个 clique(图论中的定义)

> $$C\subseteq V$$ is a clique, iff $$C \subseteq \{X, Neighbor(X)\} for \forall X \in C$$

在 MN 中基本结构就是 clique，有时也叫 potential

其实CPD 就是 factor 的一种形式。比如A，B，C 节点都指向 D 节点，这是一个 v structure，给定 D，那么 A，B，C 三者两两都是依赖关系，所以 ABC 形成了一个 factor。

能否用以边来定义 的factor 来表达联合分布呢？不行！以边定义有 n 个节点全连接的图，有 4$$\tbinom{2}{n}$$个参数，而用节点定义的会有$$2^n-1$$个参数，以边定义会丢失一些高阶的独立信息。

## **Gibbs Distribution**

表示一个联合分布

![](/Users/chenlu/strongman1995.github.io/assets/images/2019-06-02-chap5/1.png)

这么一看Gibbs Distribution 就是把factor 都连乘再进行归一化。Z 也叫 partition function

## Hammersley-Clifford Theorem——MN 的分解算法

I-map: I(H)$$\subseteq$$ I(P)

> $$D_i$$ encodes the cliques in the graph H
>
> Factorization to I-map: Given a undirected graph H, if P canbe factorized as $$P(X)=\frac{1}{Z} \prod \pi_i [D_i]$$, H is an I-Map of P
>
> I-Map to Factorization: Given a undirected graph H, if H is an I-map of P, P can be factorized as $$P(X)=\frac{1}{Z} \prod \pi_i[D_i]$$

所以分布 P 根据无向图 H 分解，分布 P 的分解式子和在 H 上的 Gibbs distribution 是一样的

Maximal Clique 就是合并一些小 clique，直到 clique 不能继续扩展。


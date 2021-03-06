---
layout: post
title: "概率图学习——Inference as Optimization: Cluster Graph & Belief Propagation 聚类图，置信传播"
category: pgm
tags: [推断, 聚类图, 置信传播]
date:   2018-10-26 13:25:35 +0200
---

## Variable Elimination(VE) 消元法

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/1.png)

线性链上的消元：

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/2.png)

按照 A、B、C、D 的顺序依次消去

### VE in complex graphs

induced graph in VE

step 1: Moralizing for BN （即在 v-structure 的两个父亲节点连边）

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/3.png)![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/4.png)

step 2: Triangulation （即在消元过程中做三角化操作）

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/5.png)![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/6.png)

最后出来会是 chordal，即是一个弦图，图中只有三角，没有四边形

如果每次 inference 的时候都要遍历整个图，那就太蠢了，这里可以采用动态规划的算法，消元时候的中间结果是可以拿来重用的。

提前计算好定义在每个 clique 上的 marginal distribution，在做 inference 时候也能快很多。

## Exact Inference: Clique Tree 

### Cluster graph and clique tree

clique tree 是定义在弦图上的，根据 clique 生成的树状结构。

clique tree 有两个非常重要的性质：

1. tree and family preserving: 原来的 induced graph 转化为 clique tree 后具有树状结构，而且和原来的结构是可以相互转换的。clique tree 的每一个节点是代表一个 clique，边上是两个 clique 间重叠的部分。

2. running intersection property：是指变量 X 存在一条连续的树的子路径上。如 G 出现在了 Clique2和 clique4中，那么中间的 clique3和 clique5

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/7.png)![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/8.png)

每个 clique 都是有他们对应的 local CPD

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/9.png)

### Message passing: Sum Product

顺序1：

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/10.png)

顺序2：

![在这里插入图片描述](http://strongman1995.github.io/assets/images/2018-10-26-inference-as-optimization/11.png)

### Clique Tree Calibration

calibration（校准）：使得两个相邻 clique 之间传送的消息相等。

### Message passing: Belief Update
### Constructing clique tree3
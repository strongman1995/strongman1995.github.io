---
layout: post
title: "Chap9 Structured Variational Inference"
category: PGM
tags: []
date:   2019-06-05 13:25:35 +0200
---

Structure Variational Inference 是在 PGM 经常使用的推断方法！

Variational Inference的基本思想和 Importance Sampling 有点像(从分布 Q 中采样来模拟从 P 中采样的样本)：为了推断目标 P(X), 找到另一个分布 Q(X)来近似，这样就可以在 Q(X) 上更容易做 inference。因为有的时候 P(X)很复杂无法exactly 求出来，可以用分布 Q(X) 近似 P(X)，可能不是完全细节一致，但是大体是一致(这也是为什么后面用 Variational Inference 来做 autoencoder，可以去掉一些噪声)，分布 Q(X)比 P(X)简单。

本章只讨论有向图 G 上的变分推断

> Variational inference is an **optimization** process to find a variational distribution **Q(X)** from a distribution family by **maximizing an energy functional**

P(X)：target distribution

Q(X): variational/proposal distribution，Q 是由比 P更简单形式构成的分布族，如高斯分布族，所以可以用来 de-noising 或者 dimension reduction

------

KL divergence：是衡量两个分布距离的测量标准

$$D_{KL}(Q\|P)$$ Or $$D_{KL}(P\|Q)$$

$$D_{KL}(Q\|P)=-E_{X\sim Q}\ln\frac{P}{Q}=E_{X\sim Q}\ln \frac{Q}{P}=\sum_x Q(x) \ln\frac{Q}{P}$$

Q 在前面，对 Q 做期望，有负号是为了凑熵的形式

------

**目标：$$\mathop{\min}_Q D_{KL}(Q\|P)$$**

找到和分布P KL 距离最小的分布 Q，其实只要是测量分布距离的标准都可以，目标就是尽量使 Q 在这个标准下靠近 P

Structured variational inference

Q(X)：定义在简单结构G上的分布，比如没有边的图(mean field algorithm), G 上都是相互独立的节点

假设 Q(X)是高斯分布族：

- Forward KL(M-projection):$$D_{KL}(P\|Q)$$， Q 需要 cover P 的全部范围
- **Backward KL(I-projection): $$D_{KL}(Q\|P)$$** 一般都使用 backward KL，Q 只需要 model P 的一个 peak 就行

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/1.png)

### Energy Functional

令 target distribution P(X\|Z=z),那么 KL 距离：

$$\begin{align}D_{KL}(Q\|P)&=E_{X\sim Q}(\ln \frac{Q(X)}{P(X\|z)})\\ &= E_{X\sim Q}(\ln \frac{Q(X)}{P(X,z)})  + \ln P(z) \\ &= E_{X\sim Q}(\ln Q(X)) - E_{X \sim Q}(\ln P(X, z)) + \ln P(z) \\&= H(Q(X)) - E_{X \sim Q}(\ln P(X, z)) + \ln P(z)\end{align}$$

$$\begin{align} \ln P(z) - D_{KL} (Q\|P) &= E_{X \sim Q}(\ln P(X, z)) -  H(Q(X)) \end{align}$$ <— **Energy functional**

Q1: 如何选择Q(X)分布的分布族？

Q2:如何最大化 energy functional？

原始目标 $$\min D_{KL}(Q\|P) $$

$$\rightarrow \max \ln P(z) - D_{KL} (Q\|P) $$

$$\rightarrow \max  E_{X \sim Q}(\ln P(X, z)) +  H(Q(X))$$ 

energy functional 第一项 $$E_{X \sim Q}(\ln P(X, z)) $$，是用来近似 P 的，P 和 Q 越像，第一项的值就越大

energy functional 第二项 $$H(Q(X))$$，是用来防止过拟合的，让 Q(X)四散，尽量 cover 更多范围，防止拟合到

### Evidence Lower Bound(ELBO)

$$L(Q)= E_{X \sim Q}(\ln P(X, z)) +  H(Q(X))=\ln P(z) - D_{KL} (Q\|P) \leq \ln P(z) \quad D_{KL} \geq 0$$ 

因为 KL 距离是非负的，所以 energy functional 的 bound 是 ln P(z)

L(Q) 也叫 ELBO

## Mean Field Variational Inference

目标：推断(计算)p(x\|z)——很难直接计算

使用简单的分布 q 来模拟分布 p，这个简单的分布 q 就是完全分解的形式$$q(x;\theta)=\prod_i q_i (x_i)$$

$$E_{x \sim q}(\ln p(x, z)) = \sum_x [\prod_i q_i (x_i)] \ln p(x, z)$$

$$H(q) = \sum_x [\prod_i q_i(x_i)][-\sum_i \ln q_i (x_i)]$$

$$L(q) = \sum_x[\prod_i q_i (x_i)][\ln p(x, z)-\sum_k \ln q_k (x_k)]$$

打公式好累。。。直接上图吧。。。

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/2.png)

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/3.png)

L(q),ELBO 的形式和 KL 距离的形式一致，由 KL 距离得知，当 q 和 p 的分布越接近，kl 距离越小。

同理，$$q_j(x_j)$$和$$\exp \{E_{-q_j [\ln \hat{p} (x)]}\}$$ 成正比能够求得L(q_j)的最大值。

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/4.png)

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/5.png)

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/6.png)

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/7.png)

总结一下structure variational inference：

1. KL 距离
2. 由 KL 距离，推出 energy functional L(q)。且最小化 KL 距离等于最大化 energy functional
3. 由$$L(q_i)$$和 KL 距离公式形式一样，推出最优解使得$$\ln q_j(x_j)$$最优值正比于$$E_{-q_j }[\ln \hat{p} (x)]$$ 
4. 所以如果要求出分布$$\ln q^*_j(x_j)$$，求$$E_{-q_j} [\ln \hat{p} (x)]$$就行
   1. 由图求出$$\ln \hat{p}(x)$$
   2. 设 $$q_i(x_i; \theta_i)$$的 分布族(the family of the variational distribution)
   3. 代入 1，2 得到$$E_{-q_j} [\ln \hat{p} (x)]$$,即得到$$\ln q^*(x_j;\theta_j)$$ 
   4. 得到最优参数$$\theta_j$$
   5. 迭代求得每个参数$$\theta_j$$
5. 得到 $$q^*(x;\theta)=\prod_i q_i (x_i)$$ 后就从 q*做 inference ，而不是 p 了，因为此时 q *已经近似了 p

LDA 中的变分推断，需要特别开一章进行学习。

注意，变分推断中，近似的 q ，变量 variables 需要和 p 相同，参数可不同

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/8.png)

变量都是$$\theta 和 z$$，但是参数是$$\gamma$$和$$\phi_n$$ 

在很多情况下，无法用平均场的变分推断求出近似解，此时随机梯度下降的方法是经常使用到的。

![](https://strongman1995.github.io/assets/images/2019-06-05-chap9/9.png)

> Variational inference can be regarded as a kind of learning: find the optimal parameters by maximizing an energy functional
>
> Learning as MLE: find the optimal parameters by maximizing likelihood function given data.


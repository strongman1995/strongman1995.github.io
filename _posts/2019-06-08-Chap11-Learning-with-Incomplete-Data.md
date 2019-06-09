---
layout: post
title: "Chap11 Learning with Incomplete Data"
category: PGM
tags: []
date:   2019-06-08 13:25:35 +0200
---

### Reason of Incomplete Data

变量无法被监测：

1. 变量无法被观测到：
2. 变量只存在于概念中：HMM

### HMM

HMM 的学习已经在 HMM 那一章节提到，是先使用 Gibbs Sampling 完成 inference 过程，然后再重新估计参数，直到迭代收敛的方法完成学习

## Gaussian Mixture Model

GMM 是通过 MLE，最大似然估计的方法学习的：$$\arg \max_\theta p(D\|\theta)$$其中 D={y[1], …, y[M]}

#### Complete data case

在数据完整的情况下，MLE 的学习是比较简单的。

- complete data：$$D_c=\{(x[i], y[i])\}_{i=1…M}$$ 
- k的估计：$$\pi_k^* = \frac{M[x=k]}{M}$$
- 第 k 个 mixture Gaussian component:
  - $$\mu_k^* = \frac{1}{M[x=k]} \sum_m y[m]\|_{x[m]=k}$$
  - $$\sum_k^*=\frac{1}{M[x=k]} \sum_m (y[m]-\mu_k^*) (y[m]-\mu_k^*)^T\|_{x[m]=k}$$

### Parially observed data

我们无法观测到 component label，即 x=k 不知道。(这个不就是 EM算法吗？┑(￣Д ￣)┍)

在数据部分观测的情况下，给参数的情况下，我们能计算label 的后验(inference) $$P(x=k\|y, \theta)$$

$$Q(x=k)=P(x=k\|y, \theta)=\frac{p(y\|x=k, \theta)P(x=k\|\theta)}{\sum_{k=1}^K p(y\|x=k, \theta)P(x=k\|\theta)}$$

$$Q^{(t)}(x[m]=k)=\frac{\pi_k^{(t)}N_k^{(t)}(y[m])}{\sum_{k=1}^K \pi_k^{(t)}N_k^{(t)}(y[m])}$$ 

其中

$$P(x=k\|\theta)=\pi_k$$

$$P(y\|x=k, \theta)=N_k(y[m])$$ 

$$N_k(y)=\frac{1}{\sqrt{\|2\pi \sum\|}} \exp(-\frac{1}{2}(y-\mu_k)^T\sum^{-1}(y-\mu_k))$$

------

所以参数$$\pi$$可以用 MLE 学习方法更新：$$\pi_k^{(t+1)}=\frac{1}{M}\sum_{m=1}^M Q^{(t)}(x[m]=k)$$ 原来是一个个样本，现在是概率个样本

而 P(y\|x=k)的参数也可以用 $$Q^{(t)}(x[m]=k)$$来更新：

$$\mu_k^{(t+1)}=\frac{\sum_{m+1}^M Q^{(t)}(x[m]=k) y[m]}{\sum_{m+1}^M Q^{(t)}(x[m]=k) }$$

$$\Sigma_k^{(t+1)}=\frac{\sum_{m+1}^M Q^{(t)}(x[m]=k) (y[m]-\mu_k^{(t+1)})^T(y[m]-\mu_k^{(t+1)})}{\sum_{m+1}^M Q^{(t)}(x[m]=k)}$$ 

根据上面更新过的参数，可以很容易地计算新一步的$$Q^{(t+1)}(x[m]=k)$$ 

------

总结：

- GMM 的学习是 component label x 和模型参数$$\pi, \mu, \Sigma$$都不知道
- 通过 label 的后验(inference)迭代地更新 component labels ，再根据 MLE(learning)更新模型参数

GMM inference：Marginal， MAP， Sampling(MCMC)

GMM learning：Hierachical Bayesian Model，MLE

### Expectation Maximization(EM)

EM 算法：

1. INFERENCE: Given parameters do INFERENCE for unobserved data
2. LEARNING: Maximize the marginal likelihood according to inference results
3. Iteratively do INFERENCE and LEARNING

MAP 和 MLE 的区别是在 P(x[m]=k)那一步，MLE 是概率，MAP 是 x[m]=arg max_k Q(x[m]=k) 

## 总结：

- 初始化参数
- 根据observed variables推断 hidden variables
  - 用 probability(期望) 填充
  - 用值填充(MCMC 或者 MAP)
- 学习(更新)参数
  - MLE 或者 MAP 求点估计
    - Expectation maximization(hard EM)
    - Stochastic gradient ascent
  - Bayesian Learning, 求分布
    - MCMC sampling
    - Variational bayesian learning
- 重复上述过程直到收敛
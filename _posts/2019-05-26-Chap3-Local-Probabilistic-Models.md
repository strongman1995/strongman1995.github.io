---
layout: post
title: "Chap3 Local Probabilistic Models"
category: PGM
tags: []
date:   2019-05-26 13:25:35 +0200
---

Local Probabilistic Model

Conditional Probability Distributions (CPDs)

**The Basic Structures in BNs**

回顾基于 G 的概率分解$$P(X_1, …, X_n)=\prod_{i=1}^n P(X_i\|Pa(X_i))$$ 

所需要的表达项是 local conditional probability distributions(CPDs) encoded by the target variable $$X_i$$given its parents $$Pa(X_i)$$ 

## Discrete CPDs

### Tabular CPDs

对于(Y\|X1, X2, …, Xk) 有$$2^k$$ 个独立参数。参数是关于父亲个数呈指数级增长的

### Rule CPDs

即把 Tabular CPDs 相同项合并起来。

Tree CPDs 可以直接转化为 rule CPD

### Independence of causal influence

"Causal"\|"Associative"

## Continuous CPDs

### Logistic CPDs

$$P(Y=y^1\|X_1, …, X_k)=sigmoid(w_0+\sum_{i=1}^k w_i X_i)$$  这个其实就是 NN 了，X1...Xk 是输入节点，Y 是 hidden layer 中的一个点，sigmoid是 activation function

Log-odds: $$O=\frac{P(Y=y^1\|X_1, …, X_k)}{P(Y=y^0\|X_1, …, X_k)}=e^Z$$ 

更大的 odds 表示更有可能导致 positive result

如果 log-odds 中的二元变量 Xj 改变$$\Delta O=\frac{O(X_{-j}, X_j=x_j^1)}{O(X_{-j}, X_j=x_j^0)}=e^{w_j}$$ 

$$w_j>0$$ 代表有 positive contribution

### Generalized Linear Models

- Linear Gaussian
- Binomial distribution (Logistic CPD)
- Poisson distribution

## A few CPDs in neural networks

ReLu: Rectified Linear Units

Rectifier: $$y=f(s)=max(0, s)$$

Softplus: $$y=f(s)=ln(1+exp(s))$$ 

## Bayesian networks representation examples

![](https://strongman1995.github.io/assets/images/2019-05-26-chap3/1.png)

![](https://strongman1995.github.io/assets/images/2019-05-26-chap3/2.png)

![](https://strongman1995.github.io/assets/images/2019-05-26-chap3/3.png)


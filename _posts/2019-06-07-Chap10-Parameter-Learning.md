---
layout: post
title: "Chap10 Parameter Learning"
category: PGM
tags: []
date:   2019-06-07 13:25:35 +0200
---

## Learning Basics

optimal model有很多目标函数：

- minimum error
- probability
  - maximum Likelihood
  - maximum a posterior
- maximum margin
- compressive sensing



在概率框架下：

1. Parameter Learning：$$\{x[m]\}_{m=1\sim M}\|G \rightarrow P(\theta\|D)$$
2. Structure Learning: $$\{x[m]\}_{1\sim M} \rightarrow P(G, \theta \| D)$$



- Generative Models: Learn joint probability P(Y, X=x)
- Discriminative Models: Learn conditional probability P(Y\|X=x)



### IID Samples

从独立同分布的变量中采样的数据。iid samples 是互相无关的 unrelated 的数据

### Avoid Overfitting

> Model complexity should fit the complexity of data

#### 什么是 overfitting？

model complexity 比 training data 大很多，在训练的时候得到 zero expirical risk,但是在 test 时预测结果很差

#### 如何判断 overfitting？

判断是否在 testing 的时候 performance 远远低于 training

#### Generalization

为了避免 overfitting，增加学习模型的泛化能力

- penalize the model complexity
  - the regularization term in loss function
    - L1 norm：LASSO
    - L2 norm: Ridge
    - L1+L2: Elastic Net
  - simplify model structure
    - dropout
    - Batch Normalization
- separate training/testing
  - Cross validation
  - 0.632 bootstraping

## Maximum Likelihood Parameter Estimation

> Likelihood: the probability or confidence for parameter assignment, given a number of data
>
> Log likelihood is commonly used for better calculation
>
> MLE: the parameter estimation is to find the optimal parameter assignment $$\theta^*$$ which can maximize the likelihood(optimization)

$$\max L(\theta;D) \propto \log P(D\|\theta)=\sum_i \log \phi_i(x[i];\theta)$$ 

如果是伯努利分布，对似然对参数求导就能求出最优参数 theta

如果是复杂函数，可以用gradient-based methods, 如果 Likelihood function 是凸的，可以收敛到最优。

为了减少每轮迭代的计算量，使用随机梯度下降的方法，每轮迭代只计算一个样本的梯度。为了避免局部最优，可以使用启发式算法，比如模拟退火和遗传算法

### MLE in BNs

把每个 local Likelihood function分别独立求最大化的值，求出 local probability 的参数

$$L_i(\theta, X)=\prod_m P(x_i[m]\|Pa_{x_i[m]};\theta_{x_i\|Pa_{x_i}})$$ 

MLE 是点估计，point estimation 不是 prediction 或者 distribution estimation

MLE 通过最大化观测数据的联合概率找到最优的参数，但是 MLE 不能使用先验的知识或者参数的限制条件

## Bayesian Parameter Estimation

可以使用贝叶斯公式来做估计$$P(\theta\|X)=\frac{P(X\|\theta)P(\theta)}{\int P(X\|\theta)P(\theta)}$$

所以得到参数的后验分布$$P(\theta\|X)$$，而不是点估计。

但是一般情况下，bayesian model 都是用来在新数据上做 prediction 的

$$P(X^{n+1}\|X^1, …, X^n) = \frac{1}{P(X^1, …, X^n)}\int P(X^{n+1}\|\theta) P(X^1, …, X^n\|\theta)P(\theta) \, d\theta$$

右侧的 $$P(\theta)$$是$$\theta$$的先验分布，如果用 uniform distribution 代表没有先验。

当训练数据不够时，Bayesian estimation可以控制泛化能力。用Bayesian estimation做预测可以看做是 model averaging over all possible parameter settings.

### Beta Distribution——Binomial

通常使用 beta distribution 来表示先验，只适用于二项分布，$$\theta_1 + \theta_2=1$$。$$\theta \sim Beta (\alpha_1, \alpha_0)$$ if $$p(\theta)=\gamma \theta^{\alpha_1-1}(1-\theta)^{\alpha_0-1}$$ 

- $$\gamma=\frac{\Gamma(\alpha_1+\alpha_0)}{\Gamma(\alpha_1)\Gamma(\alpha_0)}$$ Is a normalizing factor
- $$\Gamma(x)=\int_0^\infty t^{x-1}e^{-t}\, dt$$ 
- $$\Gamma(n)=(n-1)!$$ for integer

$$P(\theta\|X)=\frac{P(X\|\theta)P(\theta)}{\int P(X\|\theta)P(\theta)}=Beta(M[1]+\alpha_1, M[0]+\alpha_0)$$ 

二项分布：$$P(X\|\theta)=\theta^{M[1]}(1-\theta)^{M[0]}$$

beta 分布：$$P(\theta) = Beta(\alpha_1, \alpha_0)=\gamma \theta^{\alpha_1-1} (1-\theta)^{\alpha_0-1}$$

Beta distribution 是 conjugate 共轭的。只用于 binomial distribution

预测的时候：

$$P(x[m+1]\|X)=\int P(x[m+1], \theta \| X)\, d\theta =\int P(x[m+1]\|\theta, X)P(\theta\|X)\, d\theta=\int P(x[m+1]\|\theta)P(\theta\|X)\,d\theta$$

其中$$P(x[m+1]=1\|\theta)=\theta$$ 

$$P(\theta\|X) \propto P(X\|\theta)P(\theta)\propto \theta^{M[1]}(1-\theta)^{M[0]} \theta^{\alpha_1-1} (1-\theta)^{\alpha_0-1}$$ 

所以$$P(x[m+1]=1\|X)=\int \theta P(\theta\|X)\, d\theta =\frac{M[1]+\alpha_1}{M+\alpha}$$

### Dirichlet Distribution——Multinomial

Dirichlet distribution 是多项分布 $$\sum_k \theta_k = 1$$

The likelihood function for multinomial $$L(\theta;D)=\prod_k \theta_k ^{M[k]}$$

$$\theta \sim Dirichlet(\alpha_k)$$ if $$P(\theta) \propto \prod_k \theta_k^{\alpha_k-1}$$

Dirichlet distribution 的优良性质(和 Beta distribution)一样：

1. conjugate
2. 后验：$$\alpha_k^*=M[k]+\alpha_k$$

所以$$P(x[M+1]=k)=\frac{M[k]+\alpha_k}{M+\alpha}$$

## MAP Parameter Estimation

MAP estimation is defined as $$\hat{\theta} = \arg \max_{\theta} \log P(\theta\|D)=\arg \max_{\theta} (\log P(D\|\theta) + \log P(\theta))$$

如果二元分布 beta prior $$\alpha_1=\alpha_0=1: P(\theta\|D)\propto \theta^{M[1]}(1-\theta)^{M[0]}$$

此时 MAP 和 MLE 是一样的

如果有大量的训练数据，那么 MAP estimation 主要被训练数据影响，似然函数$$\log P(D\|\theta)$$ 会占主要地位，$$\log P(\theta)$$可以看成是 regularization。

预测时的后验，$$P(X\|D)=\int P(X\|\theta)P(\theta\|D)\, d\theta \approx P(X\|\theta)$$ 



## 总结：

- MLE 和 MAP 都是点估计，可以用梯度下降求解，
- Bayesian estimation 是概率推断，需要共轭的先验，用 MCMC 做 inference
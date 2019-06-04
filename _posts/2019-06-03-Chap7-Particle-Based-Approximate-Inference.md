---
layout: post
title: "Chap7 Particle-Based Approximate Inference"
category: PGM
tags: []
date:   2019-06-03 13:25:35 +0200
---

终于到了非常重要的 Inference！

Inference 有两种类型：一个是这章说的 Particle based approximate inference，是用采样的方法来inference （近似估计）概率，是随机化算法。另一个是optimization based inference(精确算出)概率，是确定性算法。

Inference 的共同目标就是给了部分的观测值 E=e，推断目标变量 Y：

- P(Y\|E=e)
- $$\mathop{\arg \max}_{y} P(Y=y\|E=e)$$ (MAP inference)

Inference 的应用场景：

- 疾病诊断，已知病人的一些生活习惯，做的病理检查，和症状（observation E），推测疾病(target Y)
- NER：实体命名识别，已知一段文本(observation E), 推测一个 phrase 是否是一个实体(target Y)
- 图像/语音合成：风格迁移，已知原始图片(observation E)，推测出对应每个像素的后验颜色(target Y)

如果直接从联合分布 P 中做推断，用variable elimination 的方法复杂度是 NP-hard。但是我们知道很多 network structure，从而降低了复杂度，有很多高效的近似算法达到很高的推断准确度

------

Particle-based Approximate Inferences 的基本思想就是从目标分布（联合分布）中采样然后用样本来近似估计目标概率(边缘分布)，基本思想非常简单：$$E_P(f)\approx \frac{n_x}{n_t}$$ 

所以要如何得到 samples 来估计后验概率 P(x\|E=e)?

- Forward Sampling(简单，但费时)
- Likelihood Weighting and Importance Sampling(他俩有异曲同工之妙)
- Markov Chain Monte Carlo Methods(基本是万能的方法，效果也好)

## Forward Sampling

就是通过 BN 来生成samples，然后估计概率用$$P(X=e)\approx \frac{1}{M} \sum_{m=1}^M I(x[m]=e)$$

采一个样本 x[m]的过程：

1. 定义变量的拓扑排序X1, …, Xn
2. 通过 BN 已知的概率P(Xi\|pai) 采样每个 Xi ，pai 是前 i-1 个变量的子集， 已经采过值了
3. 得到样本 x[m]=(X1, …, Xn)

估计概率：$$P(X=e)\approx \frac{1}{M} \sum_{m=1}^M I(x[m]=e)$$

其他函数的期望：$$E_P(f)\approx  \frac{1}{M} \sum_{m=1}^M f(x[m])$$ 

求的目标分布只是一部分变量 Y：$$P(y)\approx \frac{1}{M} \sum_{m=1}^M I(x[m](Y=y))$$ 

这个方法复杂度高，因为采每个变量的值复杂度是 O(log(Val\|Xi))，每个n个变量样本就是 O(nlog(d)) d=max Val\|Xi，总共 m 个样本：O(Mnlog(d))

### Foward Sampling with Rejection

P(Y\|E=e), 在采样过程中把 E≠e 的样本扔掉。

这会产生一个问题，如果 E ≠e 的概率很小，那就会扔掉大量的样本，造成浪费，为了解决这个问题就产生了 Likelihood weighting 的方法，来减少浪费。

## Likelihood Weighting

虽然我们的目标是 P(Y\|E=e)，但是我们可以从先验 P(X)中得到 samples，然后用 Likelihood 加权，得到 weighting samples。

首先明确 X 是所有变量，Y 是目标变量，E 是观测变量

像 FS 一样也从 P(X)分布中采样，但是设 E=e，我们目标是 P(Y\|e)，这是 P(Y，e)的一小部分

采样过程中因为 E=e，所以e 的 CPD：$$P(E_j=e_j\|Pa(E_j))$$

所以样本要用$$\prod_j P(E_j=e_j\|Pa(E_j))$$ 加权

这样来看，其实 LW 和 FSwith rejection 是一样的本质，只不过是用加权来减去 rejection带来的计算损失

采一个样本 x[m]的过程：

1. 定义变量的拓扑排序X1, …, Xn
2. 如果变量 Xi 属于 E，直接设Xi 等于观测值 E[xi] 并且更新权值 wi：$$X_i=E[x_i], w[m]=w[m] * P(E[x_i]\|Pa_i)$$
3. 如果变量 Xi 不属于 E，从概率 P(Xi\|Pai)中采 Xi的值
4. 得到样本 x[m]=(X1, …, Xn)

估计概率：$$P(Y=y\|e)\approx \frac{\sum_{m=1}^M w[m]I\{x[m](Y=y)\}}{\sum_{m=1}^M w[m]}$$

## Importance Sampling

IS 的基本思想是不从分布 P 中采样，而是从另一个分布 Q 中采样。这么做的目的是因为分布 P 复杂，从更简单的分布 Q 来拟合 P 更容易

分布 P 被叫做 target distribution

分布 Q 叫做 proposal/sampling distribution

### Unnormalized Importance Sampling

![](https://strongman1995.github.io/assets/images/2019-06-03-chap7/1.png)

第一个公式左边是从 Q 分布中函数 $$f(X)\frac{P(X)}{Q(X)}$$采样的期望，右边是从分布 P 中函数 $$f(X)$$ 采样的期望

所以第二个公式，用 Q 分布中采样样本 x[m]来估计 P 分布。

### Normalized Importance Sampling

![](https://strongman1995.github.io/assets/images/2019-06-03-chap7/2.png)

框起来的normalized 的式子，相对于unnormalized 差距在于：在分布 Q 的期望上也做了normalized

如果用 multilated network $$G_{E=e}$$做 Q 分布的图，最后计算出来的估计公式其实和LW 是一样一样的

![](https://strongman1995.github.io/assets/images/2019-06-03-chap7/3.png)

LW 不太适用于 MN，因为需要把 MN 转成 BN

IS 的缺点是对于复杂结构 P很难选好 Q，如果 P 和 Q 太不像，收敛速度会很慢

## Markov Chain Monte Carlo

MCMC 方法基本思想就是去找到一个Markov Chain，使这个 Markov Chain 的稳态分布等于目标分布 P(Y\|E=e)。这是基于一个很重要的定理：

> Theorem: A finite state Markov chain T has a **unique stationary distribution** if and only if **it is regular**

regular：简单一点说就是任何一个状态到另一个状态都是 k 步可达

> A Markov chain is **regular** if there is *k* such that for every *x*,*x*’ $$\in$$ Val(*X*), the probability of getting from *x* to *x*’ in exactly *k* steps is greater than zero

对于 BN 的 regular：所有 CPD 严格大于0

对于 MN的 regular：所有 clique potential 严格大于 0

### Gibbs Sampling

对 Markov Chain 的转移概率$$T(x'\rightarrow x): P(X=x\|X=x')$$

全概率公式：$$P(x) = \sum_{x'} P(x')P(x\|x')$$

$$\pi(X=x)=P(X=x)$$是$$\pi(X=x)=\sum_{x'} \pi(X=x')T(x'\rightarrow x)$$的一个解

其中$$\pi(X=x)$$是 Markov Chain 的稳态概率分布，P(X=x)是目标概率分布

是唯一解吗？是的！因为刚才关于 MC regular 的定理。

采一个样本 x[m]的过程：

1. 设 x[m]=x[m-1]，并且确定一个变量更新顺序(刚才前面的直接是拓扑排序)
2. 对于每个除了观测变量之外的变量 Xi：
   1. 设 $$u_i=x[m](X-X_i)+e$$, 可以简化到 Xi 的 Markov blanket
   2. 从$$P(X_i\|u_i)$$采样得到采样值x\[m](X_i) 
3. 得到样本 x[m]=(X1, …, Xn)

之前HMM 中学习中的 inference 是用的 Gibbs sampling 方法

### Metropolis-Hastings Algorithm

这个可以看做是 Gibbs Sampling 的一个泛化方法，不是从 Markov Chain 的稳态分布来生成 samples，而是可以从任意的转移分布生成 samples。

定义了新的因子：acceptance probability 来决定是否接受一个转移 A(x—>x')

$$T(x—>x')=T^Q(x—>x') A(x—>x')$$

$$T(x—>x)=T^Q(x—>x') +\sum_{x'≠x}T^Q(x—>x') (1-A(x—>x'))$$

当MC 处于稳态，互相转移概率应该是相等的：

$$\pi(x)T^Q(x—>x')A(x—>x')=\pi(x')T^Q(x'—>x)A(x'—>x)$$

得到 A(x—>x')=min[1, $$\frac{\pi(x')T^Q(x'—>x)}{\pi(x)T^Q(x—>x')}$$]

#### MH Algorithm for Continuous Probability

转移概率一般设为多变量的高斯分布

高斯转移概率可以看做是当前状态为中心的 random walk，此时转移概率是对称的，那么acceptance rate $$A(x—>x')=min[1, \frac{\pi(x')}{\pi(x)}]=min[1, \frac{p(x')}{p(x)}]$$

### (Hybrid) Hamitonian Monte Carlo

HMC 算法是利用物理系统中动力学的概念来计算Markov链中的未来状态，而不是概率分布。相比于随机游走，通过这种方式，能够更加高效的分析状态空间，从而达到更快的收敛。HMC 也算是 MH的一种算法。[TODO]

### MCMC in practice

1. 需要等burn in time 结束Markov Chain 才能进入稳态，之后的所有样本都来自稳态
2. burn in time很难估计
3. 没有理论保证稳态的存在
4. 从相同链采样的 samples 是correlated，相关的。

所以采样的话最好是从多个链中分别采
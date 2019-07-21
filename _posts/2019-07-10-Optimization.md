---
layout: post
title: "Optimization"
category: DL
tags: []
date:   2019-07-10 13:25:35 +0200
---

# Optimization

## Objective Function

$$
\arg \min_\theta O(\mathcal{D} ; \theta)=\sum_{k}^{n} L\left(y_{i}, f\left(x_{i}\right) ; \theta\right)+\Omega(\theta)
$$

Loss function $$L(y, f(x) ; \theta)$$

![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/1.png)

Regularization $$\Omega(\theta)$$

![image-20190719135201173](https://strongman1995.github.io/assets/images/2019-07-10-optimization/2.png)

## First-Order Method

### Gradient（一阶梯度）

$$
\begin{aligned} g &=\nabla_{\theta} J(\theta) \\ g_{i} &=\nabla_{\theta_{i}} J(\theta) \end{aligned}
$$

目标函数的一阶泰勒近似：
$$
J(\theta)=J\left(\theta_{0}\right)+\left(\theta-\theta_{0}\right)^{T} g+\cdots
$$

### Gradient Descent(GD)

在 full batch 上计算梯度：$$\Delta^{t}=\nabla_{\theta} J\left(\theta^{t}\right)$$

更新参数：$$\theta^{t+1}=\theta^{t}-\eta \Delta^{t}$$

时间复杂度取决于数据量和参数维度，会在 zero gradient 的地方（鞍点）停止迭代

### Convex Function

为了保证 GD 算法的收敛性，需要对目标函数做一些假设。

> convex function: 
>
> f is convex if $$\forall \lambda \in[0,1], x, x^{\prime} \in \operatorname{dom}(f)$$:
>
> $$f\left(\lambda x+(1-\lambda) x^{\prime}\right) \leq \lambda f(x)+(1-\lambda) f\left(x^{\prime}\right)$$

关键的性质：$$f(x)-f\left(x^{\prime}\right) \leq \nabla f(x)\left(x-x^{\prime}\right)$$

## Second-Order Method

光一阶是不够的，在高维空间中，梯度为0的点叫**critical points**，有三种可能：

1. maximum
2. minimum
3. saddle point

所以，当我们找到一个梯度为 0 的点，有可能不是最低点，所以这是优化过程中的一个难点。而且鞍点会经常出现。

### Hessian（二阶梯度）

Hessian 矩阵可以表示 curvature(曲率) of function landscape
$$
H_{i j}=\frac{\partial}{\partial \theta_{j}} g_{i}
$$
目标函数的二阶泰勒近似：
$$
J(\theta)=J\left(\theta_{0}\right)+\left(\theta-\theta_{0}\right)^{T} g+\frac{1}{2}\left(\theta-\theta_{0}\right)^{T} H\left(\theta-\theta_{0}\right)+\cdots
$$
对 H 进行特征值分解：
$$
H=Q \Lambda Q^{T} \text { and } H^{-1}=Q \Lambda^{-1} Q^{T}
$$
![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/3.png)

对角阵$$\Lambda$$ 决定了 the local shape of the landscape

![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/4.png)

### Newton's Method（牛顿法）

考虑在$$\theta^t$$的小区域内二阶近似：
$$
\hat{J}(\theta)=J\left(\theta^{t}\right)+\nabla_{\theta} J\left(\theta^{t}\right)\left(\theta-\theta^{t}\right)+\frac{1}{2}\left(\theta-\theta^{t}\right)^{T} H\left(\theta-\theta^{t}\right)
$$
对$$\theta$$求导，设为 0：$$\theta^*$$是最优值
$$
\nabla_{\theta} \hat{J}\left(\theta^{*}\right)=\nabla_{\theta} J\left(\theta^{t}\right)+H\left(\theta^{*}-\theta^{t}\right)=0
$$
牛顿法的更新规则：
$$
\theta^{t+1}=\theta^{t}-H^{-1} \nabla_{\theta} J\left(\theta^{t}\right)
$$
![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/5.png)

牛顿法的优点：比 GD 能够更快地收敛

缺点：计算 Hessian 矩阵的时间复杂度 O(d^2) 空间复杂度 O(d^3)， d 是参数维度，在大规模模型中是 impractical 的

### Conjugate Gradient Descent(共轭梯度下降)

每一轮迭代：

1. 计算$$\beta^{t}=\frac{\nabla_{\theta} J\left(\theta^{t}\right)^{\mathrm{T}} \nabla_{\theta} J\left(\theta^{t}\right)}{\nabla_{\theta} J\left(\theta^{t-1}\right)^{\mathrm{T}} \nabla_{\theta} J\left(\theta^{t-1}\right)}$$
   1. $$\beta^t$$是使得$$\Delta^{t-1}$$和$$\Delta^t$$共轭（正交）：$$\left(\Delta^{t}\right)^{T} H \Delta^{t-1}=0$$

2. 计算$$\Delta^{t}=\nabla_{\theta} J\left(\theta^{t}\right)+\beta^{t} \Delta^{t-1}$$

3. 线性搜索：$$\eta^{*}=\operatorname{argmin}_{\eta} J\left(\theta^{t}-\eta \Delta^{t}\right)$$
   1. $$\eta^*$$是梯度能延伸的最大步长

4. 更新参数：$$\theta^{t+1}=\theta^{t}-\eta^{*} \Delta^{t}$$

共轭梯度下降是一阶方法，只需要计算$$\beta^t$$不需要计算 H 和$$H^{-1}$$ 

使用了牛顿法的思想来调整梯度的方向，避免了在一个方向上的重复计算

![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/6.png)

### Quasi-Newton Method(类牛顿法)

为了避免每次要计算$$H^{-1}$$所以类牛顿法通过 incremental low-rank updates 近似$$H^{-1}$$

给定一个开始位置$$x^{0} \in x, H_{0}>0$$

1. 计算类牛顿方向$$\Delta^{t}=-H_{t-1}^{-1} \nabla f\left(x^{t-1}\right)$$

2. 更新下一个位置$$x^{t}=x^{t-1}+\eta^{t} \Delta^{t}$$
3. 计算$$H_t$$

**BFGS**：
$$
H{t}=H{t-1}+\frac{z z^{T}}{Z^{T} v}-\frac{H{t-1} v v^{T} H{t-1}}{v^{T} H_{t-1} v}
$$
where $$v=x^{t}-x^{t-1}, z=\nabla f\left(x^{t}\right)-\nabla f\left(x^{t-1}\right)$$

# Optimization in DL

## Stochastic Method

### SGD

在一轮 iteration 中：

1. 随机选择一个大小为 m 的 mini batch $$\left\{\left(x^{(i)}, y^{(i)}\right)\right\}_{i=1}^{m}(m \ll n)$$ (在每个 epoch 中，都会 shuffle 数据集来避免 bias 偏差)
2. 设置目标函数：$$t^{t}\left(\theta^{t}\right)=\frac{1}{m} \sum_{i=1}^{m} L\left(\theta^{t} ; x^{(i)}, y^{(i)}\right)+\Omega\left(\theta^{t}\right)$$
3. 在 minibatch 上计算梯度：$$\Delta^{t}=\nabla_{\theta} J^{t}\left(\theta^{t}\right)$$
4. 用合适的学习率$$\eta$$更新参数：$$\theta^{t+1}=\theta^{t}-\eta \Delta^{t}$$

优点：

- Stochasticity 可帮助逃离 saddle points

- 比 full batch 的方法更快
- 在 convex 条件下，理论收敛率：$$O(1 / \sqrt{T})$$ 

缺点：

- 因为梯度是在 minibatch 下计算出来的，所以梯度是 noisy 的，在不同 minibatch 梯度可能更改的很快
- poor local minima 或者 saddle point 可能会导致 0 梯度，在高位空间中出现的概率更大
- 目标函数有很高的 condition number:$$\kappa(H)=\frac{\sigma_{\max }}{\sigma_{\min }}$$
  - ratio of largest to smallest singular value of Hessian

### SGD with Momentum

和 SGD 唯一的差别是梯度计算：$$\Delta^{t}=\beta \Delta^{t-1}+\nabla J^{t}\left(\theta^{t}\right)$$

![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/7.png)

### Nesterov Momentum

$$
\begin{aligned} \widetilde{\theta}^{t} &=\theta^{t}-\beta \Delta^{t-1} \\ \Delta^{t} &=\beta \Delta^{t-1}+\nabla J^{t}\left(\tilde{\theta}^{t}\right) \end{aligned}
$$

在加入 momentum 后计算梯度：保证梯度和 momentum 是同一个 time step。

加快了 momentum-based 的收敛速度

## Adaptive Method

Adaptive Learning Rate: Learning Rate 的选择是对收敛质量非常重要的。

### AdaGrad

1. $$r^{t}=r^{t-1}+ \nabla J^{t}\left(\theta^{t}\right) \odot \nabla J^{t}\left(\theta^{t}\right)$$
2. $$h^{t}=\frac{1}{\sqrt{r^{t}}+\delta}$$, $$\delta$$是一个数值很小的常数，防止除数为 0，
3. $$\Delta^{t}=h^{t} \odot \nabla J^{t}\left(\theta^{t}\right)$$
4. $$\theta^{t+1}=\theta^{t}-\eta \Delta^{t}$$

1，2，3 的维度和参数维度一样

随着迭代，adaptive learning rate 会衰减

### RMSProp

Exponential Moving Average

1. $$r^{t}=\rho r^{t-1}+(1-\rho) \nabla J^{t}\left(\theta^{t}\right) \odot \nabla J^{t}\left(\theta^{t}\right)$$

加上了衰减系数$$\rho$$在之前的梯度上：$$r^{t}=(1-\rho) \nabla J\left(\theta^{t}\right) \odot \nabla J\left(\theta^{t}\right)+\ldots+\rho^{t-1}(1-\rho) \nabla J\left(\theta^{0}\right) \odot \nabla J\left(\theta^{0}\right)$$

和 Highway Network、LSTM 很像

### RMSProp with Momentum

3. $$\Delta^{t}=\beta \Delta^{t-1}+h^{t} \odot \nabla J^{t}\left(\theta^{t}\right)$$

### Adam

和 RMSProp with Momentum 相比，调整了第三步：

3.1 $$s^{t}=\varepsilon s^{t-1}+(1-\varepsilon) \nabla J^{t}\left(\theta^{t}\right)$$

3.2 $$\Delta^{t}=h^{t} \odot s^{t}$$

增加了 adaptive learning rate term for momentum

建议$$\epsilon=0.9，\rho=0.999$$ 

完整版：

1. $$r^{t}=\rho r^{t-1}+(1-\rho)\left\|\nabla J^{t}\left(\theta^{t}\right)\right\|_{2}^{2}$$
2. $$\tilde{r}^{t}=r^{t} /\left(1-(\rho)^{t}\right)$$: Normalize step to eliminate bias
3. $$h^{t}=\frac{1}{\sqrt{\hat{r}^{t}}+\delta}$$
4. $$s^{t}=\varepsilon s^{t-1}+(1-\varepsilon) \nabla J^{t}\left(\theta^{t}\right)$$
5. $$\tilde{s}^{t}=s^{t} /\left(1-(\varepsilon)^{t}\right)$$: Normalize step to eliminate bias
6. $$\Delta^{t}=h^{t} \odot \tilde{s}^{t}$$
7. $$\theta^{t+1}=\theta^{t}-\eta \Delta^{t}$$

### Leraning Rate Decay

在 SGD 中 learningrate 是超参数，为了使网络收敛的平稳快速，可以将 learning rate 设置为随着时间衰减，有两个衰减策略：

1. Exponential decay strategy: $$\eta=\eta_{0} e^{-k t}$$
2. 1/t decay strategy: $$\eta=\eta_{0} /(1+k t)$$

## New Adaptive Method

在一些凸函数中，RMSProp 或者 Adam 可能会失败。这是 Exponential Moving Average 的问题：Adaptive learning rate $$h^t$$可能会随着时间 shrink

### AMSGrad

改写 Adam 的第一步

1.1 $$r^{t}=\rho \hat{r}^{t-1}+(1-\rho)\left\|\nabla J^{t}\left(\theta^{t}\right)\right\|_{2}^{2}$$

1.2 $$\hat{r}^{t}=\max \left\{r^{t}, \hat{r}^{t-1}\right\}$$

所以 AMSGrad 保证了不会有 incresing step size

### AdaBound

改写了 Adam 第2步

2. $$h^{t}=\operatorname{Clip}\left(\frac{1}{\sqrt{\hat{r}^{t}}+\delta}, \frac{\eta_{u}}{\sqrt{t}}, \frac{\eta_{l}}{\sqrt{t}}\right)$$

总结：

![](https://strongman1995.github.io/assets/images/2019-07-10-optimization/8.png)
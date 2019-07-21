---
layout: post
title: "Multilayer Perceptrons"
category: DL
tags: []
date:   2019-07-01 13:25:35 +0200
---

## Perceptron

感知机，只能表示线性可分的，所以不可表示 XOR

## MultiLayer Perceptron

可以表示任何的 Decision Boundary，把各种 Decision Boundary 进行叠加就可以。

### Activation Function

#### sigmoid 

0~1, probability, Relative smaller gradient

#### Hypoblic Tangent

-1~1, zero centered, Relatively larger saturation region

#### Rectified Linear Unit (ReLU)

Time-efficient and faster convergence, but neurons are prone to death.

#### 指数函数 exp

容易溢出，需要做处理：
$$
g(\mathbf{z})_{i}=\frac{e^{z_{i}-z_{m}}}{\sum_{j=1}^{k} e^{z_{j}-z_{m}}}
$$


$$ m=\operatorname{argmax}\left(z_{j}\right)$$

### Entropy

#### Entropy

Amount of bits to encode information (uncertainty) in q
$$
H(q)=-\sum_{j=1}^{k} q_{j} \log q_{j}
$$

$$
H(q, p)=\mathrm{KL}(q \| p)+H(q)
$$

#### Relative Entropy

Amount of extra bits to encode information in p given q:
$$
\mathrm{KL}(q \| p)=-\sum_{j=1}^{k} q_{j} \log p_{j}-H(q)
$$

#### Cross Entropy

$$
H(q, p)=-\sum_{j=1}^{k} q_{j} \log p_{j}=-\frac{1}{N} \log \prod_{j} p_{j}^{N q_{j}}
$$

Cross Entropy = Relative Entropy + Entropy
$$
H(q, p)=\mathrm{KL}(q \| p)+H(q)
$$

Cross-entropy loss:
$$
J(\boldsymbol{y}, \widehat{\boldsymbol{y}})=-\sum_{j=1}^{k} y_{j} \log \hat{y}_{j}
$$
cost function:
$$
\min J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[\sum_{j=1}^{k} \mathbf{1}\left\{y^{(i)}=j\right\} \log \frac{\exp \left(z_{j}^{\left(n_{l}\right)}\right)}{\sum_{j^{\prime}=1}^{k} \exp \left(z_{j \prime}^{\left(n_{l}\right)}\right)}\right]
$$

## Backpropagation

### Automatically Differentiation



## Practical Training Strategies

### Stochastic Gradient Descent

$$
\frac{\partial}{\partial \theta_{i j}^{(l)}} J(\theta, b)=\left[\frac{1}{m} \frac{\partial}{\partial \theta_{i j}^{(l)}} J\left(\theta, b ; x^{(i)}, y^{(i)}\right)\right]
$$

$$
\frac{\partial}{\partial b_{i}^{(l)}} J(\theta, b)=\frac{1}{m} \frac{\partial}{\partial b_{i}^{(l)}} J\left(\theta, b ; x^{(i)}, y^{(i)}\right)
$$

### SGD with momentum

$$
\begin{array}{l}{\theta_{i j}^{(l)}=\theta_{i j}^{(l)}-\eta \Delta} \\ {\Delta=\beta \Delta+\frac{\partial}{\partial \theta_{i}^{(l)}} J(\theta, b)}\end{array}
$$



### Learning rate decay

Though some algorithms can adjust learning rate adaptively, a good choice of learning rate ! could result in better performance. To make network converge stably and quickly, we could set learning rate that decays over time

#### Exponential decay strategy

$$
\eta=\eta_{0} e^{-k t}
$$

#### 1/t decay strategy

$$
\eta=\eta_{0} /(1+k t)
$$

### Weight decay

#### L1 Regularization

#### L2 Regularization

### Dropout

### Weight Initialization

Proper initialization avoids reducing or magnifying the magnitudes of signals exponentially

### Babysitting Learning

Overfit on a small data

1. Training error fluctuating? Decrease the learning rate
2. Some of the units saturated? Scale down the initialization. Properly normalize the inputs

## Expressiveness of DL

### Shallow MLP

Universal: Two-layer MLP can represent any boundary (Hornik, 1991)

Two-layer MLP requires exponentially large number of units KN->无穷

### Space Folding

是讲如何形象地把低维空间非线性映射到高维空间中线性可分
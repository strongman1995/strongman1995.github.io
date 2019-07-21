---
layout: post
title: "Adversarial Learning"
category: DL
tags: []
date:   2019-07-21 13:25:35 +0200
---

traditional ML：optimization：
$$
\min _{\theta} J(\boldsymbol{\theta})
$$
Adversarial ML: game theory:
$$
\min _{\boldsymbol{\theta}_{1}} \max _{\boldsymbol{\theta}_{2}} J\left(\boldsymbol{\theta}_{1}, \boldsymbol{\theta}_{2}\right)
$$

# Generative Adversarial Network(GAN)

## Generative Models

General idea of GAN: for some distribution metric D, 
$$
\theta :=\min _{\theta} D\left(p_{\theta}(x), p_{\hat{x}}(\hat{x})\right)
$$

### Training GANs: Two-Player Game

- Generator network (G)
  - try to fool the discriminator by generating real-looking images(complex map)
- DIscriminator network(D)
  - try to distinguish between real and fake images (two-class classification model)

![](https://strongman1995.github.io/assets/images/2019-07-21-adversarial/1.png)

- Train jointly in minimax objective function:
  $$
  \min _{G} \max _{D}\left[\mathbb{E}_{x \sim P_{\text {data}}} \log D(x)+\mathbb{E}_{z \sim P_{z}} \log (1-D(G(z)))\right]
  $$
  where: D(x) for real data x, D(G(z)) for generated fake data G(z)

- max D s.t. D(x) is close to 1 (real) and 𝐷(𝐺(𝑧)) is close to 0 (fake)
  - gradient ascent on discriminator D
  - small gradient for good samples, large gradient for bad samples
- min G s.t. D(G(z)) is close to 1 (D is fooled into thinking G(z) is real)
  - gradient descent on generator G
  - small gradient for bad samples, large gradient for good samples
  - optimizing generator objective does not work well dut to gradient vanishing

- Two-player game is matching two empirical distributions $$\hat{u}$$ and $$\hat{v}$$

### Wasserstein GAN(WGAN)

> If $$\mathbb{E}_{x \sim P} f(x)=\mathbb{E}_{y \sim Q} f(y)$$ for all function f, then P=Q

Measure distribution distance by dual form Wasserstein:
$$
W(P, Q)=\max _{\|f\|_{L i p} \leq 1}\left(\mathbb{E}_{x \sim P} f(x)-\mathbb{E}_{x \sim Q} f(x)\right)
$$
Using netwrok D to parametrize Lipchitz function class $$\|f\|_{L i p} \leq 1$$:
$$
\max _{D}\left(\mathbb{E}_{x \sim P} D(x)-\mathbb{E}_{\tilde{x} \sim G(N(0, I))} D(\tilde{x})\right) \approx W(P, Q)
$$
Wasserstein GAN:
$$
\begin{array}{c}{\max _{D}\left(\mathbb{E}_{x \sim P} D(x)-\mathbb{E}_{\tilde{x} \sim G(N(0, I))} D(\tilde{x})\right)} \\ {\min _{G} \max _{D}\left(\mathbb{E}_{x \sim P} D(x)-\mathbb{E}_{\tilde{x} \sim G(N(0, I))} D(\tilde{x})\right)}\end{array}
$$
![](https://strongman1995.github.io/assets/images/2019-07-21-adversarial/2.png)



# Adversarial Training

find small noise that can keep appearance of image and cause mistake in network

### Reason for adversarial example

- overfitting
- local linearity(ReLU)
- If a sample is near classification surface while the local surface is not nonlinear enough to fit it, this sample is exposed to the classification surface and an advesarial example can be created nearby

## Attacking Methods

### White-Box Attack

- $$\epsilon$$ bounded attack method
  - targeted attack
  - untargeted attack
  - FGSM
  - PGD

### Black-Box Attack

use transferability to attack that model

## Defense Methods

### Adversarial Training

$$
\tilde{J}(\theta, x, y)=\alpha J(\theta, x, y)+(1-\alpha) J\left(\theta, x+\varepsilon \operatorname{sign}\left(\nabla_{x} J(\theta, x, y)\right), y\right)
$$

$$
\min _{\theta} \mathbb{E}_{(x, y) \sim P} \max _{\delta \in \Delta} J(\theta, x+\delta, y)
$$

$$
\min _{\theta} \mathbb{E}_{(x, y) \sim P} \alpha J(\theta, x, y)+(1-\alpha) \max _{\delta \in \Delta} J(\theta, x+\delta, y)
$$


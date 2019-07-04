---
layout: post
title: "Adversarial Attack Overview"
category: adversarial
tags: []
date:   2019-05-02 13:25:35 +0200
---

# Adversarial Attack Paper List

## [2013 SVM] [Biggio] Evasion attacks against machine learning at test time

University of Cagliari, gradient-based approach，应用于 spam email 的攻击，二分类

test time : evasion attack

## [2013 L-BFGS] [Szegedy] Intriguing properties of neural networks 

Google 的文章，作者中有 Goodfellow（GAN 父，他还在Montreal 大学读博的时候做的），我猜是Szegedy带他入坑的。

这篇文章不知道发在哪里了，但是是非常重要的奠基文章！

这篇文章揭示了神经网络两个很有趣的性质：一个是通过 unit analysis，单独的高层的 unit和不同高层 unit 的线性组合没有什么区别，所以高层 unit 的包含的语义信息是一个空间。而第二个性质就是这个空间，在一定程度上是不连续的，导致在输入加上一些不可分辨的扰动就可以改变最后的分类结果，通过找到这个网络最大的预测错误。

另外还揭示了 adversarial attack 有 transfer 的特性。相同的扰动可以导致在不同训练数据集上训练的模型有相同的错误输出。

当然！L-BFGS 这个 optimization-based 的方法的提出！
$$
\text { Minimize } c|r|+\operatorname{loss}_{f}(x+r, l) \text { subject to } x+r \in[0,1]^{m}
$$

## [2015 FGSM] [Goodfellow] Explaining and harnessing adversarial examples 

此时的 Goodfellow 已经加入了 google，用上了 google 邮箱，通信作者是 Szegedy，我猜是他的 leader。这篇文章也是特别重要的奠基文章之一！发在了 2015 年的 ICLR 上。

在这篇文章中，他们也说明了一个性质，就是神经网络的 vulnerability 不是因为模型的非线性性质和 overfitting，而就是因为 model 在高维空间中的线性性质，才导致了模型的 vulnerability。

而且还证明了 transfer 的性质，他们在不同的训练数据集训练出来的模型，不同的模型架构，使用同样的攻击扰动，可以造成相同的攻击结果。

最重要的是！提出了 FGSM 这么快速简单的产生 adversarial sample 方法啊！然后通过这个方法产生的 samples，用来 augment 数据，可以训练出来更好的 model。FGSM 就是用 loss 对于每个像素的梯度，修改原图的像素。但是是一次性的攻击，就是只修改一次像素值。后面的方法都是迭代地去修改。
$$
\boldsymbol{X}^{a d v}=\boldsymbol{X}+\epsilon \operatorname{sign}\left(\nabla_{X} J\left(\boldsymbol{X}, y_{t r u e}\right)\right)
$$

## [2017 BIM] [Kurakin] Adversarial examples in the physical world 

又是 Google Brain的文章，这篇文章中当然有 Goodfellow 大佬，是二作。这篇文章发表在 ICLR2017 上。

我觉得这篇文章应该是把 Adversarial attack 第一次引入到 real world 中吧，还有个 youtube 小视频来证明他们的攻击效果哈哈。

当然提出了对 FGSM 的改进，BIM——Basic Iterative Method，就是把 FGSM 分成一小步一小步去调整像素点，也是用 loss 对于每个像素的梯度来做的。不过难以置信，为什么这个方法和 FGSM 隔了两年之久才提出来。
$$
\boldsymbol{X}_{0}^{a d v}=\boldsymbol{X}, \quad \boldsymbol{X}_{N+1}^{a d v}=\operatorname{Clip}_{X, \epsilon}\left\{\boldsymbol{X}_{N}^{a d v}+\alpha \operatorname{sign}\left(\nabla_{X} J\left(\boldsymbol{X}_{N}^{a d v}, y_{t r u e}\right)\right)\right\}
$$

## [2017 PGD] [Mądry] Towards deep learning models resistant to adversarial attacks 

这篇文章是 MIT 出品，Mądry大佬也在adversarial这个领域发了不少文章。不过不知道具体发表在了哪里。

这篇文章说到了攻击的对立面——防御，他们提出了一种 General 的方法可以防止 first-order adversary，而这个 attack 的方法，使用了 PGD——projected gradient descent，他们觉得是first order 攻击中最强的。但是我感觉其实就是在 BIM 的基础上再加了个随机初始值。。。
$$
x^{t+1}=\Pi_{x+\mathcal{S}}\left(x^{t}+\alpha \operatorname{sgn}\left(\nabla_{x} L(\theta, x, y)\right)\right)
$$
至于防御，感觉有点像 GAN 的机制，攻击和防御在博弈
$$
\min _{\theta} \rho(\theta), \quad \text { where } \quad \rho(\theta)=\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\max _{\delta \in \mathcal{S}} L(\theta, x+\delta, y)\right]
$$

## [2018 MIM] [Yinpeng Dong] Boosting adversarial attacks with momentum 

朱军组的董大佬在 2018 年发在 CVPR的文章。在打 2017 年 NIPS 的 adversarial attack 的比赛时使用的方法，取得了很好的成绩，他们在前人gradient-based 的基础上加上了 momentum 的机制，使得梯度下降的时候又快又稳。
$$
\boldsymbol{g}_{t+1}=\mu \cdot \boldsymbol{g}_{t}+\frac{\nabla_{\boldsymbol{x}} J\left(\boldsymbol{x}_{t}^{*}, y\right)}{\left\|\nabla_{\boldsymbol{x}} J\left(\boldsymbol{x}_{t}^{*}, y\right)\right\|_{1}}
$$

$$
\boldsymbol{x}_{t+1}^{*}=\boldsymbol{x}_{t}^{*}+\alpha \cdot \operatorname{sign}\left(\boldsymbol{g}_{t+1}\right)
$$

## [2017 C&Wl2] [Carlini&Wagner] Towards evaluating the robustness of neural networks 

这篇是2017 年伯克利两位大佬发在 Symposium on Security and Privacy 安全顶会上的文章。这个方法的名字就是用两位大佬的名字命名的 C&W

他们用的是optimization-based 的方法来生成攻击图片。效果非常的好，比基于梯度的方法改动小很多。但是生成时间会稍微长一些。他们用 C&W 这个方法打了defensive distillation 方法的脸，说并没有很好的提升模型的 robustness。因为用他们的方法攻击还是很成功的。因为当时 defensive distillation 声称能够把现有攻击方法成功率从 95%降到 0.5%。但是 C&W 方法在有 defense distillation 和没有 defense distillation 的模型上都有 100%的成功率，而且 perturbation 还特别小！
$$
\begin{array}{l}{\text { minimize }\left\|\frac{1}{2}(\tanh (w)+1)-x\right\|_{2}^{2}+c \cdot f\left(\frac{1}{2}(\tanh (w)+1)\right.} \\ {\text { with } f \text { defined as }} \\ {\qquad f\left(x^{\prime}\right)=\max \left(\max \left\{Z\left(x^{\prime}\right)_{i} : i \neq t\right\}-Z\left(x^{\prime}\right)_{t},-\kappa\right)}\end{array}
$$

## [2017 EAD] [Pinyu Chen] EAD: elastic-net attacks to deep neural networks via adversarial examples EAD

这篇是 IBM Watson 研究所贡献的文章。这篇文章也是基于优化的方法来生成攻击图。

他们在distance三个尺度（L1 L2 Linfinite）上都做了尝试，
$$
\operatorname{minimize}_{\mathbf{z} \in \mathcal{Z}} f(\mathbf{z})+\lambda_{1}\|\mathbf{z}\|_{1}+\lambda_{2}\|\mathbf{z}\|_{2}^{2}
$$
因为他们加的正则是elastic net 的正则项，所以叫 EAD 吧。

## [2016 JSMA] [Papernot] The limitations of deep learning in adversarial settings 

Papernot 又是另一个大佬，也在这个领域经常看到。这篇是在 2016 的 SSP 发的。

这个方法说是 gradient-based 但是又和前面的不同，因为前面的是通过反向传播，得到的梯度来改变像素值。而这篇与众不同，是通过前向传播，得到哪个像素对最后结果的影响力大，然后做一个 sort，先一个一个改变影响力大的像素点，直到改变最后的分类结果为 target。这个方法相对于前面 gradient-based 的方法做的扰动也是很小，当然成功率也基本保持 95 以上。但是这个方法的缺点就是慢，计算 saliency map 很慢。
$$
S(\mathbf{X}, t)[i]=\left\{\begin{array}{c}{0 \text { if } \frac{\partial \mathbf{F}_{t}(\mathbf{X})}{\partial \mathbf{X}_{i}}<0 \text { or } \sum_{j \neq t} \frac{\partial \mathbf{F}_{j}(\mathbf{X})}{\partial \mathbf{X}_{i}}>0} \\ {\left(\frac{\partial \mathbf{F}_{t}(\mathbf{X})}{\partial \mathbf{X}_{i}}\right)\left|\sum_{j \neq t} \frac{\partial \mathbf{F}_{j}(\mathbf{X})}{\partial \mathbf{X}_{i}}\right| \text { otherwise }}\end{array}\right.
$$

## [2017 Deepfool] [Moosavi-Dezfooli] Deepfool: a simple and accurate method to fool deep neural networks DeepFool

这篇和其他的文章有一点不同，它是 untargeted attack，但是高效有用方法简单。发表在 2017 的 CVPR 上。

他的思路就是，找到最近的分类面，改变的像素能够越过最近的分类面。
$$
\underset{\boldsymbol{r}_{i}}{\arg \min }\left\|\boldsymbol{r}_{i}\right\|_{2} \text { subject to } f\left(\boldsymbol{x}_{i}\right)+\nabla f\left(\boldsymbol{x}_{i}\right)^{T} \boldsymbol{r}_{i}=0
$$

## [2016 Feature adv] [Sabour] Adversarial manipulation of deep representations 

多伦多大学的一篇文章。发表在 2016 的 CVPR 上。

这篇文章的思路也不一样，别人是直接改像素值，他是在模型中的 feature 上做改动，
$$
\begin{array}{r}{I_{\alpha}=\arg \min _{I}\left\|\phi_{k}(I)-\phi_{k}\left(I_{g}\right)\right\|_{2}^{2}} \\ {\text { subject to }\left\|I-I_{s}\right\|_{\infty}<\delta}\end{array}
$$
可以从它每一层的抽象图中看到adversarial attack 从原图转到 target 目标的一个改变

## [2006 CTC] [Graves] Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks

2006年 ICML 的一篇文章，影响很大的一篇文章。CTC 是作为在语音中解码的重要工具。可以自动对齐模型的输出和true label sequence。

一篇教学文：https://distill.pub/2017/ctc/

## [] Towards deep neural network architectures robust to adversarial examples



## Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods



## Deep neural networks are easily fooled: High confidence predictions for unrecognizable images 2015



## Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks



## Bypassing feature squeezing by increasing adversary strength 2018 ICLR 短文



## Adaptive adversarial attack on scene text recognition



## Ensemble adversarial training: Attacks and defenses



## Distillation as a defense to adversarial perturbations against deep neural networks



## Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples


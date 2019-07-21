---
layout: post
title: "Attention"
category: DL
tags: []
date:   2019-07-19 13:25:35 +0200
---

# Attention Mechanism

在这里的注意力，是选择式注意力，在多个因素或者刺激物中选择，排除其他的干扰项。

# Temporal Attention（时间）

1. one to one: MLP

2. many to one: Sentiment classification, sentent->sentiment

3. one to many: Image Captioning, Image->Sentence
4. many to many(heterogeneous): Machine Translation, sentence(en)->sentence(zh)
5. many to many(homogeneous): Language Modeling, Sentence->Sentence

### Autoregressive(AR)— Language Modeling

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/1.png)

## Seq2Seq with Attention—machine translation

### Seq2Seq

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/2.png)
$$
p\left(y_{1}, \ldots, y_{T^{\prime}} \| x_{1}, \ldots, x_{T}\right)
$$

$$
=\prod_{t=1}^{T^{\prime}} p\left(y_{t} \|c,_{1}, \ldots, y_{t-1}\right).
$$

$$
=\prod_{t=1}^{T^{\prime}}  p\left(y_{t} \| c, y_{1}, \ldots, y_{t-1}\right)
$$

$$
=\prod_{t=1}^{T^{\prime}}g\left(c, s_{t-2}, y_{t-1}\right).
$$

优点：

- 直接对$$p\left(y_{1}, \ldots, y_{T^{\prime}} \| x_{1}, \ldots, x_{T}\right)$$建模
- 能够通过反向传播端到端训练

缺点：

- 将 source sequence 全部encode 到了一个hidden state c 中，使用c 去 decoding，这样会有信息丢失
- 梯度反向穿模的路径太长了，尽管有 LSTM forget，还是有梯度消失问题
- 翻译的语序可能在 source 和 target sequence 中不同

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/3.png)

attention 机制可以解决梯度消失的问题，特别是在长句的翻译中

## Global & Local Attention

### Glocal Attention

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/4.png)

### Local Attention

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/5.png)

# Spatial Attention（空间）

### Image Captioning with Spatial Attention

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/6.png)

# Self Attention

### How to calculate attention?

given <u>query vector</u> **q (target state)** and <u>key vector</u> **k (context states)**

- Bilinear: $$a(q, k)=q^{T} W k$$ , require parameters
- Dot Product: $$a(q, k)=q^{T} k$$, increases as dimensions get larger
- Scaled Dot Product: $$a(q, k)=q^{T} k / \sqrt{d_{k}}$$, scale by size of the vector
- Self Attention: $$\operatorname{Attention}(Q, K, V)=\operatorname{Softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$$
  - $$d_k$$ is dimension of Q(queries) and K(keys)
- Multi-Head Attention: $$\text { MultiHead }(Q, K, V)=\text { Concat }\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{o}$$ ,  where $$\text { head }_{\mathrm{i}}=\text { Attention }\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)$$ 
  - Allows the model to jointly attend to information from different representation subspaces at different positions
  - $$W_{i}^{Q}, W_{i}^{K}, W_{i}^{V}$$ reduce dimension from $$d_{model}$$ to $$d_q, d_k, d_v$$

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/7.png)

## Transformer: Attention is All You Need

### Positional Encoding

> Convolutions and recurrence are not necessary for NMT

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/8.png)

### Compare Transformer with RNNs

RNN

- Strength: Powerful at modeling sequences
- Drawback: Their sequential nature makes it quite slow to train; Fail on large-scale language understanding tasks like translation

Transformer

- Strength: Process in parallel thus much faster to train
- Drawback: Fails on smaller and more structured language understanding tasks, or even simple algorithmic tasks such copying a string while RNNs perform well.

## GPT & BERT

### Generative Pre-Training(GPT)

#### Stage one: unsupervised pre-training

Given an unsupervised corpus of tokens $$u=\left\{u_{1}, \dots, u_{n}\right\}$$, maximize likelihood: 

$$L_{1}(u)=\sum_{i} \log P\left(u_{i} \| u_{i-k}, \ldots, u_{i-1} ; \Theta\right)$$

The context vectors of tokens:
$$
h_{0}=U W_{e}+W_{p}
$$
where: 

- U: The context vectors of tokens
- $$W_e$$: The token embedding matrix
- $$W_p$$: The position embedding matrix

$$
h_{l}=\text { transformer_-block }\left(h_{l-1}\right) \quad \forall i \in[1, n]
$$

$$
P(u)=\operatorname{softmax}\left(h_{n} W_{e}^{T}\right)
$$

#### Stage twe: supervised fine-tuning

Given a labeled dataset C, maximize the following objective:
$$
L_{2}(\mathcal{C})=\sum_{x, y} \log P\left(y \| x^{1}, \ldots, x^{m}\right)=\sum_{x, y} \log \left(\operatorname{softmax}\left(h_{l}^{m} W_{y}\right)\right)
$$
where:

- $$h_l^m$$: The final transformer block's activation
- $$W_y$$: A linear output layer

![](https://strongman1995.github.io/assets/images/2019-07-19-attention/9.png)

### BERT

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers (BERT)

Pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers

The pre-trained BERT representations can be fine-tuned with just one additional output layer

The input embeddings is the sum of the token embeddings, the segmentation embeddings and the position embeddings.
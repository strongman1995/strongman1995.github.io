---
layout: post
title: "Attention"
category: DL
tags: []
date: 2019-08-05 13:25:35 +0200
---

# Attention

attention的通用定义如下：

- 给定一组向量集合values，以及一个向量query，attention机制是一种根据该query计算values的加权求和的机制。
- attention的重点就是这个集合values中的每个value的“权值”的计算方法。
- 有时候也把这种attention的机制叫做query的输出关注了（或者说叫考虑到了）原文的不同部分。（Query attends to the values）



attention机制就是一种根据某些规则或者某些额外信息（query）从向量表达集合（values）中抽取特定的向量进行加权组合（attention）的方法。简单来讲，只要我们从部分向量里面搞了加权求和，那就算用了attention。



针对attention的变体主要有两种方式：

1. 在attention 向量的加权求和计算方式上进行创新
2. 在attention score（匹配度/权值）的计算方式上进行创新

## Attention向量计算方式变体

- Soft attention、global attention、动态attention
- Hard attention
- “半软半硬”的attention （local attention）
- 静态attention
- 强制前向attention

在encoder的过程中保留每个RNN单元的隐藏状态（hidden state）得到（h1……hN）。假设第 t 步的（根据上一步隐藏状态输出与当前输入得到的）隐藏状态为$$S_t$$，在每个第 t 步利用 $$S_t$$ 和 $$h_i$$ 进行 dot 点积得到 attention score，也称为“**相似度**“或者“**影响度**”，或者“**匹配得分**”

### Soft attention/global attention/动态attention

这三个其实就是Soft attention，也就是我们上面讲过的那种最常见的attention，是在求注意力分配概率分布的时候，对于输入句子 X 中任意一个单词都给出个概率，是个概率分布，把attention变量（context vecor）用 $$c_t$$ 表示，attention 得分在经过了 softmax 过后的权值用 $$\alpha$$ 表示(利用 softmax 函数将 attention scores 转化为概率分布)

论文：Neural machine translation by jointly learning to align and translate

![](https://strongman1995.github.io/assets/images/2019-08-05-attention/1.png)

### Hard attention

Soft是给每个单词都赋予一个单词match概率，那么如果不这样做，直接从输入句子里面找到某个特定的单词，然后把目标句子单词和这个单词对齐，而其它输入句子中的单词硬性地认为对齐概率为0，这就是Hard Attention Model的思想。

![](https://strongman1995.github.io/assets/images/2019-08-05-attention/2.png)

### local attention

Soft attention 每次对齐的时候都要考虑前面的 encoder 的所有 hi，所以计算量会很大，因此一种朴素的思想是**只考虑部分窗口内的encoder隐藏输出**，其余部分为0，在窗口内使用softmax的方式转换为概率。

这个 local attention 相反概念的是 global attention，global attention其实就是softmax attention，这里不多赘述global attention了。

论文：Effective Approaches to Attention-based Neural Machine Translation

在这个模型中，对于是时刻t的每一个目标词汇，模型首先产生一个对齐的位置 pt（aligned position），context vector 由编码器中一个集合的隐藏层状态计算得到，编码器中的隐藏层包含在窗口[pt-D, pt+D]中，D的大小通过经验选择。

![](https://strongman1995.github.io/assets/images/2019-08-05-attention/3.png)

上式之中，大S指的是源句子的长度，Wp和vp是指的模型的参数，通过训练得到，为了支持pt附近的对齐点，设置一个围绕pt的高斯分布，其中小s是在以pt为中心的窗口中的整数，pt是一个在[0，S]之间的实数。小Sigma σ 一般取窗口大小的一半。

### 静态 attention

静态attention：对输出句子共用一个 St 的attention就够了，一般用在BiLSTM的首位hidden state输出拼接起来作为st（在图所示中为u）

论文：Teaching Machines to Read and Comprehend 以及 Supervised Sequence Labelling with Recurrent Neural Networks

这个前面的每个hidden state 不应该都和这里的 u 算一次attention score吗，怎么这里只有一个r和u进行了交互？

其实 r 表示的是加权平均的self attention，这个权就是attention $$c_t$$ 向量，这个图里面把attention $$c_t$$ 的计算过程省略了。直接跳到了$$c_t$$ 和$$s_t$$ 计算真正的s’t的部分。他这个里面用的实际的attention score的计算并不是用点积，是additive attention。

什么是additive attention呢？这个下面就会讲根据按照attention score计算的不同的attention model变体。

![](https://strongman1995.github.io/assets/images/2019-08-05-attention/4.png)

## Attention score计算方式变体

讲完了在加权向量αi（注意，此处这个αi指的是得分softmax之后的向量的第i个分量）的计算上面的各种attention变体，我们现在再回到原始的soft attention公式上面来。在 softmax 的 Attention 向量的计算总是依赖于权重求和，而权重往往是 attention score 的 softmax。

已有$$\boldsymbol{h}_{1}, \ldots, \boldsymbol{h}_{N} \in \mathbb{R}^{d_{1}}$$ 的情况下，计算 query $$s \in \mathbb{R}^{d_{2}}$$ 的 attention 向量 a（很多时候也称作上下文向量，context vector）使用的公式为：
$$
\begin{array}{c}{\alpha=\operatorname{softmax}(\boldsymbol{e}) \in \mathbb{R}^{N}} \qquad (take \ softmax)\\ 
{\boldsymbol{a}=\sum_{i=1}^{N} \alpha_{i} \boldsymbol{h}_{i} \in \mathbb{R}^{d_{1}}} \qquad (take \ weighted\  sum)
\end{array}
$$
式子中的变量e代表的就是 attention score，$$\alpha$$ 是attention的权重，a就是context vector

根据 attention score 的计算方式不同，我们可以将 attention 进一步细分分类。attention score 的计算主要有以下几种：

- 点积 attention score（Basic dot-product attention）:
  $$
  e_{i}=s^{T} h_{i} \in \mathbb{R}
  $$

​         注意点积attention score这里有个假设，就是s和h的维数要一样才能进行点积

- 乘法 attention score（multiplicative attention）：
  $$
  e_{i}=s^{T} W h_{i} \in \mathbb{R}
  $$

​          W矩阵是训练得到的参数，维度是d2 x d1，d2是s的hidden state输出维数，d1是hi的hidden state维数

- 加法 attention score（Addictive attention）：
  $$
  e_{i}=\boldsymbol{v}^{T} \tanh \left(\boldsymbol{W}_{1} \boldsymbol{h}_{i}+\boldsymbol{W}_{2} \boldsymbol{s}\right) \in \mathbb{R}
  $$

​         对两种hidden state 分别再训练矩阵然后激活过后再乘以一个参数向量变成一个得分。W1 = d3xd1，W2 = d3*d2，v = d3x1 ，d1，d2，d3分别        为h和s还有v的维数，属于超参数。

## 特殊Attention

### Self Attention

Self attention也叫做 intra-attention 在没有任何额外信息的情况下，我们仍然可以通过允许句子使用 self attention 机制来处理自己，从句子中提取关注信息。

1. 以**当前的隐藏状态**去计算和**前面的隐藏状态**的得分，作为**当前隐藏单元的attention score**，例如：
   $$
   e_{h_{i}}=h_{t}^{T} W h_{i}
   $$

2. 以**当前状态本身**去计算得分作为当前单元 attention score，这种方式更常见，也更简单，例如：
   $$
   \begin{array}{l}{e_{h_{i}}=v_{a}^{T} \tanh \left(W_{a} h_{i}\right)} \\ {e_{h_{i}}=\tanh \left(\boldsymbol{w}^{T} h_{i}+\mathrm{b}\right)}\end{array}
   $$

公式一，其中$$v_a$$和$$W_a$$是参数，通过训练得到，$$W_a$$维数dxd，$$h_i$$维数dx1，$$v_a$$维数dx1。

公式二，前面的$$\boldsymbol{w}$$是dx1的向量，$$h_i$$是dx1，b是（1，）

3. 针对 2，有矩阵变形：令矩阵 $$\mathrm{H}=[\mathrm{h} 1, \mathrm{h} 2, \mathrm{h} 3, \mathrm{h} 4 \ldots \ldots . \mathrm{hn}] \in \mathrm{n} \times 2 \mathrm{u}$$ 表示句子的隐藏状态矩阵，每个隐藏状态为 2u 维：
   $$
   \begin{array}{l}{A=\operatorname{softmax}\left(V_{a} \tanh \left(W_{a} H^{T}\right)\right)} \\ {C=A H}\end{array}
   $$

上面的式子中，使用的$$V_a$$向量是一个通用的向量，每个隐藏状态并没有区分，如果我们对不同状态计算的时候学习不同的向量$$V_a$$，也就是一个$$V_a$$矩阵，得到的就是一个attention矩阵A。

$$H$$是nx2u(双向lstm的结果拼接，每个单向LSTM的hidden units是u），$$W_a$$是dx2u，$$V_a$$是rxd，得到的attention矩阵 A 是 rxn。
C = rx2u，按论文意思来看就是把一句话编码成了一个 r x 2u 的矩阵（**sentence embedding**）

在实际操作的过程中，我们也可能在学习时候给A一个 F2范式约束项 来**避免过拟合**。

### Key-value attention

这种 attention 就是比较新的了，简单来说 Key-value attention 是将 $$h_i$$ 拆分成了两部分$$\left[k e y_{t} ; v a l u e_{t}\right]$$， 然后使用的时候只针对 key 部分**计算 attention 权重**，然后**加权求和**的时候只使用 value 部分进行加权求和。

公式如下，attention权重计算如下：
$$
\begin{array}
{rlrl}{\left[\begin{array}{c}{\boldsymbol{k}_{t}} \\ 
{\boldsymbol{v}_{t}}\end{array}\right]} & {=\boldsymbol{h}_{t}} & {} & {\in \mathbb{R}^{2 k}} \\ 
{\boldsymbol{M}_{t}} & {=\tanh \left(\boldsymbol{W}^{Y}\left[\boldsymbol{k}_{t-L} \cdots \boldsymbol{k}_{t-1}\right]+\left(\boldsymbol{W}^{h} \boldsymbol{k}_{t}\right) \mathbf{1}^{T}\right)} & {} & {\in \mathbb{R}^{k \times L}} \\ 
{\boldsymbol{\alpha}_{t}} & {=\operatorname{softmax}\left(\boldsymbol{w}^{T} \boldsymbol{M}_{t}\right)} & {} & {\in \mathbb{R}^{1 \times L}} \\ 
{\boldsymbol{r}_{t}} & {=\left[\boldsymbol{v}_{t-L} \cdots \boldsymbol{v}_{t-1}\right] \boldsymbol{\alpha}^{T}} & {} & {\in \mathbb{R}^{k}} \\ 
{\boldsymbol{h}_{t}^{*}} & {=\tanh \left(\boldsymbol{W}^{r} \boldsymbol{r}_{t}+\boldsymbol{W}^{x} \boldsymbol{v}_{t}\right)} & {} & {\in \mathbb{R}^{k}}\end{array}
$$
论文：Daniluk, M., Rockt, T., Welbl, J., & Riedel, S. (2017). Frustratingly Short Attention Spans in Neural Language Modeling. In ICLR 2017.

其中呢，这个 L 是 attention 窗口长度
$$W^Y = k\times k$$, $$k_i = k\times 1$$，$$W^h = k\times k$$，$$1^T$$是一个$$1\times L$$的向量
$$w$$ 是一个$$k\times 1$$的向量
$$W^r = k\times k$$，
$$W^x  = k\times k$$
最后得到 $$k\times 1$$的 $$h_t^*$$ 与 $$h_t$$ 一起参与下一步的运算

### multi-head attention

最后简单讲一下 google 的 attention is all you need 里面的 attention，这一段基本上是摘抄的苏剑林的科学空间的博文了。

Google 在 attention is all you need 中发明了一种叫 transformer 的网络结构，其中用到了multi-head attention。

首先，google 先定义了一下 attention 的计算，也是定义出 key，value，query 三个元素（在 seq2seq 里面，query 是$$s_t$$，key 和 value 都是 $$h_i$$）。

在self 里面，query 是当前要计算的$$h_i$$，k 和 v 仍然一样，是其他单元的 hidden state。

在 key-value attention 里面 key 和 value 则是分开了的。

然后除以了一下 $$\sqrt{d_k}$$，为了让内积不至于太大（太大的话 softmax 后就非 0 即 1 了，不够 “soft” 了）

这里我们不妨假设，Q 是 $$n\times d_k$$，K 是 $$m\times d_k$$，V 是 $$m\times d_v$$，忽略归一化和 softmax 的话就是三个矩阵相乘，得到的是 $$n\times d_v$$ 的矩阵。

我们可以说，通过这么一个 attention 层，就将一个 $$n\times d_k$$的序列Q，提取信息编码成 $$n\times d_v$$的序列了。

$$W_i$$ 用来先在算 attention 对三个矩阵做不同的矩阵变换映射一下，变成 $$n\times d_{k’}$$，mxdk’，mxdv’维度。

最后做并联，有点类似于inception 里面多个卷积核的feature map并联的感觉。
$$
\text {Attention }(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})=\operatorname{softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^{\top}}{\sqrt{d_{k}}}\right) \boldsymbol{V}
$$
从一个向量出发
$$
\text {Attention}\left(\boldsymbol{q}_{t}, \boldsymbol{K}, \boldsymbol{V}\right)=\sum_{s=1}^{m} \frac{1}{Z} \exp \left(\frac{\left\langle\boldsymbol{q}_{t}, \boldsymbol{k}_{s}\right\rangle}{\sqrt{d_{k}}}\right) \boldsymbol{v}_{s}
$$

$$
h e a d_{i}=\operatorname{Attention}\left(\boldsymbol{Q W}_{i}^{Q}, \boldsymbol{K} \boldsymbol{W}_{i}^{K}, \boldsymbol{V} \boldsymbol{W}_{i}^{V}\right)
$$



![](https://strongman1995.github.io/assets/images/2019-08-05-attention/5.png)

# 总结

总的来说，attention的机制就是一个加权求和的机制，只要我们使用了加权求和，不管你是怎么花式加权，花式求和，只要你是根据了已有信息计算的隐藏状态的加权和求和，那么就是使用了attention。

而所谓的self attention就是仅仅在句子内部做加权求和（区别与seq2seq里面的decoder对encoder的隐藏状态做的加权求和）。

self attention我个人认为作用范围更大一点，而key-value其实是对attention进行了一个更广泛的定义罢了，我们前面的attention都可以套上key-value attention，比如很多时候我们是把k和v都当成一样的来算，做self的时候还可能是quey=key=value。



# Reference

[1]: https://blog.csdn.net/hahajinbu/article/details/81940355


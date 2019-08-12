---
layout: post
title: "Transformer"
category: DL
tags: []
date: 2019-08-12 13:25:35 +0200
---

# Transformer

从**编码器**输入的句子首先会经过一个**自注意力（self-attention）层**，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。自注意力层的输出会传递到**前馈（feed-forward）神经网络**中。每个位置的单词对应的前馈神经网络都完全一样（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。

**解码器**中也有编码器的**自注意力（self-attention）层**和**前馈（feed-forward）层**。除此之外，这两个层之间还有一个**注意力层**，用来关注输入句子的相关部分（和seq2seq模型的注意力作用相似）。

![](/Users/chenlu/strongman1995.github.io/assets/images/2019-08-12-transformer/1.jpg)

word2vec过程只发生在**最底层的编码器**中。所有的编码器都有一个相同的特点，即它们接收一个向量列表，列表中的每个向量大小为512维。在底层（最开始）编码器中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出（也是一个向量列表）。向量列表大小是我们可以设置的超参数——一般是我们训练集中最长句子的长度。

Transformer的一个核心特性：在这里输入序列中每个位置的单词都有自己独特的路径流入编码器。在自注意力层中，这些路径之间存在依赖关系。而前馈（feed-forward）层没有这些依赖关系。因此在前馈（feed-forward）层时可以并行执行各种路径。

![](/Users/chenlu/strongman1995.github.io/assets/images/2019-08-12-transformer/2.jpg)











# Reference

[1]: https://baijiahao.baidu.com/s?id=1622064575970777188&amp;wfr=spider&amp;for=pc


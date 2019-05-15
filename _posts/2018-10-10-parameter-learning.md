概率图分为三大部分：
 - Representation	P <=>{P,G}
	
	 - parent→child structure(BN) & clique(MN)
	 - Gaussian model & exponential families
	 
 - Inference		P(Y|E=e, θ)

	- Particle-based inference
	- Inference as optimization
 - Learning
 
 	- $\underset{\theta}{\operatorname{max}} P(x[1], x[2], ..., x[M]|\theta)$
 	- P(θ|x[1], x[2], ..., x[M])
### Learning Basics
Learning：从观测数据中构建模型。
根据不同的准则，定义合适的 loss 或者 likelihood function
准则包括：
1. minimum error,
2. probability (maximum likelihood, maximum a posterior )
3. maximum margin
4. compressive sensing

在概率框架下的学习：
1. parameter learning：{x[m]}~m=1-M~|~G~→P(θ|D)
2. Structure learning：{x[m]}~m=1-M~→P(G, θ|D)

- Generative Models: 学习 joint probability P(X=x, Y)
- Discriminative model: 学习conditional probability P(Y|X=x)

**避免 Overfitting**
1. 如果模型复杂度比训练数据更大，能得到非常“好”的学习，0 empirical risk，但是在 测试集表现会很差
2. 增加泛化，需要去惩罚模型的复杂度，分离训练和测试数据

在理论层面：**模型的复杂度需要适应数据的复杂度**，loss function 中的正则项
在经验层面：cross-validation(LOOCV/N-fold CV)，0.632 boostraping(1-e^-1^)
什么是0.632 bootstrapping？
从 M 个样本中有放回地抽取 M 次，那么这些数据大概率会有0.632是不重复的数据，这些数据作为 training 集，剩下没有被抽到的作为 test 集。最后 performance：0.368*Performance~Train~ + 0.632*Performance~Test~

判断是否 overfitting: 在 test 集的 performance 是否严重低于training 集的 performance

**参数估计**
如果一组数据的分布参数不知道，那么这组数据的联合分布不会等于每个数据的概率乘积。假设知道了分布参数，那所有的 samples 是独立同分布的，即 i.i.d
![在这里插入图片描述](http://127.0.0.1:4000/assets/images/2018-10-10-parameter-learning/1.png)
### 最大似然参数估计 Maximum Likelihood Parameter Estimation（MLE）
首先，什么是 likelihood？
likelihood 是给定分布参数的 probability 或者是 confidence
而 log likelihood 更经常被使用，为了更好的计算$log P(D|\theta)=\sum_i log \phi_i(x[i];\theta)$
**MLE**：找到是的 likelihood 最大的参数赋值θ，参数估计的一种方法，
### Bayesian Parameter Estimation

### MAP Parameter Estimation
---
layout: post
title: "生成学习算法"
subtitle: "斯坦福大学机器学习第四讲"
date: 2016-4-12
author: "Anyinlover"
catalog: true
tags:
  - Ng机器学习系列
  - 机器学习算法
---

前面的回归算法和感知机算法都属于判别学习算法，这一章聊聊另一类算法：生成学习算法。这两者的区别可以用以下的比喻，判别学习算法是根据所有猫猫狗狗的特征，建立一个模型，区分出两类动物的分界线。生成学习算法则是分别对猫和狗建立一个模型，然后去对照看跟哪个模型更像。

生成学习算法的理论依据就是贝叶斯公式，体现了后验概率$$p({y \mid x})$$和先验概率$$p(y)$$之间的关系：

$$
p(y \mid x)=\frac{p(x \mid y)p(y)}{p(x)}
$$

对于生成学习算法而言，实际分母一致，不需要计算，只需要比较分子大小：

$$
\begin{align}
\arg \max_y p(y \mid x) &= \arg \max_y \frac{p(x \mid y)p(y)}{p(x)}\\
&=\arg \max_y p(x \mid y)p(y)
\end{align}
$$

## 高斯判别分析
当假设$$p(x \mid y)$$分布遵循多元正态分布时，我们可以使用高斯判别分析算法。在这之前先简要介绍多元正态分布。

### 多元正态分布
当正态分布从一元拓展到多元时，正态分布概率密度函数也需要做出相应的改变：

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2} \mid  \Sigma  \mid ^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

其中，$$\mu \in \mathbb{R}^n$$ 是一个平均数向量，$$\Sigma \in \mathbb{R}^{n\times n}$$是一个协方差矩阵。粗略一看，多元正态分布的表达式和一元正态分布还是有几分神似的。

### 高斯判别分布模型
对于二元分类问题，满足以下假设时可以使用高斯判别分布模型：

$$
\begin{align}
y &\sim Bernoulli(\phi) \\
x \mid y=0 &\sim \mathcal{N}(\mu_0,\Sigma)\\
x \mid y=1 &\sim \mathcal{N}(\mu_1,\Sigma)
\end{align}
$$

即下列分布函数：

$$
\begin{align}
p(y) &= \phi^y(1-\phi)^{1-y} \\
p(x \mid y=0) &= \frac{1}{(2\pi)^{n/2} \mid  \Sigma  \mid ^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)\right)\\
p(x \mid y=1) &= \frac{1}{(2\pi)^{n/2} \mid  \Sigma  \mid ^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)
\end{align}
$$

注意两个多元正态分布的平均数向量不同，协方差矩阵是一致的。然后我们就可以求解最大似然函数了：

$$
\begin{align}
\ell(\phi,\mu_0,\mu_1,\Sigma) &= \log \prod_{i=1}^m p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)\\
&= \log \prod_{i=1}^m p(x^{(i)} \mid y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi)
\end{align}
$$

各参数可解得：

$$
\begin{align}
\phi &= \frac{1}{m}\sum_{i=1}^m1\{y^{(1)}\} \\
\mu_0 &= \frac{\sum_{i=1}^m1\{y^{i}=0\}x^{(i)}}{\sum_{i=1}^m1\{y^{i}=0\}}\\
\mu_1 &= \frac{\sum_{i=1}^m1\{y^{i}=1\}x^{(i)}}{\sum_{i=1}^m1\{y^{i}=1\}}\\
\Sigma &= \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T
\end{align}
$$

### 高斯判别分布和逻辑回归
从高斯判别分布可以推出逻辑回归（表问我怎么推~）：

$$p(y=1 \mid x;\phi,\Sigma,\mu_0,\mu_1)=\frac{1}{1+\exp(-\theta^Tx)}$$

实际上高斯判别分布应用了更强的假设，即$$p(x \mid y)$$是一个多元高斯分布。因此可以从高斯判别分布可以推导到逻辑回归，但满足逻辑回归的不一定满足高斯判别分布，比如泊松判别分布也能推导到高斯分布。

这也造成了高斯判别分布的性质，当原始数据吻合高斯分布时，这是一种很有效很精确的算法。但更多时候数据不能很好的满足高斯分布，这时候高斯判别分布就失效了，相比而言，逻辑回归更稳定，也更常用。

## 朴素贝叶斯
朴素贝叶斯是个大名鼎鼎的算法，不同于高斯判别分布应用于连续型输入，朴素贝叶斯应用于离散型输入，其最常用于文本分类中，比如判别是否为垃圾邮件。

在文本分类中，一张词汇表作为一个特征向量，文本中含这个词则为1，不含则为0，最后的结果示例如下：

$$x=\begin{bmatrix}1\\0\\0\\\vdots\\1\\\vdots\\0\end{bmatrix}$$

要实现朴素贝叶斯算法，需要朴素贝叶斯假设成立。即特征值取值概率是完全相互独立的：

$$p(x_1,\cdots,x_n \mid y)=\prod_{i=1}^n p(x_i \mid y)$$

即使建立在这种强假设上，朴素贝叶斯经常很好使。

根据贝叶斯法则，需要确定的模型参数有$$\phi_{i \mid y=1}=p(x_i=1 \mid y=1),\phi_{i \mid y=0}=p(x_i=1 \mid y=0),\phi_y=p(y=1)$$。计算最大似然函数：

$$\mathcal{L}(\phi_y,\phi_{j \mid y=0},\phi_{j \mid y=1})=\prod_{i=1}^m p(x^{(i)},y^{(i)})
$$

求最大值可以解出各参数值。

$$
\begin{align}
\phi_{j \mid y=1}&=\frac{\sum_{i=1}^m1\{x_j^{(i)}=1\wedge y^{(i)}=1\}}{\sum_{i=1}^m1\{y^{(i)}=1\}}\\
\phi_{j \mid y=0}&=\frac{\sum_{i=1}^m1\{x_j^{(i)}=1\wedge y^{(i)}=0\}}{\sum_{i=1}^m1\{y^{(i)}=0\}}\\
\phi_y&= \frac{\sum_{i=1}^m1\{y^{(i)}=1\}}{m}
\end{align}
$$

$$\wedge$$表示与运算，两边都满足才为真。这三个表达式其实都很直观，就是计数。总数里为1占得比例。

预测函数也可以给出来：

$$
\begin{align}
p(y=1 \mid x)&=\frac{p(x \mid y=1)p(y=1)}{p(x)}\\
&=\frac{(\prod_{i=1}^np(x_i \mid y=1))p(y=1)}{(\prod_{i=1}^np(x_i \mid y=1))p(y=1)+(\prod_{i=1}^np(x_i \mid y=0))p(y=0)}
\end{align}
$$

对于连续型输入，也可以通过分段处理来离散化，然后使用朴素贝叶斯算法。

朴素贝叶斯算法这一块缺少实践感悟，后续需要再来研究。

### 拉普拉斯平滑处理
朴素贝叶斯算法存在一个问题，对于稀疏数据敏感。比如文本分类时有从未出现过的词，则

$$
\phi_{j \mid y=1}=\phi_{j \mid y=0}=0
$$

$$p(y=1 \mid x)=\frac{0}{0}$$

这样计算就会出问题，换一个角度，从未出现过的词不代表以后也不会出现。因此简单把其概率置为0是不合理的。

对于多元分类问题，假设z取值{1,...,k}，进行m次独立观察，则根据朴素贝叶斯算法：

$$
\phi_j=\frac{\sum_{i=1}^m 1\{z^{(i)}=j\}}{m}
$$

利用拉普拉斯平滑处理，可以解决这个问题，保证了每种情况至少有一个大于0的概率。同时保证$$\sum_{j=1}^k \phi_j=1$$。

$$
\phi_j=\frac{\sum_{i=1}^k 1\{z^{(i)}=j\}+1}{m+k}
$$

回头看上一节的垃圾邮件朴素贝叶斯，应用拉普拉斯平滑处理后：

$$
\begin{align}
\phi_{j \mid y=1}&=\frac{\sum_{i=1}^m1\{x_j^{(i)}=1\wedge y^{(i)}=1\}+1}{\sum_{i=1}^m1\{y^{(i)}=1\}+2}\\
\phi_{j \mid y=0}&=\frac{\sum_{i=1}^m1\{x_j^{(i)}=1\wedge y^{(i)}=0\}+1}{\sum_{i=1}^m1\{y^{(i)}=0\}+2}
\end{align}
$$

注意在垃圾邮件的实践应用中，$$\phi_y$$可以不用拉普拉斯平滑处理，因为一般正常邮件和垃圾邮件会有一个比较合理的比例。不可能出现为0的情况。

### 文本分类事件模型
前面使用的朴素贝叶斯模型被称为多元伯努利事件模型，在文本分类中，还有另一种针对邮件而不是词汇表处理的朴素贝叶斯模型，称为多项式事件模型。

我们让$$x_i$$表示邮件中第i个词，在{1,..., $$|V|$$}中取值，$$|V|$$ 代表词汇表大小。每一个词取值可能性相同，即满足多项式分布。

模型的参数有$$\phi_y=p(y),\phi_{k \mid y=1}=p(x_j=k \mid y=1),\phi_{k \mid y=0}=p(x_j=k \mid y=0)$$，对于训练集$$\{(x^{(i)},y^{(i)});i=1,...,m\}, x^{(i)}=(x_1^{(i)},x_2^{(i)},\cdots,x_{n_i}^{(i)}),n_i$$代表第i个训练集中单词总量。得出最大似然函数：

$$
\begin{align}
\mathcal{L}(\phi_y,\phi_{k \mid y=0},\phi_{k \mid y=1})&=\prod_{i=1}^m p(x^{(i)},y^{(i)})\\
&=\prod_{i=1}^m\left(\prod_{j=1}^mp(x_j^{(i)} \mid y;\phi_{k \mid y=0},\phi_{k \mid y=1})\right)p(y^{(i)};\phi_y)
\end{align}
$$

求最大值求解得到：

$$
\begin{align}
\phi_{k \mid y=1}&=\frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x_j^{(i)}=k\wedge y^{(i)}=1\}}{\sum_{i=1}^m1\{y^{(i)}=1\}n_i}\\
\phi_{k \mid y=0}&=\frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x_j^{(i)}=k\wedge y^{(i)}=0\}}{\sum_{i=1}^m1\{y^{(i)}=0\}n_i}\\
\phi_y&= \frac{\sum_{i=1}^m1\{y^{(i)}=1\}}{m}
\end{align}
$$

对其使用拉普拉斯平滑：

$$
\begin{align}
\phi_{k \mid y=1}&=\frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x_j^{(i)}=k\wedge y^{(i)}=1\}+1}{\sum_{i=1}^m1\{y^{(i)}=1\}n_i+ \mid V \mid }\\
\phi_{k \mid y=0}&=\frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x_j^{(i)}=k\wedge y^{(i)}=0\}+1}{\sum_{i=1}^m1\{y^{(i)}=0\}n_i+ \mid V \mid }
\end{align}
$$

朴素贝叶斯不是最好的分类方法，但常常很有效。由于其简洁简单，朴素贝叶斯经常值得一试。

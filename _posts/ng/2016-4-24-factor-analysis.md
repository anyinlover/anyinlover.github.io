---
layout: single
title: "因子分析"
subtitle: "斯坦福大学机器学习第十讲"
date: 2016-4-24
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
  - 机器学习理论
---

在前面的混合高斯模型中，我们常常假定我们有充足的样本去发现数据的内在结构。也就是样本数m远远大于特征数n。

现在考虑$$n \gg n$$的情况，在这样的条件下，单高斯模型都无法拟合，更不论混合高斯模型了。根据最大似然估计，高斯模型的拟合参数如下：

$$
\begin{align}
\mu &= \frac {1} {m} \sum_{i=1}^m x^{(i)} \\
\Sigma &= \frac {1} {m} \sum_{i=1}^m (x^{(i)}-\mu) (x^{(i)}-\mu)^T
\end{align}
$$

我们会发现$$\Sigma$$是奇异阵，$$\Sigma^{-1}$$不存在。这样就无法计算模型的密度函数了。

进一步讲，要通过最大似然估计来拟合高斯模型，必须让m远大于n，才能有比较好的结果。

那么我们如何解决这个样本数不足的问题呢？

## 限制$$\Sigma$$
如果我们没有足够的数据来拟合一个协方差矩阵，我们可以为其添加一些限制。比如考虑协方差矩阵是对角的，也就是特征间是相互独立的，在这种情况下：

$$\Sigma_{jj} = \frac {1} {m} \sum_{i=1}^m (x_j^{(i)}-\mu_j)^2$$

二元高斯分布在平面的投影是个椭圆，对角阵意味着椭圆轴线与坐标轴平行。

进一步，我们还能控制$$\Sigma$$不仅是对角的，而且对角元素相同。$$\Sigma= \sigma^2 I$$，$$\sigma^2$$可以通过最大似然估计得到：

$$\sigma^2 = \frac {1}{mn} \sum_{j=1}^n \sum_{i=1}^m (x_j^{(i)}-\mu_j)^2 $$

在高斯分布平面投影上椭圆变成了圆。

假如不对$$\Sigma$$做限制，我们必须在$$m \geq n+1$$的条件下才能保证$$\Sigma$$不是一个奇异阵，在上面的约束下，只需要$$m \geq 2$$就能保证非奇异。

但上述的假设太强，意味着特征之间完全相互独立，假如我们想要挖掘数据内部的关系时，就需要使用到因子分析模型。

## 边缘和条件高斯分布
在描述因子分析之前，我们先来讨论一下如何找到联合多元高斯分布的条件分布和边缘分布。

假定我们有一个随机变量：

$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

这里$$x_1 \in \mathbb{R}^r, x_2 \in \mathbb{R}^s, \Sigma_{11} \in \mathbb{R}^{r\times r}, \Sigma_{12} \in \mathbb{R}^{r \times s}$$ 假定 $$ x \sim \mathcal{N} (\mu, \Sigma) $$，有：

$$ \mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix},
\Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix} $$

$$x_1$$ 和$$x_2$$被称为联合多元分布，那么$$x_1$$的边缘分布是什么？很容易可以看到$$E[x_1]=\mu_1$$, $$Cov(x_1) = E[(x_1-\mu_1)(x_1-\mu_1)]=\Sigma_{11}$$
。因为根据协方差的定义：

$$
\begin{align}
Cov(x) &= \Sigma \\
&= \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}
    \end{bmatrix} \\
&= E[(x-\mu)(x-\mu)^T] \\
&= E[\begin{pmatrix} x_1-\mu_1 \\ x_2-\mu_2 \end{pmatrix}
{\begin{pmatrix} x_1-\mu_1 \\ x_2-\mu_2 \end{pmatrix}}^T] \\
&= E \begin{pmatrix} (x_1-\mu_1)(x_1-\mu_1)^T (x_1-\mu_1)(x_2-\mu_2)^T \\
(x_2 - \mu_2)(x_1 - \mu_1)^T (x_2-\mu_2)(x_2-\mu_2)^T \end{pmatrix}
\end{align}
$$

因此可以得出$$x_1$$的边缘分布是$$x_1 \sim \mathcal{N} (\mu_1, \Sigma_{11})$$。

下面再来考虑在$$x_2$$给定下$$x_1$$的条件分布。可以记作$$x_1 \mid x_2 \sim \mathcal{N} (\mu_{1\mid 2}, \Sigma_{1 \mid 2})$$，可以计算如下：

$$
\begin{align}
\mu_{1 \mid 2} &= \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2) \\
\Sigma_{1 \mid 2} &= \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}
\end{align}
$$

## 因子分析模型
在因子分析模型中，我们定义(x,z)的联合分布如下，其中$$z \in \mathbb{R}^k $$是一个潜在随机变量：

$$
\begin{align}
z & \sim \mathcal{N}(0, I) \\
x \mid z & \sim \mathcal{N} (\mu + \Lambda z, \Psi)
\end{align}
$$

其中向量$$\mu \in \mathcal{R}^n $$，矩阵 $$ \Lambda \in \mathcal{R}^{n \times k} $$， 对角阵$$\Psi \in \mathcal{R}^{n \times n}$$，k的取值一般都要小于n。

我们相当于把数据从k维映射到n维$$\mu+ \Lambda z$$，最后再填上一个噪音$$\Psi$$。上面的因子分析模型也可以表示成下面的形式：

$$
\begin{align}
z & \sim \mathcal{N} (0, I) \\
\epsilon & \sim \mathcal{N} (0, \Psi) \\
x & = \mu + \Lambda z + \epsilon
\end{align}
$$

我们的随机变量z，x构成了一个联合高斯分布：

$$
\begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N} (\mu_{zx}, \Sigma)
$$

下面来找到$$\mu_{zx}$$和$$\Sigma$$。

容易直到$$E[z]=0$$，因为z满足标准正态分布。$$E[x]=\mu$$可有下式求解：

$$
\begin{align}
E[x] &= E[\mu+ \Psi z + \epsilon] \\
&= \mu + \Psi E[z] + E[\epsilon] \\
&= \mu
\end{align}
$$

因此可以得到$$\mu_{zx}$$:

$$\mu_{zx} =
\begin{bmatrix}
\overrightarrow{0} \\
\mu
\end{bmatrix}
$$

下面继续计算$$\Sigma$$，因为$$\Sigma_{zz} = Cov(z) = I$$，

$$
\begin{align}
\Sigma_{zx} &= E[(z - E[z])(x - E[x])^T] \\
&= E[z (\mu + \Lambda z + \epsilon - \mu)^T ] \\
&= E[zz^T] \Lambda^T + E [z \epsilon^T] \\
&= \Lambda^T
\end{align}
$$

$$
\begin{align}
\Sigma_{xx} &= E[(x - E[x])(x - E[x])^T] \\
&= E[(\mu + \Lambda z + \epsilon - \mu)(\mu + \Lambda z + \epsilon - \mu)^T ] \\
&= E[ \Lambda z z^T \Lambda^T + \epsilon z^T \Lambda^T + \Lambda z \epsilon^T + \epsilon \epsilon^T] \\
&= \Lambda E[z z^t] \Lambda^T + E [\epsilon \epsilon^T] \\
\Lambda \Lambda^T + \Psi
\end{align}
$$

最终我们得到联合分布如下：

$$
\begin{bmatrix}
z \\ x
\end{bmatrix}
\sim \mathcal{N} \left (
\begin{bmatrix}
\overrightarrow{0} \\ \mu
\end{bmatrix},
\begin{bmatrix}
I & \Lambda^T \\
\Lambda & \Lambda \Lambda^T + \Psi
\end{bmatrix}
\right)
$$

x的边缘分布是$$x \sim \mathcal{N} (\mu, \Lambda \Lambda^T + \Psi) $$，因此可以得出其最大似然函数的表达式：

$$\ell (\mu, \Lambda, \Psi) = \log \prod_{i=1}^m \frac {1} {2\pi)^{n/2}
{| \Lambda\Lambda^T + \Psi |}^{1/2} } \exp \left( - \frac{1}{2}
(x^{(i)}-\mu)^T (\Lambda \Lambda^T + \Psi)^{-1} (x^{(i)} - \mu) \right) $$

对于上述的最大似然函数估计，同样的无法直接求解，需要用最大期望算法来解决。

## 因子分析的最大期望算法

在E步，我们需要计算$$Q_i(z^{(i)}) = p(z^{(i)} \mid x^{(i)}; \mu, \Lambda, \Psi)。根据前面条件分布的公式，我们知道$$ z^{(i)} \mid x^{(i)}; \mu, \Lambda, \Psi \sim \mathcal{N} (\mu_{z^{(i)} \mid x^{(i)}}, \Sigma_{z^{(i)} \mid x^{(i)}})，其中：

$$
\begin{align}
\mu_{z^{(i)} \mid x^{(i)}} &= \Lambda^T (\Lambda\Lambda^T + \Psi)^{-1} (x^{(i)} - \mu) \\
\Sigma_{z^{(i)} \mid x^{(i)}} &= I - \Lambda^T (\Lambda\Lambda^T + \Psi)^{-1} \Lambda
\end{align}
$$

后面的推导偷懒不写了~一句话就是用EM算法去求解，感觉更多的是考验数学水平。

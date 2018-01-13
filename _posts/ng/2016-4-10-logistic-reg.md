---
layout: single
title: "逻辑回归"
subtitle: "斯坦福大学机器学习第二讲"
date: 2016-4-10
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
  - 机器学习算法
---

线性回归是一种连续型模型，对于y值是离散的情况就无能为力了。对于分类问题而言，很常用的是另一种算法：逻辑回归。逻辑回归虽然是名为回归，实际却是解决分类问题的，这也是有趣的地方。我们这里仅限讨论二元分类，对于多元分类，原理一致。

## 分类与逻辑回归
在逻辑回归中，由于y取值只有0和1两种可能性，很自然的一个做法是先把X做一个映射，映射到（0，1）空间，而logistic函数很好的满足了这个性质：两侧很快的趋于边界。（注意logistic也不是唯一的选择）逻辑回归的模型就是在线性回归外又做了一个logistic计算：

$$
h(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$

对于离散情况我们不能再用损失函数去衡量模型的拟合度，只能从最大似然性角度去衡量。逻辑回归和线性回归的似然性表示也有很大区别，线性回归通过引入一个误差的高斯分布来表示，逻辑回归则可以直接用$$h_\theta(x)$$来表示：

$$
\begin{align}
&P(y=1\mid x;\theta)=h_\theta(x)\\
&P(y=0\mid x;\theta)=1-h_\theta(x)
\end{align}
$$

把两式归纳起来可以写成：

$$
p(y\mid x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$

所以有最大似然函数：

$$
\begin{align}
L(\theta)&=\prod_{i=1}^m p(y^{(i)}\mid x^{(i)};\theta) \\
&= \prod_{i=1}^m(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
\end{align}
$$

后面的套路都是一样的，两边求对数，用梯度下降法求对数最大似然函数最大值。

对数最大似然函数：

$$
\begin{align}
\ell(\theta)&=\log L(\theta)\\
&=\sum_{i=1}^m y^{(i)}\log h(x^{(i)})+(1-y^{(i)})\log (1-h(x^{(i)}))
\end{align}
$$

对于单个训练集来说，对其对数最大似然函数求导：

$$
\begin{align}
\frac{\partial}{\partial\theta_j}&=(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)})\frac{\partial}{\partial\theta_j}g(\theta^Tx) \\
&=(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)})g(\theta^Tx)(1-g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^Tx\\
&=(y(1-g(\theta^Tx))-(1-y)g(\theta^Tx))x_j\\
&=(y-h_\theta(x))x_j
\end{align}
$$

利用增量梯度下降法：

$$
\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$

这个式子和线性回归竟然一模一样！当然这里的$$h_\theta(x)$$是不同的，其实后面可以证明，这可不仅仅是巧合哦。

## 题外话：感知机
感知机是个历史悠久的模型，刚开始用来模型神经元的原理。后来被用在机器学习中，虽然简单，但却非常有效。至今仍是神经网络算法的基石。

感知机和逻辑回归不同的地方在于把logistic函数用域函数替代了：

$$
g(z)=
\begin{cases}
1 \quad\text{if }z\geq0 \\
0 \quad\text{if }z<0
\end{cases}
$$

仍然使用上面的梯度下降法进行迭代。

乍看之下感知机和逻辑回归非常相像，其实有很多的区别。感知机模型的输出也是离散量，而逻辑回归的输出是连续量。特别的，感知机没法用概率去解释，黑白分明，自然也没法用最大似然函数去衡量拟合情况。

## 牛顿法
牛顿法是另一种可以求解对数似然函数最大值得方法。注意牛顿法也必须应用在凸函数上。想要求得$$f(\theta)=0$$，可以用下式迭代：

$$
\theta:=\theta-\frac{f(\theta)}{f'(\theta)}
$$

使用在求最大值上，即$$f'(\theta)=0$$，则使用下式迭代：

$$
\theta:=\theta-\frac{\ell'(\theta)}{\ell' '(\theta)}
$$

对于向量化的$$\theta$$，比较复杂，需要使用到海森矩阵（一个1+n,1+n的矩阵）：

$$
\theta:=\theta-H^{-1}\nabla_{\theta}\ell(\theta)
$$

海森矩阵（一个1+n,1+n的矩阵）的计算方法如下，还是另篇专述：

$$
H_{ij}=\frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}
$$

牛顿方法通常比梯度下降法更快的收敛，这个可以从他们的迭代步子中也可以粗略的感知。但和直接计算法类似，牛顿法也使用了矩阵的逆，当特征数很大时，这一步的计算会很费时。所以，天下没有免费的午餐。

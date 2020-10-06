---
layout: single
title: "主成分分析"
subtitle: "斯坦福大学机器学习第十一讲"
date: 2016-5-22
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
  - 机器学习理论
---

前一讲因子分析中，将数据投射到k维子空间，它是一个可能性模型，使用EM算法拟合参数。本讲中的主成分分析完成类似的功能，但却简单的多，只需要用到特征向量，不需要借用EM算法。

在实现PCA算法之前，需要先对数据进行预处理：

1. 令$$\mu = \frac{1}{m} \sum_{i=1}^m x^{(i)}$$
2. 将$$x^{(i)}$$ 用$$x^{(i)} - \mu$$代替
3. 计算$$\sigma_j^2 = \frac{1}{m} \Sigma_i(x_j^{(i)})^2$$
4. 将$$x_j^{(i)}$$用$$x_j^{(i)}/\sigma_j$$

如果知道两个特征的范围大小一致，后两步可以省略。

现在我们就是要找出一个方向，使得投射后的数据仍然有高的方差。即最大化下式：

$$
\begin{align}
\frac{1}{m} \sum_{i=1}^m ({x^{(i)}}^T u)^2 &= \frac{1}{m} \sum_{i=1}^m u^T x^{(i)} {x^{(i)}}^T u \\
&= u^T \left( \frac{1}{m} \sum_{i=1}^m x^{(i)} {x^{(i)}}^T \right) u
\end{align}
$$

当$$\|u\|_2 = 1$$时，很容易得知u取经验协方差矩阵$$\Sigma = \frac{1}{m} \sum_{i=1}^m x^{(i)} {x^{(i)}}^T$$的特征向量时为最大值。

当想得到一维子空间时，u取首要特征向量。更一般化，要投射到k维子空间时，取头k个特征向量：$$u_1, \cdots, u_k$$。

子空间的特征可以用对应的向量表示：

$$ y^{(i)} =
\begin{bmatrix}
u_1^T x^{(i)} \\
u_2^T x^{(i)} \\
\vdots \\
u_k^T x^{(i)} \\
\end{bmatrix}
\in
{\mathbb{R}}^k
$$

PCA也被称为降维算法。向量$$u_1, \cdots, u_k$$被称为k个主成分。

PCA算法有很多应用场景。

* 将数据压缩到二或三维可以用作可视化
* 在监督学习前使用PCA来降低数据复杂度，避免过拟合。
* PCA算法可以用来降噪。

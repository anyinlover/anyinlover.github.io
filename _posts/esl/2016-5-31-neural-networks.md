---
layout: post
title: "神经网络"
subtitle: "ESL第11章"
date: 2016-5-31
author: "Anyinlover"
catalog: true
tags:
  - ESL
  - 机器学习算法
---

## 简介
在这一章我们描述了一类学习方法，它们分别在统计和人工智能领域独立发展过，但模型却是一致的。其中的核心思想就是将输入的线性组合作为特征，将目标建模为这些特征的非线性函数。这种方法在很多领域广泛应用。我们首先介绍投影寻踪模型，然后再介绍神经网络模型。

## 投影寻踪回归
假设我们有一个输入向量X长度为p，和一个目标值Y。令$$\omega_m, m=1,2,\cdots,M$$是一组长度为p的向量构成的未知参数，则投影寻踪回归模型具有以下的形式：

$$f(X) = \sum_{m=1}^M g_m(\omega_m^T X)$$

这是一个加性模型，但特征是$$V_m = \omega_m^T X$$而不是输入本身。函数$$g_m$$是非指定的。

函数$$g_m(\omega_m^T X)$$被称为$$\mathbb{R}^p$$的桥函数。其只在向量$$\omega_m$$方向变化。$$V_m = \omega_m^T X$$相当于$$X$$在单位向量$$\omega_m$$方向上的投射。$$\omega_m$$是需要模型拟合的参数。

PPR模型主要用于分类，对于产生解释性模型比较无力，只有$$M=1$$是个例外，相当于线性回归的更一般化。

给定训练样本$$(x_i,y_i), i=1,2,\cdots,N$$，如何来拟合PPR模型？令损失函数为：

$$\sum_{i=1}^N [y_i - \sum_{m=1}^M g_m (\omega_m^T x_i)]^2$$

和其他平滑问题一样，需要为$$g_m$$添加复杂的限制来避免过拟合。

考虑$$M=1$$的情况，给定方向向量$$\omega$$，我们得到$$v_i = \omega^T x_i$$，然后通过散点平滑方法来得到g的估计。

另一方面，给定g后，需要最小化$$\omega$$。高斯牛顿搜索可以解决这个问题，通过将二次导舍弃，我们可以近似得到：

$$g(\omega^T x_i) \approx g(\omega_{old}^T x_i) + g' (\omega_{old}^T x_i)
(\omega - \omega_{old})^T x_i$$

因此可以近似得到：

$$\sum_{i=1}^N [y_i - \sum_{m=1}^M g (\omega^T x_i)]^2 \approx \sum_{i=1}^N g' (\omega_{old}^T x_i)^2 [(\omega_{old}^T + \frac{y_i - g (\omega_{old}^T x_i)}{g' (\omega_{old}^T x_i)}) - \omega^T x_i]^2$$

通过对于上面两步的交替迭代，直到拟合。

## 神经网络
神经网络被传得神神道道，其实只不过是一个非线性统计模型。

一个典型的神经网络是一个两层的回归或分类模型。作为回归模型时上层只有一个$$Y_1$$。对于k类分类问题，上层有K个单元，第k个单元代表第k类的可能性，用$$Y_k, k=1,\cdots,K$$来衡量，每一个都是0-1之间的取值。

提取的特征$$Z_m$$通过对输入进行线性组合构造成，目标值$$Y_k$$则是$$Z_m$$的函数：

$$
\begin{align}
Z_m &= \sigma(\alpha_{0m} + \alpha_m^T X), m = 1,\cdots,M, \\
T_k &= \beta_{0k} + \beta_k^T Z, k=1,\cdots,K, \\
f_k(X) &= g_k(T), k=1,\cdots,K
\end{align}
$$

激励函数$$\sigma(v)$$常常选择S型函数$$\sigma(v)=1/(1+e^{-v})$$，有时候高斯径向基函数也被用作$$\sigma(v)$$，这时神经网络被称为径向基网络。

从上面的公式中可以看出，对每个隐藏层和输出层，都添加了额外的偏差$$\alpha_{0m}, \beta_{0k}$$。

输出函数$$g_k(T)$$允许对输出T做最后的转换。对于回归，一般都选择本身$$g_k(T) = T_k$$，对于K类分类，则选择softmax函数：

$$g_k(T) = \frac {e^{T_k}}} {\sum_{l=1}^K e^{T_l}}$$

如果把$$Z_m$$看做原始输入X的基本扩展，那么神经网络就是一个标准的线性模型。

如果$$\sigma$$也是个恒等函数，那模型也可以看做针对输入的线性模型。因此神经网络可以被视为对线性模型的非线性泛化，这个结论对回归和分类都使用。

注意到一层隐含层的神经网络模型和上一节中的投影寻踪模型具有一致的形式。

$$
\begin{align}
g_m(\omega_m^T X) &= \beta_m \sigma(\alpha_{0m} + \alpha_m^T X) \\
&= \beta_m \sigma(\alpha_{0m} + \| \alpha_m \| (\omega_m^T X))
\end{align}
$$

## 神经网络拟合
神经网络的未知参数，常常被称为权重。

$$
\begin{align}
\{\alpha_{0m}, \alpha_m; m =1,2,\cdots,M\} \quad M(p+1) weights, &
\{\beta_{0k}, \beta_k; k =1,2,\cdots,K\}  \quad K(M+1) weights
\end{align}
$$

对于回归来说，使用平方和误差作为误差函数：

$$R(\theta) = \sum_{k=1}^K sum_{i=1}^N (y_{ik} - f_k(x_i))^2 $$

对于分类而言，使用平方和误差或者互熵：

$$R(\theta) = - \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log f_k(x_i)$$

对应的分类就是 $$G(x) = argmax_k f_k(x)$$。使用softmax激励函数和互熵误差函数，神经网络在隐藏层就是一个逻辑回归模型，所有参数可以通过最大似然性得到。

标准的最小化$$R(\theta)$$做法是通过梯度递减，在这里被称为反向传播算法。

令$$z_{mi} = \sigma (\alpha_{0m} + \alpha_m^T x_i)$$，令$$z_i = (z_{1i}, z_{2i}, \cdots, z_{Mi})$$。对于平方和误差而言，由于：

$$R(\theta) = \sum_{i=1}^N R_i = \sum_{i=1}^N \sum_{k=1}^K (y_{ik} - f_k(x_i))^2$$

其导数为：

$$
\begin{align}
\frac{\partial R_i}{\partial \beta_{km}} &= -2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i) z_{mi} \\
\frac{\partial R_i}{\partial \alpha_{ml}} &= -\sum_{k=1}^K 2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i) \beta_{km} \sigma' (\alpha_m^T x_i) x_{il}
\end{align}
$$

得到导数后，根据梯度下降算法：

$$
\begin{align}
\beta_{km}^{(r+1)} &= \beta_{km}^{(r)} - \gamma_r \sum_{i=1}^N \frac {\partial R_i} {\partial \beta_{km}^{(r)}} \\
\alpha_{ml}^{(r+1)} &= \alpha_{ml}^{(r)} - \gamma_r \sum_{i=1}^N \frac {\partial R_i} {\partial \alpha_{ml}^{(r)}}
\end{align}
$$

其中$$\gamma_r$$是学习率。

将导数记为：

$$
\begin{align}
\frac{\partial R_i}{\partial \beta_{km}} &= \delta_{ki} z_{mi} \\
\frac{\partial R_i}{\partial \alpha_{ml}} &= s_{mi} x_{il}
\end{align}
$$

其中$$\delta_{ki},s_{mi}$$是模型在输出层和隐藏层的误差。根据定义，它们满足以下关系：

$$s_{mi} = \sigma' (\alpha_m^T x_i) \sum_{k=1}^K \beta_{km} \delta_{ki}$$

上面这个式子被称为反向传播等式。利用这个，参数更新可以通过两步走。正向走时，当前权重固定，预测值$$\hat{f}_k (x_i)$$计算得到，反向走时，误差$$\delta_{ki}$$计算得到，进一步得到$$s_{mi}$$。然后再去更新参数。这个算法也被称为反向传播算法。

反向传播算法的优点在于其简单和本地属性。但它的速度比较慢，牛顿方法同样不适用（R的海森矩阵计算量大）。更好的拟合方法是共轭梯度和变度算法。

## 神经网络训练的讨论

### 初始值

### 过拟合

### 输入范围

### 隐藏单元和层的数目

### 多极小值

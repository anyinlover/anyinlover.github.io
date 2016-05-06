---
layout: post
title: "优化算法线性不变习题"
subtitle: "斯坦福大学机器学习习题集一之五"
date: 2016-5-6
author: "Anyinlover"
catalog: true
tags:
  - Ng机器学习系列
---

## 证明牛顿法是线性不变的

令$$g(z)=f(Az)$$，需要找到$$\nabla_z g(z)$$和$$\nabla_z^2 g(z)$$的$$f(z)$$表示。

$$
\begin{align}
\frac {\partial g(z)} {\partial z_i} &= 
\sum_{k=1}^n \frac{\partial f(Az)} {\partial (Az)_k}
\frac{\partial (Az)_k} {partial z_i} \\
&= \sum_{k=1}^n \frac{\partial f(Az)} {\partial (Az)_k} A_{ki} \\
&= \sum_{k=1}^n \frac{\partial f(Az)} {\partial x_k} A_{ki}
\end{align}
$$

上式等同于：

$$
\frac {\partial g(z)} {\partial z_i} =
A_{\cdot i}^T \nabla_x f(Az)
$$

其中$$A_{\cdot i}$$是A的第i列。因此有：

$$
\nabla_z g(z) = A^T \nabla_x f(Az)
$$

再来定义海森矩阵$$\nabla_z^2 g(z)$$：

$$
\begin{align}
\frac {\partial^2 g(z)} {\partial z_i \partial z_j}
&= \frac {\partial} {\partial z_j} \sum_{k=1}^n \frac{\partial f(Az)} {\partial (Az)_k} A_{ki} \\
&= \sum_l \sum_k \frac{\partial^2 f(Az)} {\partial x_l \partial x_k} A_{kj} A_{lj}
\end{align}
$$

因此有:

$$H_g(z) = A^T H_f(Az) A$$

下面来推导对于函数$$f(Ax)$$的牛顿方法：

$$
\begin{align}
z^{(i+1)} &= z^{(i)} - H_g(z^{(i)})^{-1} \nabla_z g(z^{(i)}) \\
&= z^{(i)} - (A^T H_f(Az^{(i)}) A)^{-1} A^T \nabla_x f(Az^{(i)}) \\
&= z^{(i)} - A^{-1}H_f(Az^{(i)})^{-1}(A^T)^{-1}A^T \nabla_x f(Az^{(i)}) \\
&= z^{(i)} - A^{-1}H_f(Az^{(i)})^{-1}\nabla_x f(Az^{(i)})
\end{align}
$$

只需要证明$$Az^{(i+1)}=x^{(i+1)}$$，即完成证明：

$$
\begin{align}
Az^{(i+1)} &= A(z^{(i)} - A^{-1}H_f(Az^{(i)})^{-1}\nabla_x f(Az^{(i)})) \\
&= Az^{(i)} - H_f(Az^{(i)})^{-1}\nabla_x f(Az^{(i)}) \\
&= x^{(i)} - H_f(x^{(i)})^{-1} \nabla_x f(x^{(i)}) \\
&= x^{(i+1)}
\end{align}
$$

## 证明梯度下降法不是线性不变
在$$g(z)$$上应用梯度下降规则：

$$
z^{(i+1)} = z^{(i)} - \alpha A^T \nabla_x f(Az^{(i)})
$$

在$$f(x)%%上应用梯度下降规则：

$$
x^{(i+1)} = x^{(i)} - \alpha \nabla_x f(x^{(i)})
$$

要使得$$x^{(i+1)} = A z^{(i+1)}$$ 成立，

$$
\begin{align}
Az^{(i+1)} &= z^{(i)} - \alpha AA^T \nabla_x f(Az^{(i)}) \\
&= x^{(i)} - \alpha AA^T \nabla_x f(x^{(i)})
\end{align}
$$

必须在$$AA^T=I$$成立的条件下才能成立，因此梯度下降法不是线性不变的。
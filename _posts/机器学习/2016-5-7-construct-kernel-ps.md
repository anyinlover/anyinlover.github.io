---
layout: single
title: "核函数构造习题"
subtitle: "斯坦福大学机器学习习题集二之一"
date: 2016-5-7
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
---

## 相加

是核函数，两个半正定矩阵相加仍然是半正定。

$$
\begin{align}
& \forall z \, z^TG_1z \geq 0, z^TG_2z \geq 0 \\
\implies & \forall z \, z^T G z = z^TG_1z + z^TG_2z \geq 0
\end{align}
$$

## 相减

不是核函数，令$$K_2 = 2K_1$$，则：

$$ \forall z \, z^T G z = z^T (G_1 - 2G_1) z = - z^T G_1 z \leq 0$$

## 正系数

是核函数

$$
\begin{align}
& \forall z \, z^TG_1z \geq 0 \\
\implies & \forall z \, z^T G z = az^TG_1z  \geq 0
\end{align}
$$

## 负系数

不是核函数

$$
\begin{align}
& \forall z \, z^TG_1z \geq 0 \\
\implies & \forall z \, z^T G z = -az^TG_1z  \leq 0
\end{align}
$$

## 相乘

是核函数，由于$$K_1, K_2$$是核函数：

$$
\begin{align}
& \exists \phi^{(1)} \, K_1(x,z) = \phi^{(1)}(x)^T\phi^{(1)}(z)=\sum_i \phi_i^{(1)}(x)\phi_i^{(1)}(z) \\
& \exists \phi^{(1)} \, K_2(x,z) = \phi^{(2)}(x)^T\phi^{(2)}(z)=\sum_i \phi_i^{(2)}(x)\phi_i^{(2)}(z)
\end{align}
$$

因此可以推导得到：

$$
\begin{align}
K(x,z) &= K_1(x,z)K_2(x,z) \\
&= \sum_i \phi_i^{(1)}(x)\phi_i^{(1)}(z)\sum_i \phi_i^{(2)}(x)\phi_i^{(2)}(z) \\
&= \sum_i \sum_j \phi_i^{(1)}(x)\phi_i^{(1)}(z) \phi_j^{(2)}(x)\phi_j^{(2)}(z) \\
&= \sum_i \sum_j (\phi_i^{(1)}(x)\phi_j^{(2)}(x))(\phi_i^{(1)}(z)\phi_j^{(2)}(z)) \\
&= \sum_{(i,j)} \psi_{i,j}(x)\psi_{i,j}(z) \\
&= \psi(x)^T \psi(z)
\end{align}
$$

## 函数相乘

是核函数。上一种情况的特殊化，令$$\psi(x) = f(x)$$。

## 映射核函数

是核函数，仍然保持半正定。

## 多项式

是核函数，通过上面的证明，相加，系数，幂，截距运算都保持核函数性质。

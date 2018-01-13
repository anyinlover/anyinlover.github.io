---
layout: single
title: "局部加权回归习题"
subtitle: "斯坦福大学机器学习习题集一之二"
date: 2016-5-3
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
---

## 加权系数的矩阵表示
令$$W_{ii}=\frac{1}{2} w^{(i)}，W_{ij}=0 \text{ for } i \neq j, \overrightarrow{z}=X\theta-\overrightarrow{y}, z_i=\theta^T x^{(i)}-y^{(i)}$$，因此可以推导出下式：

$$
\begin{align}
J(\theta) &= (X\theta-\overrightarrow{y})^TW(X\theta-\overrightarrow{y}) \\
&=\overrightarrow{z}^TW \overrightarrow{z} \\
&=\frac{1}{2} \sum_{i=1}^m w^{(i)} z_i^2 \\
&= \frac{1}{2} \sum_{i=1}^m w^{(i)} (\theta^T x^{(i)}-y^{(i)})^2
\end{align}
$$

## 局部加权的标准方程
根据上式得到的损失函数，可以计算如下：

$$
\begin{align}
\nabla_{\theta}J(\theta)&=\nabla_{\theta}\frac{1}2(X\theta-\overrightarrow{y})^T W (X\theta-\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}(\theta^T X^T W X\theta-\theta^TX^T W \overrightarrow{y}-\overrightarrow{y}^T W X\theta+\overrightarrow{y}^T W \overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}tr(\theta^TX^T W X\theta-\theta^TX^T W \overrightarrow{y}-\overrightarrow{y}^T W X\theta+\overrightarrow{y}^T W \overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}(tr\theta^TX^T W X\theta-2tr\overrightarrow{y}^T W X\theta)\\
&=\frac{1}2(X^T W X\theta+X^TWX\theta-2X^TW^T\overrightarrow{y}）\\
&=X^TWX\theta-X^TW\overrightarrow{y}
\end{align}
$$

令上式为0，最终我们得出局部加权的标准方程为：

$$
\begin{align}
X^TWX\theta&=X^TW\overrightarrow{y}\\
\theta &= (X^TWX)^{-1}X^TW\overrightarrow{y}
\end{align}
$$

## 以y的方差表示加权系数

$$
\begin{align}
\ell(\theta)&=\log{L(\theta)}\\
&=\log\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma^{(i)}}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2(\sigma^{(i)})^2})\\
&=\sum_{i=1}^m \log\frac{1}{\sqrt{2\pi}\sigma^{(i)}}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2(\sigma^{(i)})^2})\\
&=\sum_{i=1}^m\log\frac{1}{\sqrt{2\pi}\sigma^{(i)}}-\frac{1}{2}\sum_{i=1}^{m}\frac{1}{(\sigma^{(i)})^2} (h_\theta(x^{(i)})-y^{(i)})^2 \\
&=\sum_{i=1}^m\log\frac{1}{\sqrt{2\pi}\sigma^{(i)}} - \frac{1}{2}\sum_{i=1}^{m} w^{(i)} (h_\theta(x^{(i)})-y^{(i)})^2
\end{align}
$$

即：

$$w^{(i)} = \frac{1}{(\sigma^{(i)})^2} $$

## 回归问题

### 实现一般回归
用标准方程来写一般回归是比较容易的。

~~~
from numpy import *
import pandas as pd
orx = pd.read_csv('q2x.dat', sep = '\s+', header=None).values
y = pd.read_csv('q2y.dat', sep = '\s+', header=None).values.ravel()
X = hstack((ones((orx.shape[0],1)), orx))
theta = dot(dot(linalg.inv(dot(X.T,X)), X.T),y)
print(theta)

import matplotlib.pyplot as plt
plt.scatter(x = orx.ravel(), y = y, marker='x', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10,15)
xl = arange(-10,16,1)
yl = theta[0]+theta[1]*xl
plt.plot(xl, yl)
~~~

![ps1_2_1](\img\ps1_2_1.png)

### 实现局部加权线性回归

~~~
Xl = arange(-10, 15.01, 0.02)
def plot_t(t):
    plt.hold('on')
    Xl = arange(-10, 15.01, 0.02)
    Ylo = []
    for xl in Xl:
        w = exp(-(orx.ravel()-xl)**2/(2*t**2))
        W = diag(w)
        theta = dot(dot(dot(linalg.pinv(dot(dot(X.T,W),X)),X.T),W),y)
        yl = dot(theta.T,array([1,xl]))
        Ylo.append(yl)
        Yl = array(Ylo)
    plt.plot(Xl,Yl,color='red')

 plt.scatter(x = orx.ravel(), y = y, marker='x', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10,15)
plot_t(0.8)
plot_t(0.3)
plot_t(2)
~~~

最终的图形如下（这图还可以优化一下），局部线性回归可以较好的拟合散点。

![ps1_2_1](\img\ps1_2_2.png)

### 带宽的影响
带宽决定了影响模型的样本数量多少。带宽越小，影响模型的样本越少，模型也更容易受噪音影响，上面当带宽变0.1时，直接导致矩阵不可逆。带宽越大，影响模型的样本越多，模型更趋于一般线性回归。

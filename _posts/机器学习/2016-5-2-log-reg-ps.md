---
layout: single
title: "逻辑回归习题"
subtitle: "斯坦福大学机器学习习题集一之一"
date: 2016-5-2
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
---

## 逻辑回归

### 证明逻辑回归的对数最大似然函数的海森矩阵是半负定矩阵

对数最大似然函数表示如下：

$$
\ell(\theta) = \sum_{i=1}^m y^{(i)} \log h(x^{(i)})
+ (1 - y^{(i)}) \log (1 - h(x^{(i)}))
$$

其一阶导根据讲义的证明是：

$$
\frac {\partial \ell(\theta)} {\partial \theta_k} =
\sum_{i=1}^m (y - h_\theta (x^{(i)}))x_k^{(i)}
$$

海森矩阵单个元素可表示为：

$$
\begin{align}
H_{kl} &= \frac {\partial \ell(\theta)} {\partial \theta_k \theta_l} \\
&= - \sum_{i=1}^m \frac {\partial h_\theta (x^{(i)})} {\partial \theta_l} x_k^{(i)} \\
&= - \sum_{i=1}^m h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) x_l^{(i)}x_k^{(i)}
\end{align}
$$

海森矩阵可表示为：

$$H = - \sum_{i=1}^m h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) x^{(i)} x^{(i)T}$$

下面来证明$$z^T H z \leq 0$$ 恒成立。

$$
\begin{align}
z^T H z &= - z^T \sum_{i=1}^m h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) x^{(i)} x^{(i)T} z \\
&= - \sum_{i=1}^m h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) z^T x^{(i)} x^{(i)T} z \\
&= - \sum_{i=1}^m h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) (z^T x^{(i)})^2 \\
& \leq 0
\end{align}
$$

因此逻辑回归的对数最大似然函数的海森矩阵是一个半负定矩阵，其只有一个唯一的全局最大值。

### 用牛顿法来拟合逻辑回归模型

尽管牛顿法的表达式已经给出，但用 python 跑出这个程序还是花了我三个小时。主要的难点在于把前面的代数表示转化为矩阵表示，包括梯度和海森矩阵。下面给出我的 python 代码，用的是 python3。

```python
from numpy import *
import pandas as pd

orx = pd.read_csv('q1x.dat', sep='\s+', header=None).values
y = pd.read_csv('q1y.dat', sep='\s+', header=None).values.ravel()

X = hstack((ones((orx.shape[0], 1)), orx))
theta = zeros(X.shape[1])


def h(theta):
    return 1/(1+exp(-dot(X, theta)))


def hd(theta):
    return dot(X.T, y - h(theta))


def hdd(theta):
    return -dot(X.T, tile(h(theta)*(1-h(theta)), (theta.size, 1)).T*X)


maxtry = 50

for i in range(maxtry):
    theta = theta - dot(linalg.inv(hdd(theta)), hd(theta))

print(theta)
```

### 画图

这个纯粹的就是画图能力的考验，这一块前面看过《python machine learning》，还是比较容易的。

```python
import matplotlib.pyplot as plt
plt.scatter(x = orx[y==1,0], y = orx[y==1,1], marker='o', color='red', label='y=1')
plt.scatter(x = orx[y==0,0], y = orx[y==0,1], marker='x', color='blue', label='y=0')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='upper left')
plt.xlim(0,9)
x1 = arange(0,10,1)
x2 = (-theta[0]-theta[1]*x1)/theta[2]
plt.plot(x1, x2)
```

最后得到的图：

![ps1_1](\img\ps1_1.png)

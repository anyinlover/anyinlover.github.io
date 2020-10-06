---
layout: single
title: "训练测试异分布习题"
subtitle: "斯坦福大学机器学习习题集二之五"
date: 2016-5-16
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
---

## 误差表示

我们首先用$$\varepsilon_0$$来表示$$\varepsilon_\tau$$，然后再反过来表示。

误差的出现只有两种可能，第一种是原始分布有误差，点没误差。第二种是原始分布无误差，点有误差，因此误差$$\varepsilon_\tau$$可以被表示成下式：

$$\varepsilon_\tau = \varepsilon_0(1-\tau) + (1-\varepsilon_0)\tau$$

解得$$\varepsilon_0$$等于：

$$ \varepsilon_0 = \frac{\varepsilon_\tau - \tau}{1 - 2\tau}$$

## 最优表示

利用下面的三个条件，我们可以进行推导：

$$ \forall h \in H, | \varepsilon_\tau(h) - \hat{\varepsilon}_\tau(h)| \leq \bar{\gamma} \quad w.p.(1-\delta), \quad \delta=2K \exp(-2\bar{\gamma}^2m)$$

$$\varepsilon_\tau=(1-2\tau)\varepsilon+\tau, \quad \varepsilon_0 = \frac{\varepsilon_\tau - \tau}{1-2\tau}$$

$$\forall h \in H, \hat{\varepsilon}_\tau(\hat{h}) \leq \hat{\varepsilon}_\tau(h), \quad \text{in particular for }h^*$$

可以得到：

$$
\begin{align}
\varepsilon_0(\hat{h}) &= \frac {\varepsilon_\tau(\hat{h})-\tau}{1-2\tau} \\
&\leq \frac {\hat{\varepsilon}_\tau(\hat{h})+\bar{\gamma}-\tau}{1-2\tau}\quad w.p.(1-\delta) \\
&\leq \frac {\hat{\varepsilon}_\tau(h^*)+\bar{\gamma}-\tau}{1-2\tau}\quad w.p.(1-\delta) \\
&\leq \frac {\varepsilon_\tau(h^*)+2\bar{\gamma}-\tau}{1-2\tau}\quad w.p.(1-\delta) \\
&=\frac {(1-2\tau)\varepsilon_0(h^*)+\tau+2\bar{\gamma}-\tau}{1-2\tau}\quad w.p.(1-\delta) \\
&= \varepsilon_0(h^*) + \frac {2\bar{\gamma}}{1-2\tau}\quad w.p.(1-\delta) \\
&= \varepsilon_0(h^*) + 2\gamma\quad w.p.(1-\delta)
\end{align}
$$

最后一步令$$\bar{\gamma}=\gamma(1-2\tau)$$，再代回第一个条件，就得到

$$m \geq \frac{1}{2(1-2\tau)^2\gamma^2} \log \frac{2|H|}{\delta}$$

这个式子与同分布相比多了一个分母$$(1-2\tau)^2$$。意味着分布误差越大，所需的训练样本数越多。

## 讨论

$$\tau$$越接近0.5，得到相同生成误差边界所需的样本个数就越多。当$$\tau$$接近0.5时，训练数据越来越趋于随机，当$$\tau=0.5$$时就没有有用的信息了。

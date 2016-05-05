---
layout: post
title: "高斯判别分析习题"
subtitle: "斯坦福大学机器学习习题集一"
date: 2016-5-3
author: "Anyinlover"
catalog: true
tags:
  - Ng机器学习系列
---

## 高斯判别分析与逻辑回归的关系

$$
\begin{align}
p(y=1 \mid x; \phi, \Sigma, \mu_0, \mu_1) &= \frac {p(x \mid y=1; \phi, \Sigma, \mu_0, \mu_1) p(y=1; \phi, \Sigma, \mu_0, \mu_1)} {p(x; \phi, \Sigma, \mu_0, \mu_1)} \\
&= \frac {p(x \mid y=1; \phi, \Sigma, \mu_0, \mu_1) p(y=1; \phi, \Sigma, \mu_0, \mu_1)} {p(x \mid y=1; \phi, \Sigma, \mu_0, \mu_1) p(y=1; \phi, \Sigma, \mu_0, \mu_1) + p(x \mid y=0; \phi, \Sigma, \mu_0, \mu_1) p(y=0; \phi, \Sigma, \mu_0, \mu_1)} \\
&= \frac { \exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)\phi} { \exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)\phi + \exp\left(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)\right)(1-\phi)} \\
&=\frac {1} {1+ \exp\left(\log( \frac {1-\phi} {\phi})-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)+\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)} \\
&= \frac{1} {1 + \exp \left(-\frac{1}{2} (-2\mu_0^T\Sigma^{-1}x + \mu_0^T\Sigma^{-1}\mu_0 + 2\mu_1^T\Sigma^{-1}x - \mu_1^T\Sigma^{-1}\mu_1) + \log ( \frac {1-\phi} {\phi})\right)}
\end{align}
$$

因此可以得到：

$$
\theta =
\begin{bmatrix}
\frac{1}{2}(\mu_0^T\Sigma^{-1}\mu_0-\mu_1^T\Sigma^{-1}\mu_1) - \log( \frac {1-\phi} {\phi}) \\
\Sigma^{-1}\mu_1 - \Sigma^{-1}\mu_0
\end{bmatrix}
$$

## 模型参数推导

$$
\begin{align}
\ell(\phi,\mu_0,\mu_1,\Sigma) &= \log \prod_{i=1}^m p(x^{(i)} \mid y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi) \\
&= \sum_{i=1}^m \log p(x^{(i)} \mid y^{(i)}; \mu_0, \mu_1, \Sigma) + \sum_{i=1}^m \log p(y^{(i)}; \phi) \\
&\simeq \sum_{i=1}^m [\frac{1}{2} \log \frac{1}{\left| \Sigma \right|} -\frac{1}{2}(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}}) + y^{(i)} \log \phi + (1 - y^{(i)}) \log(1-\phi)]
\end{align}
$$

然后分别对各参数求偏导。

首先求解$$\phi$$

$$
\begin{align}
\frac{\partial \ell} {\partial \phi} &= \sum_{i=1}^m \left[\frac{y^{(i)}}{\phi} - \frac {1-y^{(i)}} {1 - \phi} \right] \\
&= \frac{\sum_{i=1}^m 1\{y^{(i)}=1\}}{\phi} - \frac{m - \sum_{i=1}^m 1\{y^{(i)}=1\}}{1-\phi}
\end{align}
$$

令上式为0，可以得到：

$$
\phi = \frac{1}{m}\sum_{i=1}^m1\{y^{(1)}\}
$$

再求解$$\mu_0$$

$$
\begin{align}
\nabla_{\mu_0} \ell &= -\frac{1}{2} \sum_{i:y^{(i)}=0}\nabla_{\mu_0}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0) \\
&= -\frac{1}{2} \sum_{i:y^{(i)}=0}\nabla_{\mu_0} [\mu_0^T\Sigma^{-1}\mu_0-{x^{(i)}}^T \Sigma^{-1} \mu_0 - \mu_0^T \Sigma^{-1} x^{(i)}] \\
&= -\frac{1}{2} \sum_{i:y^{(i)}=0}\nabla_{\mu_0} tr[\mu_0^T\Sigma^{-1}\mu_0-{x^{(i)}}^T \Sigma^{-1} \mu_0 - \mu_0^T \Sigma^{-1} x^{(i)}] \\
&= -\frac{1}{2} \sum_{i:y^{(i)}=0} [2\Sigma^{-1} \mu_0 - 2\Sigma^{-1} x^{(i)}]
\end{align}
$$

令上式为0，可以得到，$$\mu_1$$同理：

$$
\mu_0 = \frac{\sum_{i=1}^m1\{y^{i}=0\}x^{(i)}}{\sum_{i=1}^m1\{y^{i}=0\}}
$$

最后求解$$\Sigma$$，为方便计算，对$$S = \Sigma^{-1}$$求偏导：

$$
\begin{align}
\nabla_S \ell &= \sum_{i=1}^m \nabla_S [\frac{1}{2} \log \left| S \right| - \frac{1}{2}(x^{(i)}-\mu_{y^{(i)}})^TS(x^{(i)}-\mu_{y^{(i)}})] \\
&= \sum_{i=1}^m [\frac{1}{2\left| S \right|}\nabla_S \left| S \right| - \frac{1}{2}\nabla_S(x^{(i)}-\mu_{y^{(i)}})^TS(x^{(i)}-\mu_{y^{(i)}})] \\
&= \sum_{i=1}^m[\frac{1} {2} S^{-1} - \frac{1} {2} (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T] \\
&= \frac{1}{2} \sum_{i=1}^m[\Sigma - (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T]
\end{align}
$$

令上式为0，可以得到：

$$
\Sigma = \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T
$$
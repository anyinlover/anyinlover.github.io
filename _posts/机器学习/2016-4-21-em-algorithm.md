---
layout: single
title: "最大期望算法"
subtitle: "斯坦福大学机器学习第九讲"
date: 2016-4-21
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - 机器学习算法
  - Ng机器学习系列
---

在前一讲中，我们谈到最大期望算法应用于混合高斯模型中，在这一讲，我们给出最大期望算法的一般形式，展示其如何能运用于求解潜在变量的预测问题。

## 琴生不等式

假如是一个凸函数，即$$f''(x) \geq 0$$，或者$$H \geq 0$$。恒取大于号是称为严格凸函数。X是随机分布，琴生不等式可表达如下：

$$ E|f(X)| \geq f(EX)$$

当等号成立时必须有$$X=E \mid X \mid $$恒成立，即X是常量。

琴生不等式可以用图形直观的去解释。

同理，当f时一个凹函数时，不等式反向成立。

## 最大期望算法

给定训练集$$\{x^{(1)},\cdots,x^{(m)}\}$$由m个独立样本构成。我们需要针对数据拟合模型$$p(x,z)$$的参数，最大似然函数由下式给出：

$$
\begin{align}
\ell(\theta) &= \sum_{i=1}^m \log p(x;\theta) \\
&= \sum_{i=1}^m \log \sum_z p(x,z;\theta)
\end{align}
$$

对上式直接求偏导计算无法得到解析解。因为$$z^{(i)}$$是一个潜在变量，只有$$z^{(i)}$$是已知的条件下，才可能解出，因此引出了最大期望算法。它的策略就是分两步走，E步首先给出$$\ell$$的下限，然后在M步优化这个下限。

对每个i，令$$Q_i$$是对z的分布（$$\sum_z Q_i(z) = 1, Q_i(z) \geq 0$$）运用琴生不等式，有下面的关系：

$$
\begin{align}
\sum_i \log p(x^{(i)}; \theta) &= \sum_i \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta) \\
&= \sum_i \log \sum_{z^{(i)}} Q_i(z^{(i)}) \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})} \\
&\geq \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})}
\end{align}
$$

我们可以看出分布$$[p(x^{(i)}，z^{(i)};\theta)/Q_i(z^{(i)})]$$针对$$z^{(i)}$$关于$$Q_i$$的期望值就是

$$\sum_{z^{(i)}} Q_i(z^{(i)}) \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})}$$

又由于对数函数是一个凹函数，因此有：

$$f\left(E_{z^{(i)} \sim Q_i} \left[ \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})} \right] \right) \geq E_{z^{(i)} \sim Q_i} \left[ f \left( \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})} \right) \right] $$

因此对于任意分布$$Q_i$$，上式给出了对$$\ell(\theta)$$的下限。在选择$$Q_i$$时，一个很自然的做法是l令琴生不等式等号成立。即变量为常量：

$$\frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)}    )} = c$$

其中c是独立于$$z^{(i)}$$的常量。又因为$$\sum_z Q_i(z^{(i)})=1$$，所以有：

$$
\begin{align}
Q_i(z^{(i)}) &= \frac {p(x^{(i)}, z^{(i)}; \theta)} {\sum_z p(x^{(i)}, z^{(i)}; \theta)} \\
&= \frac {p(x^{(i)}, z^{(i)}; \theta)} {p(x^{(i)}; \theta)} \\
&= p(z^{(i)} \mid x^{(i)}; \theta)
\end{align}
$$

也就是$$Q_i$$是在$$x^{(i)}$$给定下$$z^{(i)}$$的后验分布。现在我们可以得出最大期望算法的数学表达：
重复下面的步骤直到收敛：{

E步，对每个i，令：

$$Q_i(z^{(i)}) := p(z^{(i)} \mid x^{(i)}; \theta) $$

M步，令：

$$\theta := \arg \max_\theta \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})} $$

}

但我们如何能证明最大期望算法一定能收敛呢？通过下式可以得到：

$$
\begin{align}
\ell(\theta^{(t+1)}) & \geq \sum_i \sum_{z^{(i)}} Q_i^{(t)} (z^{(i)}) \log \frac {p(x^{(i)}, z^{(i)}; \theta^{(t+1)})} {Q^{(t)}_i(z^{(i)})} \\
& \geq \sum_i \sum_{z^{(i)}} Q_i^{(t)} (z^{(i)}) \log \frac {p(x^{(i)}, z^{(    i)}; \theta^{(t)})} {Q^{(t)}_i(z^{(i)})} \\
&= \ell(\theta^{(t)})
\end{align}
$$

因此最大期望算法一定是逐渐收敛的，在实际应用中，我们常常是给定一个容忍系数来终止算法拟合。

假如我们定义：

$$J(Q, \theta) =
\sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac {p(x^{(i)}, z^{(i)}; \theta)} {Q_i(z^{(i)})} $$

最大期望算法可以被视作对于J的坐标上升法。在E步优化Q，在M步优化$$\theta$$。

## 混合高斯模型再回顾

有了最大期望算法的一般定义，我们再回头来拟合混合高斯模型的参数。

E步很简单：

$$\omega_j^{(i)}=Q_i(z^{(i)}=j)=P(z^{(i)}=j \mid x^{(i)}; \phi, \mu, \Sigma)$$

在M步，我们需要分别对参数$$\phi, \mu, \Sigma$$求目标函数的最大化:

$$
\begin{align}
& \sum_{i=1}^m \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac {p(x^{(i)},z^{(i)}; \phi, \mu, \Sigma)}
{Q_i(z^{(i)})} \\
=& \sum_{i=1}^m \sum_{j=1}^k Q_i(z^{(i)}=j) \log \frac {p(x^{(i)} \mid z^{(i)}=j; \mu, \Sigma)
p(z^{(i)}=j; \phi)} {Q_i(z^{(i)}=j)} \\
=& \sum_{i=1}^m \sum_{j=1}^k \omega_j^{(i)} \log \frac {\frac{1} {(2\pi)^{n/2} |\Sigma_j|^{1/2}}
\exp (-\frac{1}{2} (x^{(i)}-\mu_j)^T \Sigma_j^{-1} (x^{(i)}-\mu_j)) \cdot \phi_j} { \omega_j^{(i)}} \\
\end{align}
$$

首先来求解$$\mu_j$$，对其求梯度可以得到：

$$
\begin{align}
& \nabla_{\mu_j} \sum_{i=1}^m \sum_{j=1}^k \omega_j^{(i)} \log \frac {\frac{1} {(2\pi)^{n/2} |\Sigma_j|^{1/2}}
121 \exp (-\frac{1}{2} (x^{(i)}-\mu_j)^T \Sigma_j^{-1} (x^{(i)}-\mu_j)) \cdot \phi_j} { \omega_j^{(i)    }} \\
=& -\nabla_{\mu_j}\sum_{i=1}^m \sum_{j=1}^k \omega_j^{(i)} \frac{1}{2} (x^{(i)}-\mu_j)^T \Sigma_j^{-1} (x^{(i)}-\mu_j) \\
=& \frac{1}{2} \sum_{i=1}^m \omega_j^{(i)} \nabla_{\mu_j} (2\mu_j^T\Sigma_j^{-1}x^{(i)} -\mu_j^T\Sigma_j^T \mu_j) \\
=& \sum_{i=1}^m \omega_j^{(i)} (\Sigma_j^{-1}x^{(i)}-\Sigma_j^T \mu_j)
\end{align}
$$

将上式等于0可以得到：

$$\mu_j := \frac {\sum_{i=1}^m \omega_j^{(i)}x^{(i)}} {\sum_{i=1}^m \omega_j^{(i)}}$$

再来考虑$$\phi_j$$，我们需要最大化：

$$ \sum_{i=1}^m \sum_{j=1}^k \omega_j^{(i)} \log \phi_j $$

但认识到$$\phi_j$$并不是完全独立的，存在$$ \sum_{j=1}^k \phi_j = 1 $$的关系，因此我们可以构造拉格朗日方程：

$$ \mathcal{L} (\phi) = \sum_{i=1}^m \sum_{j=1}^k \omega_j^{(i)} \log \phi_j + \beta (\sum_{j=1}^k \phi_j - 1) $$

求偏导我们得到：

$$ \frac {\partial} {\partial \phi_j} \mathcal{L} (\phi) = \sum_{i=1}^m \frac {\omega_j^{(i)}} {\phi_j} + \beta $$

因此有：

$$ \phi_j = \frac { \sum_{i=1}^m \omega_j^{(i)} } { -\beta } $$

根据$$ \sum_{j} \phi_j = 1 $$的约束关系，$$ -\beta = \sum_{i=1}^m \sum_{j=1}^k \omega_j^{(i)} = \sum_{i=1}^m 1 = m $$，最终我们可以得到：

$$\phi_j := \frac {1} {m} \sum_{i=1}^m \omega_j^{(i)} $$

关于$$\Sigma_j$$ 也可以通过类似方法得到，不过让我自己推出来还是蛮困难的~

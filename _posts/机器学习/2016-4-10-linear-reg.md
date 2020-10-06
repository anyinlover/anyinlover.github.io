---
layout: single
title: "线性回归"
subtitle: "斯坦福大学机器学习第一讲"
date: 2016-4-10
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
  - 机器学习算法
---


几乎所有的机器学习书籍都以线性回归为起点，这是有道理的。第一，线性回归比较容易入门，第二，线性回归是后续很多现代算法的基础，第三，线性回归是一种应用非常广泛的算法。

简单线性回归的模型很简单，可以下面的公式表示：

$$
h(x)=\displaystyle\sum_{i=0}^{n} \theta_ix_i=\theta^Tx
$$

其中$$\theta$$称为参数或权重。上式实现从X空间到Y空间的映射。

拟合程度的好坏可以用下面的函数表示：

$$
J(\theta)=\frac{1}{2}\displaystyle\sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})^2
$$

$$J(\theta)$$被称为损失函数，这里的损失函数用估计值与实际值差的平方和来表示，是一种最常见的表示误差的方法。我们的目标就是使这个损失函数降到最小，实现最佳拟合。求最小二乘常用的有梯度下降法和举证计算法。

## 梯度下降法

梯度下降法的本质就是用迭代的思想，每一次都沿着最陡的方向（梯度）下降一小步，最终到达最小值。注意梯度下降法本身只能求局部最优解，但由于线性回归的J是个凸二次函数，因此局部最小值就是全局最小值。

先对$$\theta$$给出一个原始猜测值（一般可以取0），然后执行下面的迭代，其中$$\alpha$$是个固定值，称为学习率：

$$
\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
$$

对$$J(\theta)$$求偏导可得下式：

$$
\begin{align}
\frac{\partial}{\partial\theta_j}J(\theta)&=\frac{\partial}{\partial\theta_j}\frac{1}2(h_\theta(x)-y)^2 \\&=2\cdot\frac{1}2(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_j}(h_\theta-y)\\&=(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_j}(\displaystyle\sum_{i=0}^n \theta_ix_i-y)\\&=(h_\theta(x)-y)x_j
\end{align}
$$

所以有最终迭代的方程：

$$
\theta_j:=\theta_j+\alpha(y-h_\theta(x))x_j
$$

直观理解，二次函数的导数就是个一次函数，沿着导数方向下降，就是上面这式子。

上述式子是在只有一个训练样本时的情况，实际肯定会有大量的训练样本。对于多个训练样本，可以有两种变式。
第一种方法叫批梯度下降，每一次迭代时，我把所有训练样本的误差都跑一遍，叠加后对$$\theta$$进行更新。

$$
\theta_j:=\theta_j+\alpha\sum_{i=1}^m(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$

另一种方法叫随机梯度下降，每跑一个样本，我就迭代一次。

$$
\text{for }i=1\text{ to }m:\\
\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$

显然随机梯度下降会比批梯度下降跑的快，由于每一次迭代都要计算所有样本，批梯度非常耗时。另一方面，随机梯度很可能无法达到最小值，而是在最小值左右徘徊。因此对大数据集我们倾向于选择随机梯度，而对于小数据集则选择批梯度。

## 标准方程法

对于损失函数最小值的求解，除了上面的梯度下降法，还可以用矩阵直接进行计算。但其过程涉及到很多的矩阵推导，需要具备较高的数学基础。这种算法的本质就是利用了函数最小值处导数等于0的性质。

把上面的损失函数向量化，就是下面的表达式：

$$
J(\theta)=\frac{1}2(X\theta-\overrightarrow{y})^T(X\theta-\overrightarrow{y})
$$

将上面的损失函数对$$\theta$$求导，可以得到：

$$
\begin{align}
\nabla_{\theta}J(\theta)&=\nabla_{\theta}\frac{1}2(X\theta-\overrightarrow{y})^T(X\theta-\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}(\theta^TX^TX\theta-\theta^TX^T\overrightarrow{y}-\overrightarrow{y}^TX\theta+\overrightarrow{y}^T\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}tr(\theta^TX^TX\theta-\theta^TX^T\overrightarrow{y}-\overrightarrow{y}^TX\theta+\overrightarrow{y}^T\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}(tr\theta^TX^TX\theta-2tr\overrightarrow{y}^TX\theta)\\
&=\frac{1}2(X^TX\theta+X^TX\theta-2X^T\overrightarrow{y}）\\
&=X^TX\theta-X^T\overrightarrow{y}
\end{align}
$$

上述推导利用了矩阵的迹及相关性质，另篇专述。

令等式等于0，可以得到：

$$
X^TX\theta=X^T\overrightarrow{y}
$$

容易求出$$\theta$$：

$$
\theta = (X^TX)^{-1}X^T\overrightarrow{y}
$$

注意当训练集和特征值很多时，求矩阵的逆会很耗时，此时算法的性能不如前面的梯度下降法。

## 概率解释

这一节从概率的角度出发，论证了为什么把最小二乘作为损失函数是令人信服的选择。个人觉得这是一个比较有趣的角度。

假设我们的模型估计值和y真实值存在一个偏差$$\epsilon$$，表示未被模型考虑的因素或者是随机的噪音。进一步假设$$\epsilon^{(i)}$$是独立同分布的，服从$$\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2)$$的高斯分布：

$$
p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
$$

似然性函数等于：

$$
\begin{align}
L(\theta)&=\prod_{i=1}^m p(y^{(i)}\mid x^{(i)};\theta) \\
&=\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}
$$

根据最大似然性准则，我们需要选择$$\theta$$使似然性函数取到最大值。

两边同取log函数，得到对数似然性函数：

$$
\begin{align}
\ell(\theta)&=\log{L(\theta)}\\
&=\log\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})\\
&=\sum_{i=1}^m \log\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})\\
&=m\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\cdot\frac{1}{2}\displaystyle\sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})^2
\end{align}
$$

可以看出为了让$$\ell(\theta)$$最大，必须让$$\frac{1}{2}\displaystyle\sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})^2$$最小，这就是前面的损失函数$$h(\theta)$$。还可以发现$$\theta$$的取值和$$\sigma$$是无关的。

需要注意的是最小二乘并不是唯一合理的损失函数，最大似然性作为一种假设也不是推导损失函数的必要条件。我们还有其他合理的损失函数可以选择。

## 局部加权线性回归

局部加权线性回归的本质在于考虑了预测样本，对于训练集中和预测样本相似的训练样本，给予其更高的权值。

局部加权线性回归修改了简单线性回归的损失函数，添加了一个权值函数$$w^{(i)}$$：

$$
h(\theta)=\frac{1}{2}\sum_{i=1}^{m}w^{(i)} (h_\theta(x^{(i)})-y^{(i)})^2
$$

这个权值函数又是和预测样本相关联的，下面是一种标准的选择，x是预测样本的输入，$$\tau$$称为带宽，可以调整：

$$
w^{(i)}=\exp(-\frac{(x^{(i)}-x)^2}{2\tau^2})
$$

局部加权线性回归显然是一种比简单线性回归优化的算法，但他也被称为非参数算法，因为模型参数会随着预测样本变化，因此需要每一次进行即时计算。

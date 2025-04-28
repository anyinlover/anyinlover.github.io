# Chap2 Exercise

## Exercise 2.1

根据贝叶斯定理：

$$
p(H=h | Y = y) = \frac{p(H = h)p(Y = y | H = h)}{p(Y=y)}
$$

把$E_1 = e_1, E_2 = e_2$同时发生作为事件$Y = y$，代入即得：

$$
p(H = h | e_1, e_2) = \frac{p(H = h)p(e_1, e_2 | H = h)}{p(e_1, e_2)}
$$

因此需要条件二可以计算。

当$E_1, E_2$条件独立于$H$时，可以得到$p(E_1 | H)p(E_2 | H) = p(E_1, E_2 | H)$，此时有条件一就可以计算。

## Exercise 2.2

要证明随机变量两两独立不代表相互独立，只需要举出一个反例即可。设$X_1, X_2$是独立的二元随机变量，$X_3 = X_1 \oplus X_2 $，此时$p(X_3 | X_1) = p(X_3)$，因为$X_2$不可知。因此$X_1, X_2, X_3$是两两独立的，但$p(X_3 | X_1, X_2) = 1 \neq p(X_3)$。

## Exercise 2.3

对$p(x,y | z) = g(x, z)h(y, z)$两边对$x%积分，得到：

$$
\begin{align}
\int_X p(x,y | z)dx &= \int_X g(x,z)h(y, z) dx \\
p(y | z) &= h(y, z) \int_X g(x, z) dx \\
\end{align}
$$

令$\eta(z) = 1/\int_X g(x, z) dx $，可以得到$ h(y, z) = \eta(z)p(y | z) $

同理可得：$g(x, z) = \nu(z)p(x|z)$

因此有：$p(x,y|z) = \eta(z)\nu(z)p(x|z)p(y|z)$，两边对$x,y$积分，可以得到$\eta(z)\nu(z) = 1$，得证。

## Exercise 2.4

证明两个高斯分布的卷积还是一个高斯分布，即

$$ p(y) = \mathcal{N}(x_1 | \mu_1, \sigma_1^2 ) \otimes \mathcal{N}(x_2 | \mu_1, \sigma_1^2) = \mathcal{N}(y | \mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2) $$

其中 $ y = x_1 + x_2, x_1 \sim \mathcal{N}(\mu_1, \sigma_1^2 ), x_2 \sim \mathcal{N}(\mu_2, \sigma_2^2 )$

证明可以见[wikipedia](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables)，这里用了最粗暴的一种方式。

这个问题直观上容易理解成两个高斯密度函数相加，实际上不是这样的。
问题先从直观上理解，我们知道掷两个骰子之和近似满足高斯分布，那么如果是四个骰子之和，从直观上讲更加逼近高斯分布，而这就可以看作两个高斯分布的卷积。

![多个骰子之和满足高斯分布](https://en.wikipedia.org/wiki/Probability_distribution#/media/File:Dice_Distribution_(bar).svg)

令$\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}$，
在卷积下，密度函数有如下性质：

$$
\begin{align}
p(y) &= p(x_1 + x_2) \\
&= \int p_1(x_1)p_2(y - x_1) dx_1\\
&= \int \frac{1}{\sqrt{2\pi\sigma_1^2}} \exp(-\frac{1}{2\sigma_1^2}(x_1 - \mu_1)^2) \frac{1}{\sqrt{2\pi\sigma_2^2}} \exp(-\frac{1}{2\sigma_2^2}(y - x_1 - \mu_2)^2)dx_1 \\
&=  \int \frac{1}{2\pi \sigma_1 \sigma_2} \exp(-\frac{(\sigma_1^2 + \sigma_2^2)x^2 - 2(\sigma_1^2(y - \mu_2)+\sigma_2^2 \mu_1)x + \sigma_1^2(y^2 + \mu_2^2 - 2y\mu_2) + \sigma_2^2\mu_1^2}{2\sigma_1^2 \sigma_2^2}) dx \\
&= \int \frac{1}{2\pi \sigma_1 \sigma_2} \exp(-\frac{x^2 - 2\frac{\sigma_1^2(y - \mu_2) + \sigma_2^2 \mu_1}{\sigma^2}x  + \frac{\sigma_1^2(y^2 + \mu_2^2 - 2y\mu_2) + \sigma_2^2\mu_1^2}{\sigma^2}}{2(\frac{\sigma_1 \sigma_2}{\sigma})^2})dx \\
&= \int \frac{1}{\sqrt{2\pi}\sigma} \frac{1}{\sqrt{2\pi} \frac{\sigma_1 \sigma_2}{\sigma}} \exp(-\frac{(x - \frac{\sigma_1^2(y - \mu_2) + \sigma_2^2 \mu_1}{\sigma^2})^2 - (\frac{\sigma_1^2(y - \mu_2) + \sigma_2^2 \mu_1}{\sigma^2})^2  + \frac{\sigma_1^2(y^2 + \mu_2^2 - 2y\mu_2) + \sigma_2^2\mu_1^2}{\sigma^2}}{2(\frac{\sigma_1 \sigma_2}{\sigma})^2})dx \\
&= \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{\sigma^2(\sigma_1^2(y^2 + \mu_2^2 - 2y\mu_2) + \sigma_2^2\mu_1^2) - (\sigma_1^2(y - \mu_2) + \sigma_2^2 \mu_1)^2}{2\sigma^2 \sigma_1^2 \sigma_2}) \int \frac{1}{\sqrt{2\pi} \frac{\sigma_1 \sigma_2}{\sigma}} \exp(-\frac{(x - \frac{\sigma_1^2(y - \mu_2) + \sigma_2^2 \mu_1}{\sigma^2})^2}{2(\frac{\sigma_1 \sigma_2}{\sigma})^2})dx \\
&= \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(y - (\mu_1 + \mu_2))^2}{2\sigma^2})
\end{align}
$$

其中利用了右式也是一个高斯分布，pdf积分为1的特性。

## Exercise 2.5

这个问题从cdf角度考虑会比较简单。令$Z = \min(X, Y)$，则存在：

$$ Pr(Z > a) = Pr(X > a, Y > a) = (1-a)^2 $$

因此最小值的cdf函数为$P(z) = 1 - (1-z)^2$，进一步可以得到其pdf函数$p(z) = -2z^2 + 2z$，求得期望为：

$$ E(Z) = \int_0^1 p(z) = \frac{1}{3} $$

## Exercise 2.6

$$
\begin{align}
\mathbb{V}[X + Y] &= \mathbb{E}[(X+Y)^2] - (\mathbb{E}[X+Y])^2 \\
&= \mathbb{E}[X^2 + Y^2 + 2XY] - (\mathbb{E}[X] + \mathbb{E}[Y])^2 \\
&= \mathbb{E}[X^2] - (\mathbb{E}[X])^2 + \mathbb{E}[Y^2] - (\mathbb{E}[Y])^2 + 2(\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]) \\
&= \mathbb{V}[X] + \mathbb{V}[Y] + 2Cov[X, Y]
\end{align}
$$

## Exercise 2.7

令$ g = f^{-1}$，可以得到$x = g(y) = 1/x$

根据变量转换公式可以得：

$$
\begin{align}
p_y(y) &= p_x(g(y))|\frac{d}{dy}g(y)| \\
&= \frac{b^a}{\Gamma(a)} y^{1-a} e^{-b/y}y^{-2} \\
&= \frac{b^a}{\Gamma(a)} y^{-a-1} e^{-b/y}
\end{align}
$$

## Exercise 2.8

mode 比较好算，通过导数等于0可以得到：

$$
\begin{align}
0 &= \frac{df}{dx} \\
&= (a-1)x^{a-2}(1-x)^{b-1} - (b-1)x^{a-1}(1-x)^{b-2} \\
&= (a-1)(1-x) - (b-1)x
\end{align}
$$

得到$x = \frac{a-1}{a+b-1}$ 即mode

计算mean需要一些技巧

$$
\begin{align}
\mathbb{E}[x] &= \int_0^1 \frac{1}{B(a, b)}x^{a-1} (1-x)^{b-1}xdx \\
&= \frac{1}{B(a, b)} \int_0^1 x^{a+1-1} (1-x)^{b-1}dx \\
&= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \frac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+1+b)} \\
&= \frac{a}{a+b}
\end{align}
$$

这里利用了Gamma函数的定义式及其[$\Gamma(x+1) = x\Gamma(x)$](https://en.wikipedia.org/wiki/Gamma_function)的性质。

类似的可以计算得到：

$$
\mathbb{E}[x^2] = \frac{a}{a+b} \frac{a+1}{a+b+1}
$$

进而计算得到：

$$
\mathbb{V} = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 = \frac{ab}{(a+b)^2(a+b+1)}
$$

## Exercise 2.9

$p(H=h)$是得这种病的先验概率0.0001，$p(Y=y|H=h)$是真的得这种病后检验准确率0.99，重点要计算$p(Y=y)$，有两种情况，一种是真得被检验为真，令一种是假得被检验为真，综合概率为(0.0001\*0.99+0.9999\*0.01)

$$
p(H=h | Y=y) = \frac{p(H=h)p(Y=y|H=h)}{p(Y=y)} = 0.0098
$$

由此可见，对于罕见病而言，要考虑先验概率很小得情况，检测一次为阳性也不要害怕。这时的概率还很低，从上式还能看到基本只是先验概率*100的真实概率。

## Exercise 2.10

使用贝叶斯定理的时候一定要确定基础概率和新增信息。在这个案例中，法官其实是想计算在新增证据（血型符合）的条件下嫌疑人无罪的概率，令$E$表示证据，$I$表示无罪，$G$表示有罪，可以得到：

$$
p(I|E) = \frac{p(E|I)p(I)}{p(E|I)p(I) + p(E|G)p(G)} = \frac{0.01p(I)}{0.01p(I)+(1-p(I))}
$$

在未知$p(I)$先验概率的情况下，其实是无法计算得到$p(I|E)$的。只有当$p(I) = 0.5$的时候，才有$p(I|E) = p(E|I) = 0.01$。

律师的概率计算没有问题，但是不能说证据无关。因为其的确提高了嫌疑人有罪概率。

$$ \frac{p(G)}{p(I)} = \frac{1}{799999} $$

$$ \frac{p(G|E)}{p(I|E)} = \frac{p(E|G)p(G)}{p(E|I)p(I)} = \frac{1}{8000} $$

## Exercise 2.11

情况a是$2/3$, 不可能是两个男孩。剩下两种情况一男一女概率更高。

情况b是$1/2$，因为两个孩子性别概率是独立的，看到一个男孩不影响另一个孩子的性别概率分布。

## Exercise 2.12

$$
\begin{align}
Z^2 &= \int_0^{2\pi} \int_0^\infty r \exp(-\frac{r^2}{2\sigma^2})drd\theta \\
&= \int_0^{2\pi} d\theta \int_0^\infty -\sigma^2\exp(-\frac{r^2}{2\sigma^2})d(-\frac{r^2}{2\sigma^2}) \\
&= \sigma^22\pi
\end{align}
$$
从而得到$Z = \sqrt{\sigma^22\pi}$

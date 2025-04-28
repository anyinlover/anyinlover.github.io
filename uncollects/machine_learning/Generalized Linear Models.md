# Generalized Linear Models

In this chapter, we'll discuss generalized linear models. When I first encountered this chapter, I found it fascinating. After all, isn't the essence of knowledge about generalization? Being able to generalize specific models is a beautiful thing. In this case, both linear regression and logistic regression are considered as part of the generalized linear models (GLMs).

## The Exponential Family

Before discussing generalized linear models, it's necessary to understand the exponential distribution family. Any distribution that can be expressed in the following form belongs to the exponential distribution family:

$$
p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))
$$

Here, $\eta$ is known as the natural parameter, $T(y)$ is the sufficient statistic, which is often $T(y)=y$ for machine learning applications, and $a(\eta)$ is the logarithmic function used as a normalizing constant.

Next, we can demonstrate that both the Bernoulli distribution and the Gaussian distribution belong to the exponential distribution family.

$$
\begin{aligned}
p(y;\phi)&=\phi^y(1-\phi)^{1-y}\\
&=\exp(y\log\phi+(1-y)\log(1-\phi))\\
&=\exp((\log(\frac{\phi}{1-\phi}))y+\log(1-\phi))
\end{aligned}
$$

Therefore, we have $\eta=\log(\frac{\phi}{1-\phi})$, which leads to $\phi=1/(1+e^{-\eta})$. The other parameters can be obtained as follows:

$$
\begin{aligned}
T(y) &= y \\
a(\eta) &= -\log(1-\phi)\\
&= \log(1+e^\eta)\\
b(y) &= 1
\end{aligned}
$$

For the Gaussian distribution, since $\sigma^2$ does not affect the final result, we can set $\sigma^2=1$.

$$
\begin{aligned}
p(y;\mu)&=\frac{1}{\sqrt{2\pi}}\exp(-\frac{1}{2}(y-\mu)^2)\\
&=\frac{1}{\sqrt{2\pi}}\exp(-\frac{1}{2}y^2)\cdot\exp(\mu y-\frac{1}{2}\mu^2)
\end{aligned}
$$

Therefore, we have:

$$
\begin{aligned}
\eta &= \mu \\
T(y) &= y \\
a(\eta) &= \mu^2/2=\eta^2/2\\
b(y) &=(1/\sqrt(2\pi))\exp(-y^2/2)
\end{aligned}
$$

In addition, there are several other distributions that also belong to the exponential distribution family:

- Multinomial distribution: for modeling multiple discrete outputs.
- Poisson distribution: for modeling count processes.
- Gamma distribution and exponential distribution: for modeling continuous non-negative random variables, such as time intervals.
- Beta distribution and Dirichlet distribution: for modeling probability distributions.
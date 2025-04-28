# Regression

When we want to predict a real-valued quantity $y \in \R$, we have a regression problem. A common instance of it is the house price prediction problem.

The most common choice of loss function for regression is to use quadratic loss $\ell_2(y, \hat{y}) = (y - \hat{y})^2$, and the empirical risk when using quadratic loss is equal to the **mean squared error (MSE)**

$$
\text{MSE}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^{N} \left( y_n - f(\mathbf{x}_n; \boldsymbol{\theta}) \right)^2
$$

MSE is a reasonable choice and can be proved from [[Maximum Likelihood Estimate]].

Assume that there is a deviation $\epsilon$ between our model's estimated value and the true value $y$, representing unconsidered factors or random noise. Furthermore, assume that $\epsilon_n$ is independently and identically distributed, following a [[Gaussian Distribution]] $\epsilon\sim\mathcal{N}(0,\sigma^2)$,then we have the following conditional probability distribution:

$$
p(y_n | \mathbf{x}_n; \boldsymbol{\theta}) = p(\epsilon_n | \mathbf{x}_n; \boldsymbol{\theta}) = \frac{1}{\sqrt{2\pi}\sigma}\exp \left( -\frac{1}{2\sigma^2} (y_n - f(\mathbf{x}_n; \boldsymbol{\theta}))^2 \right)
$$

Then we get the corresponding average negative log likelihood:

$$
\begin{aligned}
\text{NLL}(\boldsymbol{\theta}) &= -\frac{1}{N} \sum_{n=1}^{N} \log \left[ \left( \frac{1}{2\pi\sigma^2} \right)^{\frac{1}{2}} \exp \left( -\frac{1}{2\sigma^2} (y_n - f(\mathbf{x}_n; \boldsymbol{\theta}))^2 \right) \right] \\
&= \frac{1}{2\sigma^2} \text{MSE}(\boldsymbol{\theta}) + \text{const}
\end{aligned}
$$

We see that the NLL is proportional to the MSE.

It should be noted that least squares is not the only reasonable loss function, and maximum likelihood as an assumption is not a necessary condition for deriving the loss function. There are other reasonable choices for the loss function.

Reference:

1. [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)
2. [CS229 Machine Learning](https://cs229.stanford.edu/)

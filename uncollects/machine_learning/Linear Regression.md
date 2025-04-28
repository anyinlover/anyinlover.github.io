# Linear Regression

The model for linear regression is straightforward and can be represented by the following formula:

$$
h(x)=\displaystyle\sum_{i=0}^{d} \theta_ix_i=\theta^Tx
$$

Here, $\theta$ refers to the parameters or weights. The equation implements a mapping from the $X$ space to the $Y$ space.

Linear Regression is a kind of [[Regression]] problem, so it has loss function:

$$
J(\theta)=\frac{1}{2}\sum_{i=1}^{n} (h_\theta(x^{(i)})-y^{(i)})^2
$$

## The Normal Equations

We can find the minimum of the loss function by directly solving a matrix equation. The essence of this algorithm relies on the property that the derivative of a function is zero at the minimum.

By vectorizing the loss function, we obtain the following expression:

$$
J(\theta)=\frac{1}2(X\theta-\overrightarrow{y})^T(X\theta-\overrightarrow{y})
$$

Taking the derivative of the loss function with respect to $\theta$, we obtain:

$$
\begin{aligned}
\nabla_{\theta}J(\theta)&=\nabla_{\theta}\frac{1}2(X\theta-\overrightarrow{y})^T(X\theta-\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}(\theta^TX^TX\theta-\theta^TX^T\overrightarrow{y}-\overrightarrow{y}^TX\theta+\overrightarrow{y}^T\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}tr(\theta^TX^TX\theta-\theta^TX^T\overrightarrow{y}-\overrightarrow{y}^TX\theta+\overrightarrow{y}^T\overrightarrow{y})\\
&=\frac{1}2\nabla_{\theta}(tr\theta^TX^TX\theta-2tr\overrightarrow{y}^TX\theta)\\
&=\frac{1}2(X^TX\theta+X^TX\theta-2X^T\overrightarrow{y}）\\
&=X^TX\theta-X^T\overrightarrow{y}
\end{aligned}
$$

Setting the equation to zero, we have:

$$
X^TX\theta=X^T\overrightarrow{y}
$$

From which we can solve for $\Theta$:

$$
\theta = (X^TX)^{-1}X^T\overrightarrow{y}
$$

Note that computing the inverse of a matrix is computationally expensive, especially for large datasets or a large number of features. In such cases, this method may be less efficient.

## Gradient Descent

A more common way is to use [[Stochastic Gradient Descent]], for linear regression, the loss function $J$ is a convex quadratic function, which means that the local minimum is also the global minimum.

Taking the partial derivative of $J(\theta)$ yields the following equation:

$$
\begin{aligned}
\frac{\partial}{\partial\theta_j}J(\theta)&=\frac{\partial}{\partial\theta_j}\frac{1}2(h_\theta(x)-y)^2 \\&=2\cdot\frac{1}2(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_j}(h_\theta-y)\\&=(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_j}(\displaystyle\sum_{i=0}^n \theta_ix_i-y)\\&=(h_\theta(x)-y)x_j
\end{aligned}
$$

Thus, the final update equation for gradient descent is:

$$
\theta_j:=\theta_j+\alpha(y-h_\theta(x))x_j
$$

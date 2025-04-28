# Chap3 Exercise

## Exercise 3.1

$$
Cov[X, Y] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X^3] = 0
$$

因此 $\rho(X, Y) =0$

## Exercise 3.2

$$
\begin{align}
0 &\leq \mathbb{V}[\frac{X}{\sigma_X} + \frac{Y}{\sigma_Y}] \\
&= \mathbb{V}[\frac{X}{\sigma_X}] + \mathbb{V}[\frac{Y}{\sigma_Y}] + 2Cov[\frac{X}{\sigma_X}, \frac{Y}{\sigma_Y}] \\
&= 1 + 1 + 2\rho \\
&= 2(1+\rho)
\end{align}
$$

可以得到$\rho \geq -1$

同理:

$$ 0 \leq \mathbb{V}[\frac{X}{\sigma_X} + \frac{Y}{\sigma_Y}] = 2(1-\rho) $$

可以得到$\rho \leq 1$

## Exercise 3.3

$$
\begin{align}
\rho(X,Y) &= \frac{Cov[X,Y]}{\sqrt{\mathbb{V}[X]} \sqrt{\mathbb{V}[Y]}} \\
&= \frac{\mathbb{E}[aX^2 + bX] - \mathbb{E}[X]\mathbb{E}[aX+b]}{|a|(\mathbb{E}[X^2] - \mathbb{E}^2[X])} \\
&= \frac{a}{|a|}    \\
&= \begin{cases}
    1 &\text{if } a > 0 \\
    -1 &\text{if } a < 0
    \end{cases}
\end{align}
$$

## Exercise 3.4

$$
\begin{align}
Cov[\bold{A}\bold{x}] &= \mathbb{E}[(\bold{A}\bold{x} - \mathbb{E}[\bold{A}\bold{x}])(\bold{A}\bold{x} - \mathbb{E}[\bold{A}\bold{x}])^T] \\
&= \mathbb{E}[\bold{A}(\bold{x} - \mathbb{E}[\bold{x}])(\bold{x} - \mathbb{E}[\bold{x}])^T\bold{A}^T] \\
&= \bold{A}\mathbb{E}[(\bold{x} - \mathbb{E}[\bold{x}])(\bold{x} - \mathbb{E}[\bold{x}])^T]\bold{A}^T \\
&= \bold{A} Cov[\bold{x}] \bold{A}^T
\end{align}
$$

# Newton Method

Newton's method is another way to find the maximum value of the log-likelihood function. Note that Newton's method must also be applied to a convex function. To find $f(\theta) = 0$, we can use the following iterative formula:

$$
\theta:=\theta-\frac{f(\theta)}{f'(\theta)}
$$

For finding the maximum value, i.e., when $f'(\theta)=0$, we use the following iterative formula:

$$
\theta:=\theta-\frac{\ell'(\theta)}{\ell''(\theta)}
$$

For vectorized $\theta$, the process becomes more complex and involves the use of the Hessian matrix (a $1+n,1+n$ matrix):

$$
\theta:=\theta-H^{-1}\nabla_{\theta}\ell(\theta)
$$

The calculation of the Hessian matrix (a $1+n,1+n$ matrix) is performed using the following method:

$$
H_{ij}=\frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}
$$

Newton's method generally converges faster than [[Stochastic Gradient Descent]], which can be roughly perceived from their iterative steps. However, similar to direct computation methods, Newton's method also involves matrix inversion, which can be computationally expensive when the number of features is large. Therefore, there is no free lunch in the world.

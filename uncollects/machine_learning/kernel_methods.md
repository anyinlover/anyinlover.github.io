# Kernel Methods

my shallow understanding is that kernel methods seem like small tricks, but they actually reflect the intrinsic nature of things. We can mapping attributes into high-dimensional spaces to scrap the rules between the attributes.

## Feature maps

In the prediction of housing prices, besides using a linear function, we can also use $x$, $x^2$, and $x^3$ to obtain a cubic function. Here, we call the original $x$ as the attribute and $x$, $x^2$, and $x^3$ as features. We denote the feature mapping as $\phi$. In our example, we have:

$$\phi(x)=\begin{bmatrix}x \\ x^2 \\ x^3\end{bmatrix}$$

## LMS with features and kernel tricks

Using feature mapping, in the linear regression problem, we have the batch gradient descent:

$$
\theta \coloneqq \theta+\alpha\sum_{i=1}^m(y^{(i)}-h_\theta(\phi(x^{(i)})))\phi(x^{(i)}) = \theta+\alpha\sum_{i=1}^m(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)})
$$

When features $\phi(x)$ is high-dimensional, then the direct computation costs a lot. However, we can apply a kernel trick that not compute and storage $\theta$ in the batch gradient descent. The main observation is that at any time, $\theta$ can be represented as a linear combination of the vector $\phi(x^{(1)})\cdots\phi(x^{(n)})$. At initialization, we have $\theta = 0 = \sum_{i=1}^n 0 \phi(x^{(i)})$. Assume at some point, $\theta$ can be represented as $\theta = \sum_{i=1}^n \beta_i \phi(x^{(i)})$, then in the next iteration, $\theta$ is still a linear combination.

$$
\begin{aligned}
\theta &\coloneqq \theta+\alpha\sum_{i=1}^m(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)}) \\
&= \sum_{i=1}^n \beta_i \phi(x^{(i)}) +\alpha\sum_{i=1}^m(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)}) \\
&= \sum_{i=1}^n (\beta_i + \alpha(y^{(i)}-\theta^T\phi(x^{(i)})))\phi(x^{(i)})
\end{aligned}
$$

So our strategy here is to implicitly represent the $\theta$ by a set of coefficients $\beta_1\cdots\beta_n$ by the iteration equation:

$$
\beta_i \coloneqq \beta_i + \alpha(y^{(i)}-\sum_{i=1}^n \beta_i \phi(x^{(i)})^T\phi(x^{(i)})) \text{ } \forall i \in \{1\cdots n\}
$$

We often rewrite $\phi(x^{(i)})^T\phi(x^{(i)})$ as $\langle \phi(x^{(i)})^T ,\phi(x^{(i)})z\rangle$ to emphasize that it's the inner product of the two feature vectors.

Now we have translated the problem, and it has two main advantages. First we can precompute $\phi(x^{(i)})^T\phi(x^{(i)})$. Second computing $\langle \phi(x^{(i)})^T ,\phi(x^{(i)})z\rangle$ can be efficient and does not necessarily require computing $\phi(x^{(i)})$ explicitly.

We define the corresponding kernel function as:

$$
K(x,z) \triangleq \langle \phi(x), \phi(z) \rangle
$$

We finally have the algorithm:

1. Compute all the values $K(x^{(i)}, x^{(j)})$ for all $i,j \in \{1\cdots n\}$. Set $\beta \coloneqq 0$
2. Loop
$$
\beta_i \coloneqq \beta_i + \alpha(y^{(i)}-\sum_{i=1}^n \beta_i K(x^{(i)}, x^{(j)})) \text{ } \forall i \in \{1\cdots n\}
$$

And we the prediction

$$
\theta^T \phi(x) = \sum_{i=1}^n \beta_i \phi(x^{(i)}) \phi(x) = \sum_{i=1}^n \beta_i  K(x^{(i)}, x)
$$

## Properties of kernels

In last subsection, we have seen the power of kernel function. In practice, we only need to apply $K(x,z)$ in the algorithm without needing to know the value of $\phi(x)$. Consider the following two examples.

For the kernel function $K(x,z)=(x^Tz)^2$:

$$
\begin{aligned}
K(x,z) &= (\sum_{i=1}^d x_iz_i)(\sum_{j=1}^d x_iz_i) \\
&= \sum_{i=1}^d \sum_{j=1}^d x_i x_j z_i z_j \\
&= \sum_{i,j=1}^d (x_ix_j)(z_iz_j)
\end{aligned}
$$

Thus, the feature mapping function is as follows, requiring $O(d^2)$ time, while the kernel function only requires $O(d)$:

$$\phi(x)=\begin{bmatrix}x_1x_1 \\ x_1x_2 \\ x_1x_3 \\ x_2x_1 \\ x_2x_2 \\ x_2x_3 \\ x_3x_1 \\ x_3x_2 \\ x_3x_3\end{bmatrix}$$

Now consider another related kernel function:

$$K(x,z)=(x^Tz+c)^2=\sum_{i,j=1}^d (x_ix_j)(z_iz_j)+ \sum_{i=1}^d( \sqrt{2c}x_i)(\sqrt{2c}z_i) + c^2$$

The corresponding feature function is:

$$\phi(x)=\begin{bmatrix}x_1x_1 \\ x_1x_2 \\ x_1x_3 \\ x_2x_1 \\ x_2x_2 \\ x_2x_3 \\ x_3x_1 \\ x_3x_2 \\ x_3x_3 \\ \sqrt{2c}x_1 \\ \sqrt{2c}x_2 \\ \sqrt{2c}x_3 \\ c\end{bmatrix}$$

More generally, the kernel function $K(x,z)=(x^Tz+c)^k$ maps the features to a $\binom{d+k}{k}$-dimensional feature space. It requires $O(d^k)$ time, while the kernel function still only requires $O(d)$.

From another perspective, roughly speaking, $K(x,z)=\phi(x)^T \phi(z)$ reflects the proximity relationship between $\phi(x)$ and $\phi(z)$—the closer they are, the larger the kernel function, and the further away they are, the smaller it is. For example, the following Gaussian kernel function effectively measures the relationship between x and z:

$$K(x,z)=\exp(- \frac{\|x-z\|^2}{2\sigma^2})$$

But now there is a question: how do I know that this kernel function is meaningful, i.e., it can find a feature mapping $\phi$ such that $K(x,z)=\phi(x)^T \phi(z)$?

Assuming K is a valid kernel function, given a finite point set $\{x^{(1)},\cdots,x^{(m)}\}$, let a kernel matrix be $K \in \mathbb{R}^{m \times m}, K_{ij}=K(x^{(i)},x^{(j)})$.

If K is a valid kernel function, then:

$$K_{ij}=K(x^{(i)},x^{(j)})=\phi(x^{(i)})^T\phi(x^{(j)})=\phi(x^{(j)})^T\phi(x^{(i)})=K(x^{(j)},x^{(i)})=K_{ji}$$

It can also be proved that the kernel matrix is positive semidefinite:

$$
\begin{aligned}
z^TKz &= \sum_i\sum_jz_iK_{ij}z_j \\
&=\sum_i\sum_jz_i\phi(x^{(i)})^T\phi(x^{(j)})z_j \\
&=\sum_i\sum_jz_i\sum_k\phi_k(x^{(i)})\phi_k(x^{(j)})z_j\\
&=\sum_k\sum_i\sum_jz_i\phi_k(x^{(i)})\phi_k(x^{(j)})z_j \\
&= \sum_k(\sum_iz_i\phi_k(x^{(i)}))^2 \\
&\geq 0
\end{aligned}
$$

Therefore, when the kernel function is valid, the kernel matrix is symmetric positive semidefinite. In fact, this is not only a necessary condition but also a sufficient condition, which can be summarized as the Mercer theorem.

In addition to being applied in support vector machines, kernel functions are widely used in other algorithms.

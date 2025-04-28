# Linear Algebra Reference

Matrix operations form the foundation of various algorithms.

## Linear Algebra

### Systems of Linear Equations

### Matrices

#### Matrix Addition and Multiplication

An identity matrix has 1s along the main diagonal and 0s elsewhere. It is often used as a way of creating something out of nothing, like a passerby on the road. Its amicable nature is reflected in the equation:

$$AI=A=IA$$

#### Inverse and Transpose

The transpose of a matrix involves swapping its rows and columns and is often required for certain operations, such as representing the sum of squares of vectors. The transpose possesses the following properties:

- $(A^T)^T=A$
- $(AB)^T=B^TA^T$
- $(A+B)^T=A^T+B^T$

The first and third properties are obvious, while the second one requires a bit of proof.

$$
\begin{aligned}
(AB)_{ij}^T&=(AB)_{ji}=\sum_{k=1}^nA_{jk}B_{ki}\\
&=\sum_{k=1}^nB_{ki}A_{jk}=\sum_{k=1}^nB_{ik}^TA_{kj}^T\\
&=(B^TA^T)_{ij}
\end{aligned}
$$

### Solving Systems of Linear Equations

### Vector Spaces

### Linear Independence

### Basis and Rank

### Linear Mappings

### Affine Spaces

## Analytic Geometry

### Norms

### Inner Products

### Lengths and Distances

### Angles and Orthogonality

### Orthonormal Basis

### Orthogonal Complement

### Inner Product of Functions

### Orthogonal Projections

### Rotations

## Matrix Decompositions

### Determinant and Trace

The trace of a matrix is the sum of its diagonal elements. Although it may not have direct applications, it is often used in intermediate steps of deductions. The trace has the following properties:

- For $A\in\mathbb{R}^{n\times n},trA=trA^T$
- For $A, B\in \mathbb{R}^{n\times n}, tr(A+B)=trA+trB$
- For $A\in\mathbb{R}^{n\times n},t\in \mathbb{R}, tr(tA)=t\,trA$
- For $A, B\text{ such that }AB\text{ is square}, trAB=trBA$
- For $A, B, C\text{ such that }ABC\text{ is square}, trABC=trBCA=trCAB$, and so on

The first three properties are straightforward, while the fifth can be derived from the fourth. Thus, only the fourth property needs to be proven.

$$
\begin{aligned}
trAB &= \sum_{i=1}^m(AB)_{ii}=\sum_{i=1}^m(\sum_{j=1}^nA_{ij}B_{ji})\\
&=\sum_{j=1}^n\sum_{i=1}^mB_{ji}A_{ij}=\sum_{j=1}^n(BA)_{jj}\\
&=trBA
\end{aligned}
$$

### Eigenvalues and Eigenvectors

### Cholesky Decomposition

### Eigendecomposition and Diagonalization

### Singular Value Decomposition

### Matrix Approximation

### Matrix Phylogeny

## Matrix Calculus

### The Gradient

The gradient involves taking the partial derivatives of a function with respect to a matrix, and the result has the same shape as the original matrix. Gradients are extensively used in machine learning and must be thoroughly understood.

The primary properties of gradients are simple:

- $\nabla_{x}(f(x)+g(x))=\nabla_{x}f(x)+\nabla_{x}g(x)$
- For $t\in \mathbb{R}, \nabla_{x}(tf(x))=t\nabla_{x}f(x)$

However, gradients also have other extended properties, which were utilized in direct computation methods during linear programming derivations.

- $\nabla_{A}trAB=B^T$
- $\nabla_{A^T}f(A)=(\nabla_Af(A))^T$
- $\nabla_{A}trABA^TC=CAB+C^TAB^T$

The proofs for these properties are as follows:

$$
\begin{aligned}
\frac{\partial}{\partial A_{ij}}trAB&=\frac{\partial}{\partial A_{ij}}\sum_{i=1}^m(AB)_{ii}\\
&=\frac{\partial}{\partial A_{ij}}\sum_{i=1}^m\sum_{j=1}^NA_{ij}B_{ji}\\
&=B_{ji}=B_{ij}^T
\end{aligned}
$$

$$
\begin{aligned}
(\nabla_{A^T}f(A))_{ij}&=\frac{\partial}{\partial A_{ij}^T}f(A)=\frac{\partial}{\partial A_{ji}}f(A)\\
&=(\nabla_{A}f(A))_{ji}=(\nabla_{A}f(A))_{ij}^T
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial}{\partial A_{ij}}trABA^TC&=tr(\frac{\partial AB}{\partial A_{ij}}A^TC+AB\frac{\partial A^TC}{\partial A_{ij}})\\
&=tr(BA^TC\frac{\partial A}{\partial A_{ij}})+tr(CAB\frac{\partial A^T}{\partial A_{ij}})\\
&=tr(BA^TC\frac{\partial A}{\partial A_{ij}})+tr(B^TA^TC^T\frac{\partial A}{\partial A_{ij}})\\
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial}{\partial A}trABA^TC&=(BA^TC)^T+(B^TA^TC^T)^T\\
&=C^TAB^T+CAB
\end{aligned}
$$

### The Hessian

Roughly speaking, if the gradient is a first-order derivative with respect to a matrix, then the Hessian is a second-order derivative with respect to a vector. This analogy helps us understand the essence of the Hessian. It can be expressed by the equation:

$$
\nabla_x^2f(x)=\nabla_x(\nabla_xf(x))^T
$$

Note that the second derivative actually involves taking the derivative of each element of $\nabla_xf(x)$, as the derivative of a vector is not defined.

The Hessian matrix is symmetric, meaning $H_{ij}=H_{ji}$.

### Gradients and Hessians of Quadratic and Linear Functions


### Least Sqaures

### Gradients of the Determinant

### Eigenvalues as Optimization

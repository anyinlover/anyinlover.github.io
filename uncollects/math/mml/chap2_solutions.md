# Chap2 Linear Algebra Solutions

2.1

We consider $(\R\backslash\{-1\}, \star)$, where:

$$
a \star b \coloneqq ab + a + b, \text{  } a,b \in \R \backslash \{-1\}
$$

a. Show that $(\R \backslash \{-1\}, \star)$ is an Abelian group.
b. Solve

$$
3 \star x \star x = 15
$$

in the Abelian group $(\R \backslash \{-1\}, \star)$, where $\star$ is defined before.

---

Closure: $x \star y = xy + x + y \in \R$

Associativity:

$$\begin{align*}
(x \star y) \star z &= (xy + x + y) \star z \\
&= (xy + x + y)z + xy + x + y + z \\
&= (yz + y + z)x + yz + y + z + x \\
&= x \star (yz + y + z) \\
&= x \star (y \star z)
\end{align*}$$

Neutral element: $\exist e = 0$, $x \star e = x$.

Inverse element: $e = 0, y = -\frac{x}{x+1}$, we have $x \star y = 0$ and $y \star x = 0$.

commutative: $x \star y = xy + x + y = yx + y + x = y \star x$

So $(\R\backslash\{-1\}, \star)$ holds all five properties of Abelian group.

$$\begin{align*}
3 \star x \star x &= 15 \\
3 \star (x \star x) &= 15 \\
3 \star (x^2 + 2x) &= 15 \\
3(x^2 + 2x) + (x^2 + 2x) + 3 &= 15 \\
x^2 + 2x -3 &= 0 \\
x &= 1 \text{ or } -3
\end{align*}$$
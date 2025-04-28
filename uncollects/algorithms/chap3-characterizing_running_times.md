# Characterizing Running Times

## $O$-notation, $\Omega$-notation, and $\Theta$-notation

There are three common asymptotic functions.

$O$-notation characterizes an upper bound on the asymptotic behavior of a function. In other words, it says that a function grows no faster than a certain rate, based on the highest-order term.

$\Omega$-notation characterizes a lower bound on the asymptotic behavior of a function. In other words, it says that a function grows at least as fast as a certain rate, based on the highest-order term.

$\Theta$-notation characterizes a tight bound on the asymptotic behavior of a function. In other words, it says that a function grows precisely at a certain rate, based on the highest-order term.

If we can show a function is both $O(f(n))$ and $\Omega(f(n))$, then the function is $\Theta(f(n))$.

Use Insertion sort as an example, based on its pseudocode, there is two loops, so the most operations is $n^2$, so it is $O(n^2)$.

If largest $n/3$ is at first $n/3$ positions, every element need to be moved into the last $n/3$ positions, so the least num of operations is $(n/3)(n/3) = n^2/9$, or $\Omega(n^2)$.

Finally we get $\Theta(n^2)$.

### Exercises

3.1-1

For the reversed-sorted array, the $i$ need to move to $n+1-i$, which at least need $n+1-2i$, so the sum move is

$$
\sum_{i=1}^{(n+1)/2} n+1-2i = (n^2-1)/4 = \Omega(n^2)
$$

3.1-2

there is two loops, so the most operations is $n^2$, so it is $O(n^2)$.

The two loops always need to run, so it it $\Omega(n^2)$ as well.

So the result is $\Theta(n^2)$.

3.1-3

$\alpha < 0.5 $

When $\alpha = 0.25$, $\alpha(1-2\alpha)$ is max.

## Asymptotic notation: formal definitions

Here is the formal definition of $O$-notation. For a given function $g(n)$, we denote by $O(g(n))$ the set of functions:

$$
O(g(n)) = \{f(n): \text{there exist positive constants } c \text{ and } n_0 \text{ such that } 0 \le f(n) \le cg(n) \text{ for all } n \ge n_0 \}
$$

Following is the formal definition of $\Omega$-notation. For a given function $g(n)$, we denote by $\Omega(g(n))$ the set of functions:

$$
\Omega(g(n)) = \{f(n): \text{there exist positive constants } c \text{ and } n_0 \text{ such that } 0 \le cg(n) \le f(n) \text{ for all } n \ge n_0 \}
$$

Following is the formal definition of $\Theta$-notation. For a given function $g(n)$, we denote by $\Theta(g(n))$ the set of functions:

$$
\Theta(g(n)) = \{f(n): \text{there exist positive constants } c_1, c_2 \text{ and } n_0 \text{ such that } 0 \le c_1g(n) \le f(n) \le c_2g(n) \text{ for all } n \ge n_0 \}
$$

Following is the theorem:

For any two functions $f(n)$ and $g(n)$, we have $f(n) = \Theta(g(n))$ if and only if $f(n) = O(g(n))$ and $f(n) = \Omega(g(n))$.

We need make sure that the asymptotic notation is used as precise as possible without overstating.

Insertion sort's worst-case running time is $\Theta(n^2)$, it's running time is $O(n^2)$ and $\Omega(n)$.

Here are two more notations:

We use $o$-notation to denote an upper bound that is not asymptotically tight.

$$
o(g(n)) = \{f(n): \text{there exist positive constants } c \text{ and } n_0 \text{ such that } 0 \le f(n) < cg(n) \text{ for all } n \ge n_0 \}
$$

We use $\omega$-notation to denote a lower bound that is not asymptotically tight.

$$
\omega(g(n)) = \{f(n): \text{there exist positive constants } c \text{ and } n_0 \text{ such that } 0 \le cg(n) < f(n) \text{ for all } n \ge n_0 \}
$$

Many of the relational properties of real numbers apply to asymptotic comparisons as well.

$f(n) = O(g(n))$ is like $a \le b$

$f(n) = \Omega(g(n))$ is like $a \ge b$

$f(n) = \Theta(g(n))$ is like $a = b$

$f(n) = o(g(n))$ is like $a < b$

$f(n) = \omega(g(n))$ is like $a > b$

And following are the properties:

**Transitivity**:

$f(n) = \Theta(g(n))$ and $g(n) = \Theta(h(n))$ imply $f(n) = \Theta(h(n))$

$f(n) = O(g(n))$ and $g(n) = O(h(n))$ imply $f(n) = O(h(n))$

$f(n) = \Omega(g(n))$ and $g(n) = \Omega(h(n))$ imply $f(n) = \Omega(h(n))$

$f(n) = o(g(n))$ and $g(n) = o(h(n))$ imply $f(n) = o(h(n))$

$f(n) = \omega(g(n))$ and $g(n) = \omega(h(n))$ imply $f(n) = \omega(h(n))$

**Reflexivity**:

$f(n) = \Theta(f(n))$

$f(n) = O(f(n))$

$f(n) = \Omega(f(n))$

**Symmetry**:

$f(n) = \Theta(g(n))$ if and only if $g(n) = \Theta(f(n))$

**Transpose symmetry**

$f(n) = O(g(n))$ if and only if $g(n) = \Omega(f(n))$

$f(n) = o(g(n))$ if and only if $g(n) = \omega(f(n))$

### Exercises

3.2-1

When $c_1 = 0.5$, $c_2 = 1$, we have:

$$
0.5(f(n) + g(n)) \le \max\{f(n), g(n)\} \le f(n) + g(n)
$$

3.2-2

$O(n^2)$ is the upper bound, its meaningless with "at least".

3.2-3

Yes, No

3.2-4

From

$$
O(g(n)) = \{f(n): \text{there exist positive constants } c_1 \text{ and } n_1 \text{ such that } 0 \le f(n) \le c_1g(n) \text{ for all } n \ge n_1 \}
$$

$$
\Omega(g(n)) = \{f(n): \text{there exist positive constants } c_2 \text{ and } n_2 \text{ such that } 0 \le c_2g(n) \le f(n) \text{ for all } n \ge n_2 \}
$$

together, we have

$$
\Theta(g(n)) = \{f(n): \text{there exist positive constants } c_1, c_2 \text{ and } \max\{n_1,n_2\} \text{ such that } 0 \le c_1g(n) \le f(n) \le c_2g(n) \text{ for all } n \ge \max\{n_1,n_2\} \}
$$

From

$$
\Theta(g(n)) = \{f(n): \text{there exist positive constants } c_1, c_2 \text{ and } n_0 \text{ such that } 0 \le c_1g(n) \le f(n) \le c_2g(n) \text{ for all } n \ge n_0 \}
$$

we get:

$$
O(g(n)) = \{f(n): \text{there exist positive constants } c \text{ and } n_0 \text{ such that } 0 \le f(n) \le cg(n) \text{ for all } n \ge n_0 \}
$$

$$
\Omega(g(n)) = \{f(n): \text{there exist positive constants } c \text{ and } n_0 \text{ such that } 0 \le cg(n) \le f(n) \text{ for all } n \ge n_0 \}
$$

3.2-5

Its worst-case running time is $O(g(n))$, so the running time is $O(g(n))$ too.

Its best-case running time is $\Omega(g(n))$, so the running time is $\Omega(g(n))$ too.

Based on Theorem 3.1, We get $\Theta(g(n))$

If it is $\Theta(g(n))$, then it is $O(g(n))$ and $\Omega(g(n))$ too, then its worst-case running time is $O(g(n))$ too, its best-case running time is $\Omega(g(n))$ too.

3.2-6

Based on definition, there always is:

$$
c_2(g(n)) < f(n) < c_1(g(n))
$$

So the set is empty.

3.2-7

$$
\Omega(g(n,m)) = \{f(n,m): \text{there exist positive constants } c, n_0 \text{ and } m_0 \text{ such that } 0 \le cg(n,m) \le f(n,m) \text{ for all } n \ge n_0 \text{ or } m \ge m_0 \}
$$

$$
\Theta(g(n.m)) = \{f(n,m): \text{there exist positive constants } c_1, c_2, n_0 \text{ and } m_0 \text{ such that } 0 \le c_1g(n,m) \le f(n,m) \le c_2g(n,m) \text{ for all } n \ge n_0 \text{ or } m \ge m_0 \}
$$

## Standard notations and common functions

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots = sum_{i=0}^\infty \frac{x^i}{i!} 
$$

When $|x| \le 1 $, we have the approximation

$$
1 + x \le e^x \le 1 + x + x^2
$$

When $ x \rightarrow 0 $

$$
e^x = 1 + x + \Theta(x^2)
$$

We have for all x,

$$
\lim_{n \rightarrow \infty} (1 + \frac{x}{n})^n = e^x
$$

For $ x > -1 $, where equality holds only for $ x = 0 $

$$
\frac{x}{1+x} \le \ln(1+x) \le x
$$

Following is Stirling's approximation

$$
n! = \sqrt{2\pi n} (\frac{n}{e})^n (1 + \Theta(\frac{1}{n}))
$$

For all $ n \ge 1 $, where $ \frac{1}{12n+1} < \alpha_n < \frac{1}{12n}$

$$
n! = \sqrt{2\pi n}(\frac{n}{e})^n e^{\alpha_n}
$$

We define the notation $\lg^\ast n$ as:

$$
\lg^\ast n = \min \{i \ge 0 :\lg^{(i)}n \le 1 \}
$$

Fibonacci numbers are related to the golden ratio $\phi$ and its conjugate $\hat \phi$, which are the two roots of the equation

$$
x^2 = x + 1
$$

We get

$$
\phi = \frac{1+\sqrt5}{2} = 1.61803...
$$

$$
\hat\phi = \frac{1-\sqrt5}{2} = -.61803...
$$

We have

$$
F_i = \frac{\phi^i - \hat\phi^i}{\sqrt5}
$$

Since $ | \hat \phi | < 1 $, so $ \frac{|\hat\phi^i|}{\sqrt5} < \frac{1}{\sqrt5} < 1/2$, so:

$$
F_i = \lfloor \frac{\phi^i}{\sqrt5} + \frac{1}{2} \rfloor
$$

Fibonacci numbers grow exponentially.

### Exercises

3.3-1

$f(n)$ and $g(n)$ are monotonically increasing functions, so for every $ m \le n $, there are $ f(m) \le f(n) $ and $g(m) \le g(n)$, we can get:

$$
f(m) + g(m) \le f(n) + g(n)
$$

$$
f(g(m)) \le f(g(n))
$$

When $f(n)$ and $g(n)$ are nonnegative:

$$
f(m)g(m) \le f(n)g(n)
$$

3.3-2

Based on

$$
x - 1 < \lfloor x \rfloor \le x \le \lceil x \rceil < x + 1
$$

We have:

$$
\alpha n - 1 < \lfloor \alpha n \rfloor \le \alpha n
$$

$$
(1-\alpha)n \le \lceil (1-\alpha)n \rceil < (1-\alpha)n + 1
$$

Add two, we get:

$$
n - 1 < \lfloor \alpha n \rfloor + \lceil (1-\alpha)n \rceil < n + 1
$$

Because its a integer, so must:

$$
\lfloor \alpha n \rfloor + \lceil (1-\alpha)n \rceil = n
$$

3.3-3

$$
\Theta(n^k) = 1/2^k n^k = (n - n/2)^k < (n + o(n))^k < (n + n)^k = 2^k n^k = \Theta(n^k)
$$

So we get $(n + o(n))^k = \Theta(n^k)$.

$$
\lceil n \rceil ^k = (n + \lceil n \rceil - n)^k = (n + o(n))^k = \Theta(n^k)
$$

$$
\lfloor n \rfloor ^k = (n + n - \lfloor n \rfloor)^k = (n + o(n))^k = \Theta(n^k)
$$

3.3-4

$$
a^{\log_b c} = a^{\log_a c / \log_a b} = c^{1 / \log_a b} = c^{\log_b a}
$$

$$
n! = 1 * 2 * \cdots * n < n * n * \cdots * n = n^n
$$

$$
n! = 1 * 2 * \cdots * n > 0.5 * (2 * 2 * \cdots * 2) = 0.5*2^n
$$

Using Stirling's approximation

$$
\lg(n!) = \lg(\sqrt{2\pi n} (\frac{n}{e})^n (1 + \Theta(\frac{1}{n}))) = \frac{1}{2} \lg(2\pi n) + n \lg(n) - n \lg(e) + \lg(\Theta(\frac{n+1}{n}))
$$

The last term is $O(\lg(n))$, so the whole expression is $\Theta(n \lg(n))$.

For $\Theta(n)$, there exist positive constants $c_1, c_2, n_0$, such that $0 \le c_1n \le f(n) \le c_2n$ for all $n \ge n_0$.

For $f'(n) = \lg f(n)$, we have $\lg c_1 + \lg n = \lg(c_1 n) \le f'(n) \le \lg(c_2 n) = \lg c_2 + \lg n$

If we let $c'_1 = 0.5, c'_2 = 2$, when $n'_0$ is large enough, there must have $ c'_1 \lg n \le \lg c_1 + \lg n \le f'(n) \le \lg c_2 + \lg n \le c'_2 \lg n$.

3.3-5

It is a hard problem, and I refer it with [solution](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/11599/selected-solutions.pdf).

We first prove that a function *f(n)* is polynomially bounded is equivalent to proving that $\lg f(n) = O(\lg n)$ for the following reasons.

If f(n) is polynomially bounded, then there exist positive constants $c, k, n_0$, $c \ge 1$, $n_0 \le 2$, $ 0 \le f(n) \le c n^k$ for all $ n \le n_0$. 

then we get $ \lg f(n) \le \lg c + k \lg n \le (\lg c + k) \lg n $, means $\lg f(n) = O(\lg n)$.

If $\lg f(n) = O(\lg n)$, then there exist positive cnstants  $c, k, n_0$, $ 0 \le f(n) \le c n^k$ for all $ n \le n_0$. Then we have:

$$
0 \le f(n) = 2^{\lg (fn)} \le 2^{c \lg n} = (2^{\lg n})^c = n^c
$$

Next we have $\lceil \lg n \rceil = \Theta(\lg n)$ as $\lceil \lg n \rceil \ge \lg n$ and $\lceil \lg n \rceil < \lg n + 1 \le 2 \lg n$ for all $n \ge 2$.

And we have $\lg(n!) = \Theta(n \lg(n))$, now we are ready to answer two questions.

$$
\begin{align*}
\lg(\lceil \lg n \rceil !) &= \Theta(\lceil \lg n \rceil \lg \lceil \lg n \rceil) \\
&= \Theta((\lg n)(\lg \lg n)) \\
&= \omega(\lg n)
\end{align*}
$$

Its not $O(\lg n)$, so $\lceil \lg n \rceil !$ is not polynomially bounded.

$$
\begin{align*}
\lg(\lceil \lg \lg n \rceil !) &= \Theta(\lceil \lg \lg n \rceil \lg \lceil \lg \lg n \rceil) \\
&= \Theta((\lg \lg n)(\lg \lg \lg n)) \\
&= o((\lg \lg n)^2) \\
&= o(\lg^2(\lg n)) \\
&= o(\lg n)
\end{align*}
$$

Therefore, $\lg(\lceil \lg \lg n \rceil !) = O(\lg n)$, so $\lceil \lg \lg n \rceil !$ is polynomially bounded.

3.3-6

If $\lg^\ast n = k$, then we have $\lg^\ast (\lg n) = k - 1$, $\lg (\lg^\ast n) = \lg k$, so the latter is larger.

3.3-7

$$
(\frac{1 +\sqrt5}{2})^2 = \frac{3 + \sqrt5}{2} = \frac{1 + \sqrt5}{2} + 1
$$

$$
(\frac{1- \sqrt5}{2})^2 = \frac{3 - \sqrt5}{2} = \frac{1 - \sqrt5}{2} + 1
$$

3.3-8

When $i=0$, $F_0 = \frac{\phi^0 - \hat\phi^0}{\sqrt5} = 0 $

When $i=1$, $F_1 = \frac{\phi^1 - \hat\phi^1}{\sqrt5} = 1 $

When $i \ge 2$, we have:

$$
\begin{align*}
F_i &= F_{i-2} + F_{i-1} \\
&= \frac{\phi^{i-2} - \hat\phi^{i-2}}{\sqrt5} + \frac{\phi^{i-1} - \hat\phi^{i-1}}{\sqrt5} \\
&=  \frac{\phi^{i-2}(1 + \phi) - \hat\phi^{i-2}(1 + \hat\phi)}{\sqrt5} \\
&= \frac{\phi^{i-2} \phi^2 - \hat\phi^{i-2} \hat\phi^2}{\sqrt5} \\
&= \frac{\phi^{i} - \hat\phi^{i}}{\sqrt5}
\end{align*}
$$

3.3-9

Refered with [rutgers solution](https://sites.math.rutgers.edu/~ajl213/CLRS/Ch3.pdf)

We have $c_1n \le k \lg k \le c_2 n$, so $\lg c_1 + \lg n = \lg (c_1n) \le \lg (k \lg k) = \lg k + \lg (\lg k)$, so $\lg n = O(\lg k)$. If $ \lg n \le c_3 \lg k$, Then

$$
\frac{n}{\lg n} \ge \frac{n}{c_3 \lg k} \ge \frac{k}{c_2c_3}
$$

so that $\frac{n}{\lg n} = \Omega(k)$. Similarly, we have $\lg k + \lg (\lg k) = \lg (k \lg k) \le \lg (c_2 n) = \lg c_2 + \lg n$, so $ \lg n = \Omega (\lg k)$. If $\lg n \ge c_4 \lg k $, then

$$
\frac{n}{\lg n} \le \frac{n}{c_4 \lg k} \le \frac{k}{c_1c_4}
$$

So that $\frac{n}{\lg n} = O(k)$, together we have $\frac{n}{\lg n} = \Theta(k)$. By symmetry, we have $ k = \Theta(\frac{n}{\lg n})$


## Problems

3-1

a. 
When $c = \max{a_i} * (d + 1)$, we have:

$$
c n^k = \sum_{i=0}^d \max{a_i} n^k >= \sum_{i=0}^d a_i n^i = p(n)
$$

b.

When $c = a_d/2$, we have:

$$
c n^k = a_d/2* n^k \le a_d n^d / 2 \le p(n)
$$

c. 

Based on a and b, when $c_1 = a_d$, $c_2 = \max{a_i} * (d + 1)$, we have:

$$
c_1 n^k \le p(n) \le c_2 n^k
$$

d. 

When $c = \max{a_i} * (d + 1)$, $n_0 = 2$, we have:

$$
c n^k = \sum_{i=0}^d \max{a_i} n^k > \sum_{i=0}^d a_i n^i = p(n)
$$

e. 

When $c = a_d/2$, we have:

$$
c n^k = a_d/2 * n^k < a_d n^d / 2 \le p(n)
$$

3.2

$A$ | $B$ | $O$ | $o$ | $\Omega$ | $\omega$ | $\Theta$
|---| --- |---- | ---- | ------- | -------- | -------
| $\lg ^k n$ | $n^\epsilon$ | yes | yes | no | no | no |
| $n^k$ | $c^n$ | yes | yes | no | no | no |
| $\sqrt n$ | $n^{\sin n}$ | no | no | no | no | no |
| $2^n$ | $2^{n/2}$ | no | no | yes | yes | no |
| $n^{\lg c}$ | $c^{\lg n}$ | yes | no | yes | no | yes |
| $\lg(n!)$ | $\lg(n^n)$ | yes | no | yes | no | yes |

3.3
a.

$$
n^{1/\lg n} = 1\\
\lg (\lg^\ast n) \\
\lg^\ast n = \lg^\ast (\lg n) \\
2^{\lg^\ast n} \\
\ln \ln n \\
\sqrt{\lg n} \\
\ln n \\
\lg^2 n \\
2^{\sqrt{2\lg n}} \\
(\sqrt2)^{\lg n} = \sqrt n \\
2^{\lg n} = n \\
lg(n!) = n\lg n \\
n^2 = 4 ^ {\lg n} \\
n^3 \\
(\lg n)! \\
n^{\lg \lg n} = (\lg n)^{\lg n} \\
(3/2)^n \\
2^n \\
n 2^n \\
e^n \\
n! \\
(n+1)! \\
2^{2^n} \\
2^{2^{n+1}} \\
$$

b.

$$
f(n) = \begin{cases}
2^{2^{n+1}} \text{ if } n \text{ mod } 2 = 0 \\
1 \text{ if } n \text{ mod } 2 = 1
 \end{cases}
$$

3-4

a. false

$ n = O(n^2) $, but $ n^2 \neq O(n)$

b. false

$ n + n^2 \neq \Theta(n) $

c. true

$ f(n) \le cg(n)$, so $ \lg f(n) \le \lg (cg(n)) = \lg c + \lg g(n) \le (\lg c + 1) \lg g(n)$

d. false

$ 2n = O(n) $ and $ 2^{2n} \neq 2^n $.

e. false

$ f(n) = 1/n $

f. true

$ f(n) \le c g(n)$ so we get $ g(n) \ge 1/c f(n)$

g. false

$ f(n) = 2^n $

h. true

Let $ g(n) = o(f(n))$, we have

$ f(n) + g(n) \ge f(n) = \Omega(f(n)) $ and $ f(n) + g(n) \le f(n) + f(n) = 2f(n) = O(f(n))$. Finally we get $f(n) + o(f(n)) = \Theta(f(n))$.

3-5

a.

Because $ c_1 f(n) \le \Theta(f(n)) \le c_2 f(n)$, so we have:

$$
c'_1 c_1 f(n) \le c'_1 \Theta(f(n)) \le \Theta(\Theta(f(n))) \le c'_2 \Theta(f(n)) \le c'_2 c_2 f(n)
$$

b.

$c_1 f(n) \le \Theta(f(n)) \le c_2 f(n)$ and $ 0 \le O(f(n)) \le c_3 f(n)$, so we have:

$$
c_1 f(n) \le \Theta(f(n)) + O(f(n)) \le (c_2 + c_3) f(n)
$$

c.

$c_1 f(n) \le \Theta(f(n)) \le c_2 f(n)$ and $c'_1 g(n) \le \Theta(g(n)) \le c'_2 g(n)$, we have:

$$
\min\{c_1, c'_1\}(f(n)+g(n)) \le c_1f(n) + c'_1f(n) \le \Theta(f(n)) + \Theta(g(n)) \le c_2f(n) + c'_2g(n) \le \max\{c_2, c'_2\}(f(n) + g(n))
$$

d.

$c_1 f(n) \le \Theta(f(n)) \le c_2 f(n)$ and $c'_1 g(n) \le \Theta(g(n)) \le c'_2 g(n)$, we have:

$$
c_1 c'_1 f(n)g(n) \le \Theta(f(n))\Theta(g(n)) \le c_2c'_2f(n)g(n)
$$

e.

$\lg^{k_2}(a_2 n) = \lg^{k_2-1}(\lg n + \lg a)$, when $n$ is large enough, there have

$$
\lg^{k_2-1}(\lg n + \lg a) \le \lg^{k_2-1}(2\lg n) = lg^{k_2-2}(\lg\lg n + \lg 2) \le lg^{k_2-2}(2\lg\lg n) \le \cdots = 2\lg^{k_2}n
$$

Similarly, we have $1/2\lg^{k_2}n \le \lg^{k_2-1}(\lg n + \lg a)$, together, we have:

$$
1/2a_1^{k_1} n_{k_1} \lg^{k_2}n \le (a_1n)^{k_1} \lg^{k_2}(a_2n) \le 2a_1^{k_1} n_{k_1} \lg^{k_2}n
$$

f. It seems right, todo.

g. I don't know, todo.

$ \prod_{k \in S} \Theta(f(k)) = \Theta(\prod_{k \in S} f(k)) $

3-6

a. If $ f(n) \neq O(g(n))$, which mean that $\forall c, n_0, \exists n \ge n_0, f(n) > c(g)$, Initially, we set $n+0 = 1, c = 1$, we get the $n = a_1$, then let $n_0 = a_i + 1$, we get the next $n = a_{i+1}$, so we can have a infinite set that $f(n) \ge cg(n)$.

b. $f(n) = \sin(n) + 1, g(n) = \cos(n) + 1$

c. Advantage: It has a property as complete opposite of O; Disadvantage: it's uneasy to apply.

d. Both directions still hold.

$ c_1g(n) \le f(n)$ and $ f(n) \le |f(n)| \le c_2g(n)$, so we have $ c_1g(n) \le f(n) \le c_2g(n).

$ 0 \le c_1g(n) \le f(n) \le c_2g(n)$, so $ |f(n) | = f(n) \le c_2g(n)$

e.

$$
\tilde\Omega(g(n)) = \{f(n): \text{there exist positive constants } c, k \text{ and } n_0 \text{ such that } 0 \le cg(n) \lg^k(n) \le f(n) \text{ for all } n \ge n_0 \}
$$

$$
\Theta(g(n)) = \{f(n): \text{there exist positive constants } c_1, c_2, k_1, k_2 \text{ and } n_0 \text{ such that } 0 \le c_1g(n)\lg^{k_1}(n) \le f(n) \le c_2g(n)\lg^{k_2} \text{ for all } n \ge n_0 \}
$$

The theroem still hold like before. 

3-7

| $f(n)$ | $c$ | $f^\ast_c(n)$ |
| ------ | --- | ------------- |
| $n-1$  | 0   | $n-1$ |
| $\lg n$ | 1 | $\lg^\ast n$ |
| $n/2$ | 1 | $\lg n$
| $n/2$ | 2 | $\lg n - 1$ |
| $\sqrt n$ | 2 | $\lg \lg n$
| $\sqrt n$ | 1 | undefined |
| $n^{1/3}$ | 2 | $\lg_3 \lg n$ |
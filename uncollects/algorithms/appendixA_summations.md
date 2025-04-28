# Summation

## Summation formulas and properties

A.1-1

$$
\begin{align*}
\sum_{k=1}^n O(f_k(i)) &\le \sum_{k=1}^n c_k f_k(i) \\
&\le \sum_{k=1}^n c_{max} f_k(i) \\
&\le c_{max} \sum_{k=1}^n f_k(i) \\
&= O(\sum_{k=1}^n f_k(i))
\end{align*}
$$

A.1-2

$$
\sum_{k=1}^n (2k-1) = 2\sum_{k=1}^n k -n = n^2 
$$

A.1-3

$$
111111111 = \sum_{k=0}^8 10^k = \frac{10^9 - 1}{10-1} = \frac{999999999}{9}
$$

A.1-4

$$
\begin{align*}
1 - \frac{1}{2} + \frac{1}{4} - \frac{1}{8} + \frac{1}{16} - \cdots &= \sum_{i=0}^{\infty} (-\frac{1}{2})^i \\
&= \frac{(-\frac{1}{2})^\infty - 1}{-\frac{1}{2} - 1} \\
&= \frac{2}{3}
\end{align*}
$$

A.1-5

$$
\sum_{k=1}^n k^c \ge \int_0^n x^c dx = \frac{n^{c+1}}{c+1} = \Omega(n^{c+1})
$$

$$
\sum_{k=1}^n k^c \le \int_1^{n+1} x^c dx = \frac{(n+1)^{c+1} - 1}{c+1} = O(n^{c+1})
$$

So we get $\sum_{k=1}^n k^c = \Theta(n^{c+1})$.

A.1-6

Differentiating both sides of A.11, we get:

$$
\sum_{k=0}^\infty k^2 x^k-1 = \frac{1+x}{1-x^3}
$$

By multiplying x, we get:

$$
\sum_{k=0}^\infty k^2 x^k = \frac{x(1+x)}{1-x^3}
$$

A.1-7

$$
\begin{align*}
\sum_{k=1}^n \sqrt{k \lg k} &\ge \int_0^n \sqrt{x \lg x} dx \\
&= \int_0^n x^{1/2} \lg^{1/2} x dx \\
&= \frac{2}{3} \int_0^n \lg^{1/2} x dx^{3/2} \\
&= \frac{2}{3} (n^{3/2} \lg^{1/2}n - \int_0^n x^{3/2} d\lg^{1/2} x) \\
&= \frac{2}{3} (n^{3/2} \lg^{1/2}n - \frac{1}{2}\int_0^n x^{1/2} dx) \\
&= \frac{2}{3} (n^{3/2} \lg^{1/2}n - \frac{1}{2}n^{3/2}) \\
&= \Omega(n^{3/2} \lg^{1/2}n)
\end{align*}
$$

Similarly,

$$
\begin{align*}
\sum_{k=1}^n \sqrt{k \lg k} &\le \int_1^{n+1} \sqrt{x \lg x} dx \\
&= \frac{2}{3} ((n+1)^{3/2} \lg^{1/2}(n+1) - \int_1^{n+1} x^{3/2} d\lg^{1/2} x) \\
&= \frac{2}{3} ((n+1)^{3/2} \lg^{1/2}(n+1) - \frac{1}{2}((n+1)^{3/2}-1)) \\
&= O(n^{3/2} \lg^{1/2}n)
\end{align*}
$$

A.1-8

$$
\begin{align*}
\sum_{k=1}^n 1/(2k-1) &= \sum_{k=1}^{2n}1/k - \frac{1}{2}\sum_{k=1}^n 1/k \\
&= \ln(2n) + O(1) - \frac{1}{2}(\ln(n) + O(1)) \\
&= \ln(\sqrt n) + O(1)
\end{align*}
$$

A.1-9

$$
\begin{align*}
\sum_{k=0}^\infty(k-1)/2^k &= \sum_{k=0}^\infty k/2^k - \sum_{k=0}^\infty 1/2^k \\
&= \frac{1/2}{(1-1/2)^2} - \frac{1}{1-1/2} \\
&= 0
\end{align*}
$$

A.1-10

$$
\begin{align*}
\sum_{k=1}^\infty(2k+1)x^{2k} &= 2\sum_{k=0}^\infty k(x^2)^k + \sum_{k=0}^\infty (x^2)^k - 1 \\
&= 2 \frac{x^2}{(1-x^2)^2} + \frac{1}{1-x^2} - 1 \\
&= \frac{3x^2 - x^4}{(1-x^2)^2}
\end{align*}
$$

A.1-11

$$
\begin{align*}
\prod_{k=2}^n(1 - 1/k^2) &= \prod_{k=2}^n \frac{(k-1)(k+1)}{k^2} \\
&= \prod_{k=2}^n \frac{k-1}{k} \frac{k+1}{k} \\
&= \frac{1}{2} \frac{n+1}{n} \\
&= \frac{n+1}{2n}
\end{align*}
$$

## Bounding summations

A.2-1

$$
\begin{align*}
\sum_{k=1}^n 1/k^2 &\le 1 + \int_0^1 1/x^2 dx \\
&= 1 + 1 - \frac{1}{n} \\
&< 2
\end{align*}
$$

A.2-2

$$
\begin{align*}
\sum_{k=0}^{\lfloor \lg n \rfloor} \lceil n / 2^k \rceil &\le \sum_{k=0}^{\lfloor \lg n \rfloor} (n/2^k + 1) \\
&= n \sum_{k=0}^{\lfloor \lg n \rfloor} 1/2^k + \lfloor \lg n \rfloor + 1 \\
&= n(2-1/2^{\lfloor \lg n \rfloor}) + \lfloor \lg n \rfloor + 1 \\
&\le n(2-1/n) + \lg n + 1 \\
&= 2n + \lg n \\
&= O(n)

\end{align*}
$$

A.2-3

$$
\begin{align*}
\sum_{k=1}^n \frac{1}{k} &\ge \sum_{i=1}^{\lfloor \lg n \rfloor} \sum_{j=0}^{2^{i-1}-1} \frac{1}{2^i - j} \\
&\ge \sum_{i=1}^{\lfloor \lg n \rfloor} 2^{i-1}/2^i \\
&= \frac{1}{2} \lfloor \lg n \rfloor \\
&\ge \frac{1}{2} (\lg n - 1) \\
&= \Omega(\lg n)
\end{align*}
$$

A.2-4

$$
\sum_{k=1}^n k^3 \ge \int_0^n x^3 dx = \frac{1}{4} n^4
$$

$$
\sum_{k=1}^n k^3 \le \int_1^{n+1} x^3 dx = \frac{1}{4} ((n+1)^4 - 1)
$$

A.2-5

If use approximation directly, we get:

$$
\sum_{k=1}^n 1/k \le \int_0^n 1/x dx = \ln x |_0^n
$$

However, $\ln 0$ is meanless, so we can't use it directly.

## Problems
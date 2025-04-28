# Counting and Probability

## Counting

C.1-1

$n-k+1$

$$
\sum_{k=1}^n n-k+1 = \sum_{k=1}^n k = n(n+1)/2
$$

C.1-2

$2^{n+1}$

$2^{n+m}$

C.1-3

$$
n!/n = (n-1)!
$$

C.1-4

$$
\binom{49}{3} + \binom{50}{2} \binom{49}{1} = 78449
$$

C.1-5

$$
\binom{n}{k} = \frac{n(n-1)\cdots(n-k+1)}{k(k-1)\cdots1} = \frac{n}{k} \frac{(n-1)\cdots(n-k+1)}{(k-1)\cdots1} = \frac{n}{k} \binom{n-1}{k-1}
$$

C.1-6

$$
\binom{n}{k} = \frac{n(n-1)\cdots(n-k+1)}{k(k-1)\cdots1} = \frac{n}{n-k} \frac{(n-1)\cdots(n-k)}{k\cdots1} = \frac{n}{n-k} \binom{n-1}{k}
$$

C.1-7

When the distinguished object is chosen, then we need to choose $k-1$ from $n-1$ objects. When it is not chosen, then we need to choose $k$ from $n-1$ objects. So we prove it.

C.1-8

1
1,1
1,2,1
1,3,3,1
1,4,6,4,1
1,5,10,10,5,1
1,6,15,20,15,6,1

C.1-9

$$
\sum_{i=1}^n i = (n+1)n/2 = \binom{n+1}{2}
$$

C.1-10

$$
\binom{n}{k+1} = \frac{n-k}{k+1} \binom{n}{k}
$$

When $k$ become larger, $\frac{n-k}{k+1}$ first larger than 1 then smaller than 1.

Let $\frac{n-k}{k+1} \le 1$, we get $k \ge (n-1)/2$. So when $n$ is even, it achieves maximum at $\lfloor n/2 \rfloor$, when $n$ is odd, it achieves maximum at $\lfloor n/2 \rfloor$ or $\lceil n/2 \rceil$.

C.1-11

$$
\binom{n}{j+k} = \frac{n(n-1)\cdots(n-j-k+1)}{(j+k)(j+k-1)\cdots1} \le \frac{n(n-1)\cdots(n-j+1)*(n-j)\cdots(n-j-k+1)}{j(j-1)\cdots1*k(k-1)\cdots1} = \binom{n}{j} \binom{n-j}{k}
$$

C.1-12




## Probability

## Discrete random variables

## The geometric and binomial distributions

## The tails of the binomial distributions

## Problems

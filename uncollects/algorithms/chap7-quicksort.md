# Quicksort

## Description of quicksort

```c
QUICKSORT(A,p,r)
    if p < r
        q = PARTITION(A,p,r)
        QUICKSORT(A,p,q-1)
        QUICKSORT(A,q+1,r)

PARTITION(A,p,r)
    x = A[r]
    i = p - 1
    for j = p to r-1
        if A[j] \le x
            i = i + 1
            exchange A[i] with A[j]
    exchange A[i + 1] with A[r]
    return i + 1
```

### Exercises

7.1-1

13,19,9,5,12,8,7,4,21,2,6,11
13,19,9,5,12,8,7,4,21,2,6,11
9,19,13,5,12,8,7,4,21,2,6,11
9,5,13,19,12,8,7,4,21,2,6,11
9,5,13,19,12,8,7,4,21,2,6,11
9,5,8,19,12,13,7,4,21,2,6,11
9,5,8,7,12,13,19,4,21,2,6,11
9,5,8,7,4,13,19,12,21,2,6,11
9,5,8,7,4,13,19,12,21,2,6,11
9,5,8,7,4,2,19,12,21,13,6,11
9,5,8,7,4,2,6,12,21,13,19,11
9,5,8,7,4,2,6,11,21,13,19,12

7.1-2

$q = r$

```c
PARTITION(A, p, r)
    x = A[r]
    c = 0
    for j = p to r-1
        if A[j] = x
            c = c + 1
    i = p - 1
    for j = p to r-1
        if A[j] \le x
            i = i + 1
            exchange A[i] with A[j]
    exchange A[i + 1] with A[r]
    return i + 1 - \lfloor c/2 \rfloor
```

7.1-3

All other lines cost constant time except for line3, which always take $\Theta(n-1)=\Theta(n)$ time.

7.1-4

```c
DECREASING-PARTITION(A,p,r)
    x = A[r]
    i = p - 1
    for j = p to r-1
        if A[j] \ge x
            i = i + 1
            exchange A[i] with A[j]
    exchange A[i + 1] with A[r]
    return i + 1
```

## Performance of quicksort

7.2-1

$$
\begin{align*}
T(n-1) + \Theta(n) &= \Theta((n-1)^2) + \Theta(n) \\
&= \Theta(n^2) -2\Theta(n) + 1 + \Theta(n) \\
&= \Theta(n^2)
\end{align*}
$$

7.2-2

$n^2$, as the split element is the last one, it's the worst case we see in 7.1-2.

7.2-3

We get the recurrence $T(n) = T(n-1) + \Theta(n)$, and by 7.2-1, we get $T(n) = n^2$.

7.2-4

The more sorted the array is, the less work INSERTION-SORT need to do, which is close to $T(n)$. However, the more empty partition the QUICKSORT to produce, which is close to $T(n^2)$.

7.2-5

The minimum depth is the most left leaf:

From $\alpha^h n=1$, we get $h=\log_{1/\alpha} n$.

The maximum depth is the most right leaf:

From $\beta^h n=1$, we get $h=\log_{1/\beta} n$.

7.2-6

Because all permutations of the elements are equally likely. The split balance is decided by the pivot element. If the pivot element is smaller than $\alpha n$ or larger than $(1-\alpha)n$, it is unbalanced. Both the probability is $\alpha$, and the balanced probability is $1-2\alpha$.

## A randomized version of quicksort

7.3-1

It represents the more typical time cost, the worst-case probability is very low here. which is $1/n!$.

7.3-2

The number of calls to RANDOM is same to the number of calls to RANDOMIZED-PARTITION.

In the worst case, $T(n) = T(n-1) + 1$, we get $T(n)$.

In the best case, we get $T(n) = 2T(n/2) + 1$, we get $T(n)$.

## Analysis of quicksort

7.4-1

We guess $T(n) \ge cn^2 + cn$, we get:

$$
\begin{align*}
T(n) &= \max \{T(q) + T(n-q-1): 0 \le q \le n-1\} + \Theta(n) \\
&\ge \max \{cq^2+cq + c(n-q-1)^2+c(n-q-1)\} + \Theta(n) \\
&= c(n-1)^2+c(n-1) + \Theta(n) \\
&= cn^2+cn + \Theta(n) - 2cn \\
&\ge cn^2+cn \text{ when } c \le c_2/2
\end{align*}
$$

7.4.2

It's equal to prove that $T(n) = \Omega(n\lg n)$ for:

$$
T(n) = \min_{1\le q \le n-1} \{T(q) + T(n-q-1): 0 \le q \le n-1\} + \Theta(n)
$$

7.4.3

$$
\begin{align*}
q^2 + (n-q-1)^2 &= 2q^2 -2(n-1)q + (n-1)^2 \\
&= 2(q - \frac{n-1}{2})^2 + \frac{(n-1)^2}{2}
\end{align*}
$$

so when $q=0$ or $q=n-1$, it achieves its maximum value.

7.4.4

$$
\begin{align*}
E[X] &= \sum_{i=1}^{n-1} \sum_{k=1}^{n-i} \frac{2}{k+1} \\
&> \sum_{i=1}^{n-1} \sum_{k=1}^{n-i} \frac{1}{k} \\
&\ge \sum_{i=1}^{n-1} \ln(n-i) \\
&= \ln((n-1)!) \\
&= \lg((n-1)!) / \lg e \\
&= \Omega((n-1)\lg(n-1)) \\
&= \Omega(n\lg n)
\end{align*}
$$

7.4-5

The recursive tree depth is $\lg(n/k)$, The partition running time is $O(n\lg(n/k))$, the insert running time is $O(k^2)*\frac{n}{k} = O(nk)$, so the total running time is $O(nk + n\lg(n/k))$.

We want to have:

$$
c_i nk + c_q n \lg(n/k) \le c_q n \lg n
$$

So we get $c_i k \le c_q \lg k$.

In practice, we need test to determine it.

7.4-6

The probability is that at least two number is in $[0, \alpha]$ or $[1-\alpha,1]$.

So we have the total probability:

$$
Pr\{A\} = 2(\alpha^3 + \binom{3}{1}\alpha^2(1-\alpha)) = (6\alpha^2 - 4\alpha^3)
$$

## Problems


# Dynamic Programming

## Rod cutting

14.1-1

First, we have $T(0) = 1 = 2^0$.

If for any $j < n$, we have $T(j) = 2^j$, then we have:

$$
T(n) = 1 + \sum_{j=0}^{n-1} T(j) = 1 + \sum_{j=0}^{n-1} 2^j = 2^n
$$

14.1-2

Using the greedy strategy, for $r_4$, we need first cut 3, so $4 = 3 + 1$. However, the value is small than $4 = 2 + 2$.

14.1-3

```c
BOTTOM-UP-COST-CUT-ROD(p,n)
let r[0:n] be a new array
r[0] = 0
for j = 1 to n
    q = p[j]
    for i = 1 to j-1
        q = max{q, p[i] + r[j-i] - c}
    r[j] = q
return r[n]
```

14.1-4

```c
CUT-ROD(p,n)
if n == 0
    return 0
if n == 1
    return p[1]
q = -\infty
for i = 1 to \lfloor n/2 \rfloor
    q = max {q, CUT-ROD(p,i) + CUT-ROD(p,n-i)}
return q

MEMOIZED-CUT-ROD-AUX(p,n,r)
if r[n] \ge 0
    return r[p]
if n == 0
    q == 0
elseif n == 1
    r[1] = p[1]
    return p[1]
else q = -\infty
    for i = 1 to \lfloor n/2 \rfloor
        q = max {q, MEMOIZED-CUT-ROD-AUX(p,i,r) + MEMOIZED-CUT-ROD-AUX(p,n-i,r)}
r[n] = q
return q
```

It doesn't effect running time.

14.1-5

```
MEMOIZED-CUT-ROD(p,n)
let r[0:n] and s[1:n] be new arrays
for i = 0 to n
    r[i] = -\infty
return MEMOIZED-CUT-ROD-AUX(p,n,r)

MEMOIZED-CUT-ROD-AUX(p,n,r)
if r[n] \ge 0
    return r[n]
if n == 0
    q = 0
else q = -\infty
    for i = 1 to n
        if q < p[i] + MEMOIZED-CUT-ROD-AUX(p, n-i, r)
            q = p[i] + MEMOIZED-CUT-ROD-AUX(p, n-i, r)
            s[n] = i
    r[n] = q
return q
```

14.1-6

```c
FIBONACCI(n)
let r[0:n] be a new array
r[0] = 0
r[1] = 1
for i = 2 to n
    r[i] = r[i-2] + r[i-1]

return r[n]
```

It have $n+1$ vertices and $2(n-1)$ edges.

## Matrix-chain multiplication

14.2-1

2010
1655 1950
405 2430 1770
330 330 930 1860
150 360 180 3000 1500
0 0 0 0 0 0

2
4 2
2 2 4
2 2 4 4
1 2 3 4 5

So the best way is $(A_1A_2)((A_3A_4)(A_5A_6))$.

14.2-2

```c
MATRIX-CHAIN(A,n)
let m[1:n,1:n] and s[1:n-1,2:n] be new tables
for i = 1 to n
    for j = 1 to n
        m[i,j] = -\infty

MATRIX-CHAIN-MULTIPLY(A,s,i,j)
if m[i,j] >= 0
    return m[i,j]
if i == j
    m[i,j] = 0
else m[i,j] = \infty
    for k = i to j - 1
        q = MATRIX-CHAIN-MULTIPLY(A,s,i,k) + MATRIX-CHAIN-MULTIPLY(A,s,k+1,j) + p_{i-1}p_kp_j
        if q < m[i,j]
            m[i,j] = q
            s[i,j] = k

return m[i,j]

```

14.2-3

When $n=1$, we have $P(n) = 1 = \Omega(2^1)$

When $n=2$, we have:

$$
P(n) = \sum_{k=1}^{n-1} P(k)P(n-k) \ge \sum_{k=1}^{n-1}c_kc_{n-k} 2^n = \Omega(2^n)
$$

14.2-4

It has $n(n+1)/2$ vertices, It has $(n^3-n)/3$ edges. Every node $A_{i:j}$ has $2(j-i)$ edges.

14.2-5

It's the same as the edges number in 14.2-4. We have:

$$
\sum_{i=1}^n 2(i-1)(n-i+1) = 2\sum_{i=0}^{n-1}i(n-i) = \frac{n^3-n}{3}
$$

14.2-6

Every pair of parentheses reduce one dimension of matrixs. For $n$ matrixs, there are $n+1$ dimensions, and finally become $2$ dimensions. So it need $n-1$ pairs of parentheses.

## Elements of dynamic programming

14.3-1

Recursive problem can omit some unnessesary computation in subproblems. So it is a more efficient way. We can prove it.

The first method's running time is $\Omega(4^n/n^{3/2})$ as stated in 14.2. For latter, we have the recurrence $T(n) \le 2\sum_{i=1}^{n-1}T(i) + cn$. By substitution, we have $T(n) = O(n3^{n-1})$, which is smaller than $\Omega(4^n/n^{3/2})$.

14.3-2

The subproblems are not overlap.

14.3-3

Yes, similary to the minimization problem, it has optimal substructure.

14.3-4

Use the 14.2-1 example, the best way to to divide $m[1:5]$ is $m[1:4,5:5]$, if use the greedy strategy, it will become $m[1:2,3:5]$.

14.3-5

The subproblem must be independent. But the total num limits of rods make the subproblems not independent.

## Longest common subsequence

14.4-1

6

14.4-2

```c
PRINT-LCS(c,X,i,j)
if i == 0 or j == 0
    return
if x_i == y_i
    PRINT-LCS(c,X,i-1,j-1)
    print x_i
elseif c[i-1,j] \ge c[i,j-1]
    PRINT-LCS(c,X,i-1,j)
else PRINT-LCS(c,X,i,j-1)
```

14.4-3

```c
MEMORIZED-LCS-LENGTH(X,Y,m,n)
let b[1:m,1:n] and c[0:m,0:n] be new tables
for i = 0 to m
    for j = 0 to n
        c[i,j] = -\infty

return MEMORIZED-LCS-LENGTH-AUX(X,Y,m,n)

MEMORIZED-LCS-LENGTH-AUX(X,Y,i,j)
if c[i,j] >= 0
    return c[i,j]

if i == 0 or j == 0
    c[i,j] = 0
elseif x_i == y_j
    c[i,j] = MEMORIZED-LCS-LENGTH-AUX(X,Y,i-1,j-1) + 1
    b[i,j] = "\\"
elseif MEMORIZED-LCS-LENGTH-AUX(X,Y,i-1,j) \ge MEMORIZED-LCS-LENGTH-AUX(X,Y,i,j-1)
    c[i,j] = MEMORIZED-LCS-LENGTH-AUX(X,Y,i-1,j)
    b[i,j] = "|"
else c[i,j] = MEMORIZED-LCS-LENGTH-AUX(X,Y,i,j-1)
     b[i,j] = "-"

return c[i,j]
```

14.4-4

```c
LCS-LENGTH(X,Y,m,n)
if n > m
    return LCS-LENGTH(Y,X,n,m)

let p[0:n], c[0:n] be new lists

for j = 0 to n
    p[j] = 0

for i = 1 to m
    c[0] = 0
    for j = 1 to n
        if x_i == y_j
            c[j] = p[j-1] + 1
        elseif p[j] \ge c[j-1]
            c[j] = p[j]
        else c[j] = c[j-1]
    
    for j = 0 to n
        p[j] = c[j]

```

```c
LCS-LENGTH(X,Y,m,n)
if n > m
    return LCS-LENGTH(Y,X,n,m)

let c[0:n] be new list

for j = 0 to n
    c[j] = 0

for i = 1 to m
    p = 0
    for j = 1 to n
        if x_i == y_j
            q = c[j-1] + 1
        elseif c[j] \ge p
            q = c[j]
        else q = p
        c[j-1] = p
        p = q
    c[n] = p
```

14.4-5

```c
LIS-LENGTH(X,n)
let c[1:n], b[1:n] be new list
best_index = 0
longest_len = 0
for i = 1 to n
    b[i] = 0
    c[i] = 1
    for j = 1 to i - 1
        if x_i > x_j and c[j] + 1 > c[i]
            c[i] = c[j] + 1
            b[i] = j

    if c[i] > longest_len
        longest_len = c[i]
        best_index = i

return best_index

PRINT-LIS(X,c,b,i)
if b[i] == 0
    return

PRINT-LIS(X,c,b,b[i])
print c[i]
```

14.4-6

```c
LIS-LENGTH(X,n)
let d[1:n] be new list
longest_len = 1
d[1] = x_1
for i = 2 to n
    if x_i > d[longest_len]
        longest_len = longest_len + 1
        d[longest_len] = x_i
    else
        binary-search the first k that d[k] \ge x_i
        d[k] = x_i

for i = 1 to longest_len
    print d[i]
```

## Optimal binary search trees

14.5-1

```c
CONSTRUCT-OPTIMAL-BST(root,n)

PRINT-BST(root,i,j)
if i > j
    print d\{j\}
else r = root[i,j]
    print k\{r\}
    PRINT-BST(root,i,r-1)
    PRINT-BST(root,r+1,j)
```

14.5-2

e

3.12
2.44 2.61
1.83 1.96 2.13
1.34 1.41 .1.48 1.55
1.02 0.93 1.04 1.01 1.2
0.62 0.68 0.57 0.57 0.72 0.78
0.28 0.30 0.32 0.24 0.30 0.32 0.34
0.06 0.06 0.06 0.06 0.05 0.05 0.05 0.05

w

1.0
0.81 0.90
0.64 0.71 0.78
0.49 0.54 0.59 0.64
0.42 0.39 0.42 0.45 0.56
0.28 0.32 0.27 0.28 0.37 0.41
0.16 0.18 0.20 0.13 0.20 0.22 0.24
0.06 0.06 0.06 0.06 0.05 0.05 0.05 0.05

r
5
3 5
3 5 5
2 3 5 6
2 3 4/5 5 6
1 2 4 4 5 6
1 2 3 4 5 6 7

14.5-3

It would not change the asymptotic running time. Because the r loop exists.

14.5-4

```c
OPTIMAL-BST(p,q,n)
let e[1:n+1,0:n], w[1:n+1,0:n], and root[1:n,1:n] be new tables
for i = 1 to n + 1
    e[i,i-1] = q_{i-1}
    w[i,i-1] = q_{i-1}
for l = 1 to n
    for i = 1 to n - l + 1
        j = i + l - 1
        e[i,j] = \infty
        w[i,j] = w[i,j-1] + p_j + q_j
        for r = e[i,j-1] to e[i+1,j]
            t = e[i,r-1] + e[r+1,j] + w[i,j]
            if t < e[i,j]
                e[i,j] = t
                root[i,j] = r
return e and root
```


## Problems

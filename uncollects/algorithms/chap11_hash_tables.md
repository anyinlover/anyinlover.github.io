# Hash Tables

## Direct-address tables

11.1-1

```c
DIRECT-ADDRESS-TABLE-MAXIMUM(S)
i = m - 1
while T[i] == NIL
    i = i - 1
return T[i]
```

The worst-case running time is $O(m)$.

11.1-2

We can use index as key, 1 represent exist, 0 represent not exist.

11.1-3

Every slot has a double link list to resolve the collison, so insert and delete both take $O(1)$. And search return the full list.

11.1-4

```c
Initialize an array S that $S.size = n$ and $top=0$.

SEARCH(T,k)
x = T[k]
if 1 \le x.skey \le top and S[x.skey] == k
    return x
else return NIL

INSERT(T,x)
if SEARCH(T,S,x.key) == NIL
    top = top + 1
    x.skey = top
    S[top] = x.key
T[x.key] = x

DELETE(T,x)
skey = x.skey
S[skey] = S[top]
top = top - 1
T[x.key] = NIL
```

## Hash tables

11.2-1

This problem is same to birthday paradox.

$$
E[X] = \binom{n}{2} \frac{1}{m} = \frac{n(n-1)}{2m}
$$

11.2-2

\
28->19->10
20
12
\
5
15->33
\
17

11.2-3

In a successful search, we can use the method similary to the book. In this time, let $x_i$ denote the $i$ th smallest element. the following analysis is just same. So we can get running time $\Theta(2 + \alpha/2 - \alpha/2n) = \Theta(1+\alpha)$.

In un unsuccessful search, the expected time to search a key $k$ is the expected time to search until the element that larger than the key. Because $k$ is random, so the expected search elements is $\alpha/2+1$, and the total running time is faster, but still holds: $\Theta(1+\alpha/2+1) = \Theta(1+\alpha)$.

The insert time need search first, and it's like the unsuccessful search, it takes $\Theta(1+\alpha/2+1) = \Theta(1+\alpha)$.

The delete time is still $\Theta(1)$.

11.2-4

Flag show if the slot is empty. It need be a doubly linked list to delete in $O(1)$ expected time.

11.2-5

If all slots store less $n$ keys, then $|U| <= (n-1)m$, which is not true. so In the worst-case, if the $n$ keys belong to the slot, then the running time is $\Theta(n)$.

11.2-6

We first randomly choose a slot, the probability is $1/m$, then in this slot, we randomly choose a number $1\le i \le L$, the probaiblity is $1/L$. If i > n_k$, then we repeat again. In this way, any element is chosen by $1/mL$, and the probability that a element can be choose in one time is $n/mL = \alpha/L$. It's a geographic distribution, it take $L/\alpha$ times to success. Every time we need $\Theta(1+\alpha), So the total time is $O((1+\alpha) L / \alpha) = O(L(1+1/\alpha))$.

## Hash functions

11.3-1

First compare $h(k)$, compare $k$ only when $h(x)$ is same.

11.3-2

```c
HASH(s,r,m)
val = 0
for i = 1 to r
    val = (val * 128 + s[i]) mod m
return val
```

11.3-3

$$
h(k) = (\sum_{i=1}^r s_i(2^p)^{r-1}) \mod (2^p - 1) = \sum_{i=1}^r s_i
$$

So by permuting its characters, the answer is same.

If we use the phone number as key, then it is bad that two permuting numbers have same key.

11.3-4

$$
h(61) = 700 \\
h(62) = 318 \\
h(63) = 936 \\
h(64) = 554 \\
h(65) = 172
$$

11.3-5

When elements spread uniformly, the total collison numbers is less and $\epsilon$ is smallest. So we have:

$$
\epsilon \ge \frac{|Q| \binom{\frac{|U|}{|Q|}}{2}}{\binom{|U|}{2}} = \frac{\frac{|U|}{|Q|}-1}{|U|-1} > \frac{\frac{|U|}{|Q|}-1}{|U|} = \frac{1}{|Q|} - \frac{1}{|U|}
$$

11.3-6


## Open addressing

11.4-1

22,88,\,\,4,15,28,17,59,31,10
22,\,59,17,4,15,28,88,\,31,10

11.4-2

```c
HASH-DELETE(T,k)
i = 0
repeat
    q = h(k,i)
    if T[q] == k
        T[q] = DELETED
        return q
    else i = i + 1
until i == m
error "not found"

HASH-INSERT(T,k)
i = 0
repeat
    q = h(k,i)
    if T[q] == NIL or T[q] == DELETED
        T[q] = k
        return q
    else i = i + 1
until i == m
error "hash table overflow"

HASH-SEARCH(T,k)
i = 0
repeat
    q = h(k,i)
    if T[q] == k
        return q
    i = i + 1
until T[q] == NIL or i == m
return NIL
```

11.4-3

When $\alpha = 3/4$,

$$ E[X_u] = \frac{1}{1-\alpha} = 4 $$

$$ E[X_s] = \frac{1}{\alpha} \ln \frac{1}{1-\alpha} = 1.85 $$

When $\alpha = 7/8$,

$$ E[X_u] = \frac{1}{1-\alpha} = 8 $$

$$ E[X_s] =\frac{1}{\alpha} \ln \frac{1}{1-\alpha} = 2.38 $$

11.4-4

The analysis is same when $\alpha < 1$. We have:

$$
\frac{1}{m} \sum_{i=0}^{m-1} \frac{m}{m-i} = \sum_{i=0}^{m-1} \frac{1}{m-i} = \sum_{k=1}^m \frac{1}{k} = H_k
$$

11.4-5

Because $h(k,i) = (h_1(k) + i h_2(k)) \mod m$. If we let $m = m'd, h_2(k) = h'_2(k)d$, then we have:

$$
h(k,i+m') - h(k,i) = m' h_2(k) \mod m = m h'_2(k) \mod m = 0
$$

So for every $i$, there are $\frac{m}{m'} = d$ other $i$ have the same hash key. So it examines $1/d$ th of the hash table.

11.4-6

We have the equation:

$$
\frac{1}{1-\alpha} - \frac{2}{\alpha} \ln \frac{1}{1-\alpha} = 0
$$

Solve it by scipy fsolve, we get $\alpha = 0.715$.

## Practical considerations

11.5-1


## Problems

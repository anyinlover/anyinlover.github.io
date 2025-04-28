# Amortized Analysis

## Aggregate analysis

16.1-1

No, in worst case, everytime run MULTIPUSH, the amortized cost of $O(k)$.

16.1-2

First increase x until for every $i$, $A[i] = 1$. Then increase and decrease it repeatly, every time cost $\Theta(k)$, the total running time is $\Theta(nk)$

16.1-3

$$
\begin{align*}
T(n) &= n - \lfloor \lg n \rfloor - 1 + \sum_{k=0}^{\lfloor \lg n \rfloor} 2^k \\
&= n - \lfloor \lg n \rfloor - 2 + 2^{\lfloor \lg n \rfloor + 1} \\
\end{align*}
$$

So

$$
T(n) \le n - \lfloor \lg n \rfloor - 2 + 2^{\lg n + 1} = 3n - \lfloor \lg n \rfloor - 2 = O(n)
$$

$$
T(n) \ge n - \lfloor \lg n \rfloor - 2 + 2^{\lg n} = 2n - \lfloor \lg n \rfloor - 2 = \Omega(n)
$$

Finally, we have $T(n) = \Theta(n)$.

## The accounting method

16.2-1

We know for a stack with $s$ elements, we can use two more stacks to copy it. First pop and push all elements to a stack, then pop and push to another stack again. So the total running time is $\Theta(4s)$.

We can assgin PUSH $5, POP 0, COPY 0, then the credit is always nonnegative. So the total running time os $O(5n) = O(n)$.

16.2-2

When $i$ is an exact power of 2, assgin $2i$, otherwise 0. Because $2^{k+1} - 2^k = 2^k$. So the credit will not be negative. The total running time is:

$$
T(n) \le 2\sum_{k=0}^{\lfloor \lg n \rfloor} 2^k = 2(2^{lfloor \lg n \rfloor + 1} - 1) = O(n)
$$

16.2-3

```c
Counter(A,k)
for i = 0 to k - 1
    A[i] = 0
high = 0

INCREMENT(A,k)
i = 0
while i < k and A[i] == 1
    A[i] = 0
    i = i + 1
if i < k
    A[i] = 1
    high = i
else
    high = 0

RESET(A,k)
for i = 0 to high
    A[i] = 0
```

## The potential method

16.3-1

Let $\varPhi'(D_i) = \varPhi(D_i) - \varPhi(D_0)$, because $\varPhi(D_i) \ge \varPhi(D_0)$, so we have $\varPhi'(D_i) \ge 0$.

$$
\hat{c'_i} = c_i + \varPhi'(D_i) - \varPhi'(D_{i-1}) = c_i + \varPhi(D_i) - \varPhi(D_{i-1}) = \hat{c_i}
$$

16.3-2

Let $\varPhi(D_i) = -2^{\lfloor \lg i \rfloor + 1} + 2i$, then we have $T(n) < 3n = O(n)$.

16.3-3

Let $\varPhi$ be $\varPhi(n)$, here $n$ is the item numbers in the heap. For Insert, we have:

$O(\lg n) = \hat{c_i} = O(\lg n) + \varPhi(n+1) - \varPhi(n)$

For EXTRAC-MIN, we have:

$0 = \hat{c_i} = O(\lg n) + \varPhi(n-1) - \varPhi(n)$

So let $\varPhi(n) - \varPhi(n-1) = O(\lg n) \le c \lg n $.

We have $\varPhi(n+1) - \varPhi(n) = O(\lg(n+1)) = O(\lg n)$ too.

Sum it, we have $\varPhi(n) = \sum_{i=1}^n \lg i$


16.3-4

$$
\sum_{i-1}^n c_i = \sum_{i=1}^n\hat{c_i} - \varPhi(D_n) + \varPhi(D_0) \le \sum_{i=1}^n 2 - s_n + s_0 = 2n - s_n + s_o
$$

16.3-5

```c
ENQUEUE(Q,x)
PUSH(S1,x)

DEQUEUE(Q)
if S2 == 0
    while S1 != 0
        PUSH(S2,POP(S1))
POP(S2)
```

We define the potential function $\varPhi$ to be twice of the number of objects in the stack $S1$.

So for ENQUEUE operation

$$
\hat{c_i} = c_i + \varPhi(D_i) - \varPhi(D_{i-1}) = 1 + 2 = 3
$$

For DEQUEUE operation when $S2$ is empty

$$
\hat{c_i} = c_i + \varPhi(D_i) - \varPhi(D_{i-1}) = 2k + 1 - 2k = 1
$$

For DEQUEUE operation when $S2$ is not empty

$$
\hat{c_i} = c_i + \varPhi(D_i) - \varPhi(D_{i-1}) = 1
$$

The amortized cost of operation is $O(1)$.

16.3-6

```c
Initialization(l)
let S[1:l] be a new array
S.size = 0
S.len = l

INSERT(S,x)
if S.size == S.len
    let S'[1:2l] ne a new array
    for i = 1 to l
        S'[i] = S[i]
    S'.len = 2l
    S'.size = S.size
    S = S'
S.size = S.size + 1
S[S.size] = x

DELETE-LARGER-HALF(S,x)
RANDOMIZED-SELECT(S,1,S.size, \lfloor S.size/2 \rfloor)
S.size = \lfloor S.size/2 \rfloor

PRINT(S)
for i = 1 to S.size
    print S[i]
```

## Dynamic tables

16.4-1

$\varPhi_0 = 0$

$\varPhi_1 = 2(num_1 - size_1/2) = 1$

$$
\hat{c_1} = c_1 + \varPhi_1 - \varPhi_0 = 2
$$

16.4-2

The expected insert time is $O(1/{1-\alpha})$. So when $\alpha$ is too large like 0.99, the hash table is too slow. We can resize the hash table when $\alpha$ reaches a certain threshold. It need resizes and rehashes all elements into the new hash table.

When the insertion trigger the resizing and rehashing, the running time can be large.

16.4-3

When it insert, charge it 3, when it delete, charge it 0, for it has 1 unit on itself. If it need constract, every element has 1 unit for moving. 

16.4-4

If no expansion or constraction happens, then $\hat{c_i} = c_i + \Delta \varPhi_i$ = 3 or -1$.

When it has an expension, we have

$$
\hat{c_i} = c_i + \Delta \varPhi_i = i + 2 - (i-1) = 3
$$

When it has a constraction, we have

$$
\hat{c_i} = size_{i-1}/3 + (2 - size_{i-1}/3) = 2
$$

## Problems

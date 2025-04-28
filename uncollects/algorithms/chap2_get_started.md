# Get Started

## Insertion sort

Sorting problem is described as following:

**Input**: A sequence of n numbers $<a_1, a_2, \cdots, a_n>$
**Output**: A permutation (reordering) $<a'_1, a'_2, \cdots, a'_n>$ of the input sequence such that $a'_1 \le a'_2 \le \cdots \le a'_n $

The numbers to be sorted are also known as the keys. A key is often associated with satellite data, they together form a record.

Insertion sort is one of the algorithms to solve the sorting problem. It is efficient for sorting a small number of elements. It works the way we might sort a hand of playing cards. Following is the pseudocode:

```c
INSERTION-SORT(A, n)
for i = 2 to n
    key = A[i]
    // Insert A[i] into the sorted subarray A[1:i-1]
    j = i - 1
    while j > 0 and A[j] > key
        A[j + 1] = A[j]
        j = j - 1
    A[j + 1] = key
```

In every loop, the subarray $A[1:i-1]$ consists of sorted elements originally in $A[1:i-1]$,  we state it as loop invariant which have following properties.

**Initialization**: It is ture prior to the first iteration of the loop.

**Maintenance**: If it is true before an iteration of the loop, it remains true before the next iteration.

**Termination**: The loop terminates, and when it terminates, the invariant -- usually along with the reason that the loop terminated -- gives us a useful property that helps show that the algorithm is correct.

Using the method of loop invariant, we can prove insertion-sort algorithm is correct.

### Exercises

2.1-1

**31**, 41, 59, 26, 41, 58

**31**, **41**, 59, 26, 41, 58

**31**, **41**, **59**, 26, 41, 58

**26**, **31**, **41**, **59**, 41, 58

**26**, **31**, **41**, **41**, **59**, 58

**26**, **31**, **41**, **41**, **58**, **59**

2.1-2

Initialization: sum is 0, which mean no element has been added.

Maintenance: sum is the sum of $A[1: i-1]$, add the i element, become the sum of $A[1:i]$.

Termination: when loop terminates, i is n+1, and sum is the sum of $A[1:n]$, so it is correct.

2.1-3

```c
DECREASING-INSERTION-SORT(A, n)
for i = 2 to n
    key = A[i]
    // Insert A[i] into the sorted subarray A[1:i-1]
    j = i - 1
    while j > 0 and A[j] < key
        A[j + 1] = A[j]
        j = j - 1
    A[j + 1] = key
```

2.1-4

```c
LINEAR-SEARCH(A, n, x)
for i = 1 to n
    if A[i] == x:
        return i

return NIL
```

Initialization: no x is found.

Maintenance: If $A[1:i-1]$ has no x, and A[i] is not equal x, then $A[1:i]$ has no x.

Termination: When x is found or i is n+1, the loop terminates. if x is found, we return i, it is the index we need, if i is n+1, $A[1:i]$ has no x, we return NIL.

2.1-5

```c
ADD-BINARY-INTEGERS(A, B, C, n)
// maintain the carry bit
carry = 0
for i = 0 to n - 1
    C[i] = A[i] + B[i] + carry
    if C[i] >= 2:
        C[i] = C[i] - 2
        carry = 1
    else:
        carry = 0
C[n] = carry
```

## Analyzing algorithms

Analyzing an algorithm has come to mean predicting the resoures that the algorithm requires. Memory, communication bandwidth or energy consumption may be considered. But most often, we want to measure computational time.

We need a model of the technology that an algorithm runs on, most time we assumes a generic one-processor, random-access machine (RAM) model. In the RAM model, instructions execute one after another, with no concurrent operations. The model assumes that each instruction takes the same amount of time as any other instruction and that each data access takes the same amount of time as any other data access. The model has limited instructions and not account for the memory hierarchy.

The running time can depend on many features of the input, we'll focus on the one that has been shown to have the greatest effect, namely the size of the its input.

Please note that the best notion for input size depends on the problem being studied. In sorting problems, it's the number of items in the input, in multiplying two integers, it's the total number of bits needed to represent the input in binary notation. In graph problem, it's the number of vertices and the number of edges.

The running time of an algorithm on a particular input is the number of instructions and data accesses executed.

```c
INSERTION-SORT(A, n)                                    cost    times
for i = 2 to n                                          c_1     n
    key = A[i]                                          c_2     n-1
    // Insert A[i] into the sorted subarray A[1:i-1]    0       n-1
    j = i - 1                                           c_4     n-1
    while j > 0 and A[j] > key                          c_5     \sum_{i=2}^n t_i
        A[j + 1] = A[j]                                 c_6     \sum_{i=2}^n (t_i - 1)
        j = j - 1                                       c_7     \sum_{i=2}^n (t_i - 1)
    A[j + 1] = key                                      c_8     n-1
```

$$
T(n) = c_1n + c_2(n-1) + c_4(n-1) + c_5 \sum_{i=2}^n t_i + c_6 \sum_{i=2}^n (t_i - 1) + c_7 \sum_{i=2}^n (t_i - 1) + c_8 (n-1)
$$

Here $t_i$ is the number of times the while loop test in line 5. And we can see that the running time may dependbon which input of that size is given.

The best case occurs when the array is already sorted. In this case, the running time is a linear function of n.

$$
\begin{align*}
T(n) &= c_1n + c_2(n-1) + c_4(n-1) + c_5 (n-1) + c_8 (n-1) \\
&= (c_1 + c_2 + c_4 + c_5 + c_8) n - (c_2 + c_4 + c_5 + c_8) = an + b
\end{align*}
$$

In the worst case, the array is reversed sorted. In this case, $t_i = i$, the running time is a quadratic function of n.

$$
\begin{align*}
T(n) &= c_1n + c_2(n-1) + c_4(n-1) + c_5 (\frac{n(n+1)}{2} - 1) + c_6 (\frac{n(n-1)}{2}) + c_7(\frac{n(n-1)}{2}) + c_8 (n-1) \\ &= (\frac{c_5 + c_6 + c_7}{2})n^2 + (c_1 + c_2 + c_4 + \frac{c_5 - c_6 - c_7}{2} + c_8) n - (c_2 + c_4 + c_5 + c_8) \\ &= an^2 + bn + c
\end{align*}
$$

In most time, we only care about worst-case running time. There are three reasons, the worst-case running time gives an upper bound on the running time for any input; For some algorithms, the worst case occurs fairly often; The average case is often roughly as bad as the worst case. In some particular cases, we use the technique of probabilistic analysis to analysis the average-case running time. 

It is the rate of growth or order of growth of the running time that really interests us. We therefore consider only the leading term of a formula, here is $n^2$, which notated as $\Theta(n^2)$.

2.2-1

$\Theta(n^3)$

2.2-2

```c
SELECTION-SORT(A, n)
for i = 1 to n-1
    for j = i+1 to n
        if A[j] < A[i]
            Exchange A[i] with A[j]
```

the last one has been exchanged.

worst-case running time is $\Theta(n^2)$, the best-case running time is same.

2.2-3

Average-case running time is $\Theta(n)$, for $1, \cdots, n$, the probability is $1/n$, so the average-case running time is:

$$
\sum_{i=1}^n i * \frac{1}{n} = \frac{n+1}{2} = \Theta(n)
$$

The worst-case running time is $\Theta(n)$ too, the worst case is the last element, so the worst running time is:

$$
n = \Theta(n)
$$

2.2-4

One skill is to check if the array is sorted, which take $\Theta(n)$ running time, if it is sorted, then no sorting need to be done.

## Designing algorithms

What if we want to design a new sorting algorithm? Insert sort uses the increnmental method, we can use another design method, known as divide-and-conquer.

Many useful algorithms are recursive in structure: to solve a given problem, they recurse one or more times to handle closely related subproblems. These algorithms typically follow the divide-and-conquer method:

**Divide** the problem into one or more subproblems that are smaller instances of the same problem.

**Conquer** the subproblems by solving them recursively.

**Combine** the subproblem solutions to form a solution to the original problem.

The merge sort algorithm closely follows the divide-and-conquer method. The key operation of the merge sort algorithm occurs in the combine step, which merges two adjacent, sorted subarrays.

```c
MERGE(A, p, q, r)
n_L = q - p + 1         // length of A[p:q]
n_R = r - q             // length of A[q+1:r]
let L[0:n_L - 1] and R[0:n_R - 1] be new arrays
for i = 0 to n_L - 1    // copy A[p:q] into L[0:n_L - 1]
    L[i] = A[p+i]
for j = 0 to n_R - 1    // copy A[+1:r] into R[0:n_R - 1]
    R[j] = A[q + j + 1]
i = 0                   // i indexes the smallest remaining element in L
j = 0                   // j indexes the smallest remaining element in R
k = p                   // k indexes the location in A to fill
// As long as each of the arrays L and R contains an unmerged element,
//      copy the smallest unmerged element back into A[p:r].
while i < n_L and j < n_R
    if L[i] <= R[j]
        A[k] = L[i]
        i = i + 1
    else A[k] = R[j]
        j = j + 1
    k = k + 1

// Having gone through one of L and R entirely, copy the
//      reminder of the other to the end of A[p:r]
while i < n_L
    A[k] = L[i]
    i = i + 1
    k = k + 1
while j < n_R
    A[k] = R[j]
    j = j + 1
    k = k + 1
```

The MERGE procedure runs in $\Theta(n)$ time, where $ n = r - p + 1$.

Following is the MERGE-SORT algorithm:

```c
MERGE-SORT(A, p ,r)         // zero or one element?
if p >= r
    return
q = [(p + r)/2]             // midpoint of A[p:r]
MERGE-SORT(A, p, q)         // recursively sort A[p:q]
MERGE-SORT(A, q+1, r)       // recursively sort A[q+1:r]
// Merge A[p:q] and A[q+1:r] into A[p:r]
MERGE(A, p, q, r)
```

When an algorithm contains a recursive call, we can often describe its running time by a recurrence equation. Let $T(n)$ be the worst-case running time on a problem of size $n$. If the problem size is small enough, say $ n < n_0$ for some constant $ n_0 > 0 $, the straightforward solution takes constant time $\Theta(1)$. Suppose that the division of the problem yields $a$ subproblems, each with size $n/b$, we get the recurrence equation:

$$
T(n) = \begin{cases}
    \Theta(1) &\text{if } n < n_0 \\
    D(n) + aT(n/b) + C(n) &\text{otherwise}
    \end{cases}
$$

We simplify here when $n/b$ isn't an integer. But it does not affect the order of growth.

In merge sort, we get the worst-case running time:

$$
T(n) = 2T(n/2) + \Theta(n)
$$

It can be solved that $T(n) = \Theta(n \lg n)$, which is better than insertion sort. If we draw a recurse tree, we can infer that the worst-case running time is $\Theta(n \lg n)$.

### Exercises

2.3-1

3, 41, 52, 26, 38, 57, 9, 49

3, 41, 52, 26 | 38, 57, 9, 49

3, 41 | 52, 26 | 38, 57 | 9, 49

3 | 41 | 52 | 26 | 38 | 57 | 9 | 49

3, 41 | 26, 52 | 38, 57 | 9, 49

3, 26, 41, 52 | 9, 38, 49, 57

3, 9, 26, 38, 41, 49, 52, 57

2.3-2

If $ r >= p $, than $ q = (p + r)/2 >= p $, so for merge-sort, there is always $ r >= p $.

2.3-3

Initialization: A[p:p-1] is empty
Maintenance: A[p:k-1] is the smallest sorted elements, find the next small one as A[k], A[p:k] remains the smallest sorted elements
Termination: A[p:k] is the smallest sorted elements, with $i = n_L$ or $j = n_R$.

copy the left elements into A, finally we get A[p:r] as sorted elements.

2.3-4

$$
T(2) = 2 = 2*\lg2
$$

$$
T(n) = 2T(n/2) + n = 2*(n/2 \lg(n/2)) + n = n \lg(n/2) + n = n(\lg n - 1) + n = n \lg n
$$

Based on mathematical induction, we get the solution.

2.3-5

```c
RECURSIVE-INSERTION-SORT(A, i)

if i = 1
    return

RECURSIVE-INSERTION-SORT(A, i-1)
key = A[i]
// Insert A[i] into the sorted subarray A[1:i-1]
j = i - 1
while j > 0 and A[j] > key
    A[j + 1] = A[j]
    j = j - 1
A[j + 1] = key
```

$ T(n) = T(n-1) + \Theta(n) $

2.3-6

```c
BINARY_SEARCH(A, n, x)
p = 1
r = n
while p <= r
    q = (p+r)/2
    if A[q] == x
        return q
    if A[q] > x
        r = q - 1
    else
        p = q + 1

return NIL
```

For every loop, $n = n/2$, in the worst-case, until $n=1$, so the worst-case running time is $\Theta(\lg n)$.

2.3-7

The worst-case running time is same, although the search time is $\Theta(\lg n)$, in the worst-case, the move time is still $\Theta(n)$. So the result is same.


2.3-8

```c
TWO_ELEMENTS_SUM_SEARCH(A, n, x)

MERGE_SORT(A, n)
for i = 1 to n - 1
    r = BINARY_SEARCH(x - A[i])
    if  r != NIL and r != i
        return true

return false
```

## Problems

2-1

a. 

$$
\Theta(k^2) * n / k = \Theta(nk)
$$

b.

```MERGE-SUBLISTS()```

We can maintain a minheap with n/k nodes and 1 heap_size, in this method, get the smallest one need $\Theta(1)$, maintain the minheap need $\Theta(\lg(n/k))$, so the all time is $\Theta(n \lg(n/k))$,

Another way to solve: regard it as a recursive merge process, it is like a tree, first merge two k length into 2k, then 4k, until n, the height of the tree is $\lg(n/k)$, at every level merge $n$ elements, so the time is $\Theta(n \lg(n/k))$.

c. 

$$
\Theta(nk + n \lg(n/k)) = \Theta(nk + n \lg n - n \lg k)
$$

$k <= \lg n$, otherwise, $\Theta(nk + n \lg n - n \lg k) > \Theta(n \lg n)$

d.

In practice, we want to make $c_1 nk + c_2 n \lg(n/k)$ smallest

$$
\begin{align*}
c_1 nk + c_2 n \lg(n/k)
&= c_2 n \lg n + n(c_1 k - c_2 \lg k)
\end{align*}
$$

So we just need find the $k = \argmin {c_1 k - c_2 \lg k}$

2.2

a. We need check that all number is in origin A. No new number is created.

b. 

Initialization:

$A[1:0]$ is empty smallest sorted array

Maintenance:

Before the loop, $A[1:i-1]$ is the smallest sorted array, Find the smallest one in the rest, bubble into the $A[i]$, so the $A[1:i]$ remain the smallest sorted array.

c.

Termination:

When it terminates, $ i = n $, $A[1:n]$ is the sorted array.

d.

$\Theta(n^2)$, the worst-case running time is same as INSERTION-SORT, but the best-case running time  $\Theta(n^2)$ is longer than INSERTION-SORT.

2.3

a. $\Theta(n)$

b.

```c
NAIVE-POLY-EVAL(A, n, x)
p = 0
for i = 0 to n
    tmp = A[i]
    while i > 0
        tmp = tmp * x
        i = i - 1
    p = p + tmp
```

$\Theta(n^2)$, larger than HORNER algorithm.

c.

Initialization: $ i = n $,

$$
p = \sum_{k=0}^{n-(i+1)} A[k + i + 1] x^k = \sum_{k=0}^{-1}A[k + n + 1] x^k = 0
$$

Maintenance: 

$$
p = A[i] + x*p = A[i] + x(\sum_{k=0}^{n-(i+1)} A[k + i + 1] x^k) = \sum_{k=0}^{n-i}A[k+i] x^k
$$

Termination:

When the loop terminates, $ i = 0 $, so the result is $ p = \sum_{k=0}^n A[k] x^k$.

2.4

a.

2, 1 | 3, 1 | 8, 6 | 8, 1 | 6, 1

b.

$n, n-1, \cdots, 1$

$n(n-1)/2$

c.

Same, In every insertion-sort loop step, inversions of $A[j]$ is the num of elements to move. 

d.

```c
INVERSIONS_MERGE(A, p, q, r)
n_L = q - p + 1         // length of A[p:q]
n_R = r - q             // length of A[q+1:r]
let L[0:n_L - 1] and R[0:n_R - 1] be new arrays
for i = 0 to n_L - 1    // copy A[p:q] into L[0:n_L - 1]
    L[i] = A[p+i]
for j = 0 to n_R - 1    // copy A[+1:r] into R[0:n_R - 1]
    R[j] = A[q + j + 1]
i = 0                   // i indexes the smallest remaining element in L
j = 0                   // j indexes the smallest remaining element in R
k = p                   // k indexes the location in A to fill
// As long as each of the arrays L and R contains an unmerged element,
//      copy the smallest unmerged element back into A[p:r].
ic = 0
while i < n_L and j < n_R
    if L[i] <= R[j]
        A[k] = L[i]
        i = i + 1
    else A[k] = R[j]
        j = j + 1
        ic = ic + 1
    k = k + 1

// Having gone through one of L and R entirely, copy the
//      reminder of the other to the end of A[p:r]
while i < n_L
    A[k] = L[i]
    i = i + 1
    k = k + 1
while j < n_R
    A[k] = R[j]
    j = j + 1
    k = k + 1
    ic = ic + 1

return ic

INVERSIONS_COUNT(A, p, r)
if p + 1 == r
    if A[p] < A[r]
        return 0
    else
        exchange A[p] with A[r]
        return 1
else if p == r
    return 0

q = [(p + r)/2]  
return INVERSIONS_COUNT(A, p, q) + INVERSIONS_COUNT(A, q+1, r) + INVERSIONS_MERGE(A, p, q, r)
```

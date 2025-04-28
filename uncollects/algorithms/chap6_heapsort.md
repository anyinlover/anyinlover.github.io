# HeapSort

Not only is the heap data structure useful for heapsort, but it also makes an efficient priority queue.

## Heaps

The heap data structure is an array that we can view as a nearly complete binary tree.

An array $A[1:n]$ represents a heap is an object with an attribute *A.heap_size*. heap_size can be smaller than the lenth of the array.

Given the index $i$ of a node, we can get the index of its parent, left and right child.

```c
PARENT(i)
    return \lfloor i/2 \rfloor

LEFT(i)
    return 2i

RIGHT(i)
    return 2i+1
```
There are two kinds of binary heaps: max-heaps and min-heaps. In a max-heap, the max-heap property is that for every node $i$ other than the root

$$ A[PARENT(i)] \ge A[i] $$

A min-heap is organized in opposite way, the min-heap property is that for every node $i$ other than the root

$$ A[PARENT(i)] \le A[i] $$

heap height:  the longest simple downward path from the node to a leaf.

### Exercises

6.1-1

minimum numbers of elements: $2^h$

maximum numbers of elements: $2^{h+1}-1$

6.1-2

Based on 6.1-1

$2^h \le n < 2^{h+1}$, we get $\lg n - 1 < h \le \lg n$. So we get $h = \lfloor \lg n \rfloor$.

6.1-3

If the largest value is not the root of the subtree, let it $i$, which mean $A[i] > A[PARENT(i)]$, it againsts the max-heap property. So the largest one must be the root of the subtree.

6.1-4

In all leafs. Based on 6.1-3, If a node has a child, then it must larger than the child. So it can't be the smallest one. Only the leafs might be the smallest element.

6.1-5

$[2:\min(k, \lfloor \lg n \rfloor + 1)]$

6.1-6

Yes, always satisfies $A[i/2] \le A[i]$

6.1-7

No, $15 = A[4] < A[9] = 16$.

6.1-8

We can use induction to solve it. We want to prove the inter nodes numbers is always $\lfloor n/2 \rfloor$.

In the base, We have only a root, and inter nodes numbers is $\lfloor 1/2 \rfloor = 0$.

When $n$ is odd, which mean the next node will be the left child (based on $2i$), if a node add a left child, it change from leaf node to a inter node, so we have $\lfloor n/2 \rfloor + 1 = \lfloor (n+1)/2 \rfloor$. When $n$ is even, which mean the next node will be the right child (based on $2i+1$), if a node add a right child, the inter node nums is same, so we have $\lfloor n/2 \rfloor = \lfloor (n+1)/2 \rfloor$.

Together, we prove that the inter node nums is always $\lfloor n/2 \rfloor$. So the leaf nodes begin from $\lfloor n/2 \rfloor + 1$ to $n$.

## Maintaining the heap property

MAX-HEAPIFY let the value at $A[i]$ float down in the max-heap so that the subtree rooted at the index $i$ obeys the max-heap property.

```c
MAX-HEAPIFY(A, i)
    l = LEFT(i)
    r = RIGHT(i)
    if l \le A.heap-size and A[l] > A[i]
        largest = l
    else largest = i
    if r \le A.heap-size and A[r] > A[largest]
        largest = r
    if largest \ne i
        exchange A[i] with A[largest]
        MAX-HEAPIFY(A, largest)
```

The children's subtrees each have size at most $2n/3$, so the running time is:

$$
T(n) \le T(2n/3) + \Theta(1)
$$

Based on master theorem, we have $T(n) = O(\lg(n))$.

Alternatively, the running time is $O(h)$.

### Exercises

6.2-1

[27,17,3,16,13,10,1,5,7,12,4,8,9,0]
[27,17,10,16,13,3,1,5,7,12,4,8,9,0]
[27,17,10,16,13,9,1,5,7,12,4,8,3,0]

6.2-2

When the left subtree is a complete binary tree and the right subtree is a complete binary tree with one less level. The left subtree have the highest ratio.

The left subtree nums is:

$$
\sum_0^i 2^i = 2^{i+1} - 1
$$

The right subtree nums is $2^i - 1$, so we have the ratio:

$$
\frac{2^{i+1} - 1}{2^{i+1} - 1 + 2^i - 1 + 1} = \frac{2^{i+1} - 1}{2^{i+1} + 2^i - 1} < \frac{2^{i+1}}{2^{i+1} + 2^i} = \frac{2}{3}
$$

$\alpha = 2/3$, no effect to the solution.

6.2-3

```c
MIN-HEAPIFY(A, i)
    l = LEFT(i)
    r = RIGHT(i)
    if l \le A.heap-size and A[l] < A[i]
        smallest = l
    else smallest = i
    if r \le A.heap-size and A[r] < A[largest]
        smallest = i
    if smallest \ne i
        exchange A[i] with A[smallest]
        MIN-HEAPIFY(A, smallest)
```

The running time is same to MAX-HEAPIFY.

6.2-4

Run once and exit.

6.2-5

Exit.

6.2-6

```c
MAX-HEAPIFY-Loop(A, i)
    while i < A.heap-size / 2
        l = LEFT(i)
        r = RIGHT(i)
        if l \le A.heap-size and A[l] > A[i]
            largest = l
        else largest = i
        if r \le A.heap-size and A[r] > A[largest]
            largest = r
        if largest \ne i
            exchange A[i] with A[largest]
            i = largest
        else return
```

6.2-7

When the root one is the smallest one in the subtree, then it floats down level by level, the height is $\lfloor \lg n \rfloor$, so the running time is $\Omega(\lg n)$.

## Building a heap

Based on exercise 6.1-8, $A[\lfloor n/2 \rfloor + 1:n]$ are leaves, so we can build a heap by running MAX-HEAPIFY on other nodes.

```c
BUILD-MAX-HEAP(A,n)
    A.heap-size = n
    for i = \lfloor n/2 \rfloor downto 1
        MAX-HEAPIFY(A,i)
```

By using loop-invarant we can prove it right.

Now we can analysis the running time of build a heap.

We use the fact that at most $\lceil n/2^{h+1}\rceil$ at any height $h$.

The total time is:

$$
\begin{align*}
\sum_0^{\lfloor \lg n \rfloor} \lceil \frac{n}{2^{h+1}}\rceil ch &\le \sum_0^{\lfloor \lg n \rfloor} \frac{n}{2^h} ch \\
&= cn \sum_0^{\lfloor \lg n \rfloor} \frac{h}{2^h} \\
&le cn \sum_0^\infty \frac{h}{2^h} \\
&\le 2cn \\
&= O(n)
\end{align*}
$$

### Exercises

6.3-1

[5,3,17,10,84,19,6,22,9]
[5,3,17,22,84,19,6,10,9]
[5,3,19,22,84,17,6,10,9]
[5,84,19,22,3,17,6,10,9]
[84,22,19,10,3,17,6,5,9]

6.3-2

$$
\begin{align*}
\lceil n / 2^{h+1} \rceil &\ge n / 2^{h+1} \\
&\ge n/2^{\lg n + 1} \\
&= 1/2
\end{align*}
$$

6.3-3

MAX-HEAPIFY request the left child and right child are already max-heap. Increasing is not fit to it.

6.3-4

Based on 6.1-8, we have proved that inter node num is $\lfloor n/2 \rfloor$, and leaf node num is $\lceil n/2 \rceil$. So when $h=0$, it's true.

If we remove all leaf nodes, then $h=1$ become leaf nodes. Using the conclusion, we have $n = \lfloor n/2 \rfloor \le n/2$.

$$
\lceil \lfloor n/2 \rfloor / 2 \rceil \le \lceil  n/2^2 \rceil
$$

Similarly, for $h=i$, we have subtree $n_i \le n/2^i$.

So the height h nums is most $\lceil  n/2^{h+1} \rceil$

## The heapsort algorithm

```c
HEAPSORT(A,n)
    BUILD-MAX-HEAP(A,n)
    for i = n downto 2
        exchange A[1] with A[i]
        A.heap-size = A.heap-size - 1
        MAX-HEAPIFY(A,1)
```

### Exercises

6.4-1

[5,13,2,25,7,17,20,8,4]
[25,13,20,8,7,17,2,5,4]
[20,13,17,8,7,4,2,5,25]
[17,13,5,8,8,4,2,20,25]
[13,8,5,2,7,4,17,20,25]
[8,7,5,2,4,13,17,20,25]
[7,4,5,2,8,13,17,20,25]
[5,4,2,7,8,13,17,20,25]
[4,2,5,7,8,13,17,20,25]
[2,4,5,7,8,13,17,20,25]

6.4-2

Initialization: Prior to the first iteration of the loop, $i = n$, By BUILD-MAX-HEAP, $A[1:n]$ is a max-heap contains the $n$ smallest numbers, and $A[n+1:n]$ is empty.

Maintenance: $A[i]$ is the largest one in $A[1:i]$, so $A[1:i-1]$ maintains the smallest numbers. And $A[i]$ is smaller than any one in $A[i+1:n]$, so $A[i:n]$ keeps the sorted largest numbers.

Termination: $A[1]$ is the smallest one, and $A[2:n]$ is the sorted largest numbers, so we have $A[1:n]$ sorted.

6.4-3

Both are $O(n\lg n)$, the increasing and decreasing order only effect BUILD-MAX-HEAP, but the running time is dominated by the MAX-HEAPIFY loop.

6.4-4

By exercise 6.2-7, we have proved MAX-HEAPIFY is $\Omega(\lg n)$, so the total HEAPSORT running time is $\Omega(n \lg n)$

6.4-5

When $A[i]$ exchange with $A[1]$, if $A[i]$ belong to $A[2]$ subtree and $A[2] > A[3]$, then $A[i]$ must float down until bottom, which take $\lfloor \lg n \rfloor$. Similarly, if if $A[i]$ belong to $A[3]$ subtree and $A[3] > A[2]$, we know there are at least $n/3-1$ nodes in subtree of $A[2]$ or $A[3]$, and half of them at bottom, so there at least $\Omega(1/2*1/3n\lg n) = \Omega(n\lg n)$

## Priority queues

A heap can support any priority-queue operation on a set of size $n$ in $O(\lg n)$ time.

When a heap implements a priority queue, we treat each array element as a pointer to an object in the priority queue. And every object has an attribute key, which determines where in the heap the object belongs.

```c
MAX-HEAP-MAXIMUM(A)
if A.heap-size < 1
    error "heap underflow"
return A[1]

MAX-HEAP-EXTRACT-MAX(A)
max = MAX-HEAP-MAXIMUM(A)
A[1] = A[A.heap-size]
A.heap-size = A.heap-size - 1
MAX-HEAPIFY(A,1)
return max

MAX-HEAP-INCREASE-KEY(A,x,k)
if k < x.key
    error "new key is smaller than current key"
x.key = k
find the index i in array A where object x occurs
while i > 1 and A[PARENT(i)].key < A[i].key
    exchange A[i] with A[PARENT(i)], updating the information that maps priority queue objects to array indices
    i = PARENT(i)

MAX-HEAP-INSERT(A,x,n)
if A.heap-size == n
    error "heap overflow"
A.heap-size = A.heap-size + 1
k = x.key
x.key = - \infty
A[A.heap-size] = x
map x to index heap-size in the array
MAX-HEAP-INCREASE-KEY(A,x,k)
```

### Exercises

6.5-1

[15,13,9,5,12,8,7,4,0,6,2,1]
[1,13,9,5,12,8,7,4,0,6,2]
[13,12,9,5,6,8,7,4,0,1,2]

6.5-2

[15,13,9,5,12,8,7,4,0,6,2,1,10]
[15,13,10,5,12,9,7,4,0,6,2,1,8]

6.5-3

```c
MIN-HEAP-MINIMUM(A)
if A.heap-size < 1
    error "heap underflow"
return A[1]

MIN-HEAP-EXTRACT-MIN(A)
min = MIN-HEAP-MINIMUM(A)
A[1] = A[A.heap-size]
A.heap-size = A.heap-size - 1
MIN-HEAPIFY(A,1)
return min

MIN-HEAP-DECREASE-KEY(A,x,k)
if k > x.key
    error "new key is larger than current key"
x.key = k
find the index i in array A where object x occurs
while i > 1 and A[PARENT(i)].key > A[i].key
    exchange A[i] with A[PARENT(i)], updating the information that maps priority queue objects to array indices
    i = PARENT(i)

MIN-HEAP-INSERT(A,x,n)
if A.heap-size == n
    error "heap overflow"
A.heap-size = A.heap-size + 1
k = x.key
x.key = \infty
A[A.heap-size] = x
map x to index heap-size in the array
MIN-HEAP-DECREASE-KEY(A,x,k)
```

6.5-4

```c
MAX-HEAP-DECREASE-KEY(A,x,k)
if k > x.key
    error "new key is larger than current key"
x.key = k
find the index i in array A where object x occurs
MAX-HEAPIFY(A,i)
```

6.5-5

Before run MAX-HEAP-INCREASE-KEY, heap must kep its max-heap priority.

6.5-6

MAX-HEAPIFY just floats down, not floats up.

6.5-7

Initialization: It satisfies the max-heap property. So if both nodes $PARENT(i)$ and $LEFT(i)$ exist, then we have $A[PARENT(i)].key \ge A[i].key \ge A[LEFT(i)].key$. If both nodes $PARENT(i)$ and $RIGHT(i)$ exist, then we have $A[PARENT(i)].key \ge A[i].key \ge A[RIGHT(i)].key$.

Maintenance: If $A[PARENT(i)].key < A[i].key$, specially, let $i = LEFT(PARENT(i))$, then based on c, we have $A[PARENT(PARENT(i))] \ge A[PARENT(i)]$ and $A[PARENT(PARENT(i))] \ge A[PARENT(i)] \ge A[RIGHT(PARENT(i))]$ and $A[PARENT(i)] \ge A[RIGHT(PARENT(i))]$, after changing and when $i = PARENT(i)$, we have $A[PARENT(i)] \ge A[LEFT(i)]$ and $A[PARENT(i)] \ge A[RIGHT(i)]$. And $A[i] \ge A[LEFT(i)]$ and $A[i] \ge A[RIGHT(i)]$, so c is hold. 

Termination: If $A[PARENT(i)].key < A[i].key$, together with c, we have the max-heap.

6.5-8

```c
MAX-HEAP-INCREASE-KEY(A,x,k)
if k < x.key
    error "new key is smaller than current key"
find the index i in array A where object x occurs
while i > 1 and A[PARENT(i)].key < k
    A[i].key = A[PARENT(i)].key, updating the information that maps priority queue objects to array indices
    i = PARENT(i)
A[i].key = k
```

6.5-9

```c
Initialize $A, i=0$
ENQUEUE(Q,x)
    i = i+1
    MAX-HEAP-INSERT(A,x,-i)

DEQUEUE(Q)
    return MAX-HEAP-EXTRACT-MAX(A)
```

```c
Initialize $A, i=0$
PUSH(Q,x)
    i = i+1
    MAX-HEAP-INSERT(A,x,i)

POP(Q)
    return MAX-HEAP-EXTRACT-MAX(A)
```

6.5-10

```c
MAX-HEAP-DELETE(A,x)
    find the index i in array A where object x occurs
    k = A[i].key
    A[i] = A[A.heap-size]
    A.heap-size = A.heap-size - 1
    if A[i].key < k
        MAX-HEAPIFY(A,i)
    else
        while i > 1 and A[PARENT(i)].key < A[i].key
        exchange A[i] with A[PARENT(i)], updating the maps.
        i = PARENT(i)
```

6.5-11

First step get the first one from $k$ sorted list and build a $k$ elements min-heap. Then every step we extract a minium one from the heap and insert one from the same list. Insertion takes $\lg k$, we have $n$ numbers, so the total running time is $O(n \lg k)$.

## Problems

6.1

a.

No, [1,2,3] BUILD-MAX-HEAP get [3,2,1] and BUILD-MAX-HEAP' get [3,1,2].

b.

When the input array is increasing, it taks $\Theta(n\lg n)$.

6.2

a.

$$
CHILD(i,k) = (i-1)d + k + 1
$$

$$
PARENT(i) = \lfloor (i-2)/d \rfloor
$$

b.

$$
h = \Theta(\lg_d n)
$$

c.

```c
MAX-D-HEAPIFY(A, i)
    largest = i
    for k from 1 to d
        c = CHILD(i,k)
        if c \le A.heap-size and A[c] > A[i]
        largest = c
    if largest \ne i
        exchange A[i] with A[largest]
        MAX-D-HEAPIFY(A, largest)

MAX-D-HEAP-MAXIMUM(A)
    if A.heap-size < 1
        error "heap underflow"
    return A[1]

MAX-D-HEAP-EXTRACT-MAX(A)
    max = MAX-D-HEAP-MAXIMUM(A)
    A[1] = A[A.heap-size]
    A.heap-size = A.heap-size - 1
    MAX-D-HEAPIFY(A,1)
return max
```

Running time is $O(d\lg_d n$)

d.

```c
MAX-D-HEAP-INCREASE-KEY(A,x,k)
    if k < x.key
        error "new key is smaller than current key"
    find the index i in array A where object x occurs
    while i > 1 and A[PARENT(i)].key < k
        A[i].key = A[PARENT(i)].key, updating the information that maps priority queue objects to array indices
        i = PARENT(i)
    A[i].key = k
```

Running time is $O(\lg_d n$)

e.

```c
MAX-HEAP-INSERT(A,x,n)
    if A.heap-size == n
        error "heap overflow"
    A.heap-size = A.heap-size + 1
    k = x.key
    x.key = - \infty
    A[A.heap-size] = x
    map x to index heap-size in the array
    MAX-D-HEAP-INCREASE-KEY(A,x,k)
```

Running time is $O(\lg_d n$)

6.3

a.

$$
\begin{matrix}
2 & 3 & 4 \\
5 & 8 & 9 \\
12 & 14 & 16
\end{matrix}
$$

b.

If $Y[1,1] = \infty$, then $Y[i,j] \ge Y[i,1] \ge Y[1,1] =  \infty$, so no element exists.and it's empty.

If $Y[m,n] = \infty$, then $Y[i,j] \le Y[i,n] \le Y[m,n] =  \infty$, so all element exists.and it's full.

c.

```c

MIN-YOUNG-TABLEAULY(Y,m,n,i,j)
    if i < m and j < n
        if Y[i+1,j] < Y[j,i+1] and Y[i+1,j] < Y[i,j]
            exchange Y[i+1,j] with Y[i,j]
            MIN-YONG-TABLEAULY(Y,m,n,i+1,j)
        else if Y[i,j+1] < Y[i+1,j] and Y[i,j+1] < Y[i,j]
            exchange Y[1,j+1] with Y[i,j]
            MIN-YONG-TABLEAULY(Y,m,n,i,j+1)
    else if i < m and Y[i+1,j] < Y[i,j]
            exchange Y[i+1,j] with Y[i,j]
            MIN-YONG-TABLEAULY(Y,m,n,i+1,j)
    else if j < n and Y[i,j+1] < Y[i,j]
            exchange Y[1,j+1] with Y[i,j]
            MIN-YONG-TABLEAULY(Y,m,n,i,j+1)

YOUNG-TABLEAU-EXTRACT-MIN
    min = Y[1,1]
    Y[1,1] = Y[m,n]
    Y[m,n] = \infty
    MIN-YONG-TABLEAULY(Y,m,n,1,1)
```

Everytime it recursively move down to right or below once, until it stop at the [m,n] (or stop earily). So the total moves are $O(m+n)$.

d.

```c
YOUNG-TABLEAU-INSERT
    Y[m,n] = k
    i = m
    j = n
    while i > 1 and j > 1
        if Y[i-1,j] > Y[i,j-1] and Y[i-1,j] > Y[i,j]
            exchange Y[i-1,j] with Y[i,j]
            i = i - 1
        else if Y[i,j-1] > Y[i-1,j] and Y[i,j-1] > Y[i,j]
            exchange Y[i,j-1] with Y[i,j]
            j = j - 1
        else
            break
        
    
    while i > 1 and Y[i-1,j] > Y[i,j]
        exchange Y[i-1,j] with Y[i,j]
        i = i - 1
    
    while j > 1 and Y[i,j-1] > Y[i,j]
        exchange Y[i,j-1] with Y[i,j]
        j = j - 1
```

e.

Run YOUNG-TABLEAU-EXTRACT-MIN $n^2$ times, each takes $O(n)$ time, together in $O(n^3)$ time.

f.

```c
YONG-TABLEAU-IS-STORED(Y,m,n,k)
    i = m
    j = 1
    while i \ge 1 and j \le n
        if Y[i,j] = k
            return true
        else if Y[i,j] < k
            j = j + 1
        else
            i = i - 1
    return false
```

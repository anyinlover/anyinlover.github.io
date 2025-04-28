# Greedy Algorithms

## An activity-selection problem

15.1-1

Based on 15.2, the running time is $\Theta(n^3)$, which is larger than the greedy algorithm's $\Theta(n)$.

15.1-2

Similary to choosing the earliest finish time, choosing the latest start time leaves the resource available for as many of the activities that before it as possible.

Similary to Theorem 15.1, We need to prove that :

Consider any nonempty subproblem $S_k$, and let $a_m$ be an activity in $S_k$ with the latest start time. Then $a_m$ is included in some maximum-size subset of mutually compatible activities of $S_k$.

Let $A_k$ be a maximum-size subset of mutually compatible activities in $S_k$, and let $a_j$ be the activity in $A_k$ with the latest start time. If $a_j = a_m$, we are done. If not, we can repalce $a_j$ with $a_m$, then we have $|A_k'| = |A_k|$.

15.1-3

Least duration not work: (1,3),(4,6),(5,6)
Fewest overlaps not work: (1,4),(3,5),(4,6),(7,8)
Earliest start time not work: (2,3),(1,5),(4,6)

15.1-4

The greedy strategy is every time select the earliest start time activity. Maintain a min-heap of finish time, if the current activity start time is larger than the min-heap minium, pop it and insert into it by the room num and current finish time. Otherwise get a new room num and insert it.

15.1-5

Let $c[i]$ represent the first $i$ th activities' total largest weights. We have the recurrence $c[i] = \max\{c[k]+w_i, c[i-1]\}$. $k$ is the largest finish time $f_k \ge s_i$, because finish time is sorted, we can use binary search. Use dynamic programming, we can solve it.

## Elements of the greedy strategy

15.2-1

We want to prove in any subproblem at most $W$ pounds, it must have the $\min\{W, w_k\}$ left valuable item $i_k$. Otherwise, if we replace it with $\min\{W, w_k\}$, the total value is larger, which can't be true.

15.2-2

Let $c[i,w]$ is the max pounds get from $i$ items not larger than $w$ pounds. We will have the following recurrence:

$$
c[i,w] = \begin{cases} \\
0 &\text{if } i = 0 \text{ or } w = 0 \\
c[i-1,w] &\text{if } w_i > w \\
\max\{v_i + c[i-1,w-w_i], c[i-1,w]\} &\text{if } w_i \le w
\end{cases}
$$

So we have the following algorithm:

```c
0-1-KNAPSACK(v,w,n,W)
let c[0:n,0:W] be new array.
for w = 0 to W
    c[0,w] = 0
for i = 1 to n
    c[i,0] = 0
    for w = 1 to W
        if w_i <= w and v_i + c[i-1,w-w_i] > c[i-1,w]
            w[i,w] = v_i + c[i-1,w-w_i]
        else w[i,w] = w[i-1,w]
return c[n,W]
```

15.2-3

We can use the greedy strategy that choosing the left most valuable item. We argue that the subproblem must have the most valuable item. otherwise, using the cut-and-paste way, we can replace any other item with the most valuable item as it is also the least weight.

15.2-4

The professor can supply water at the last point in m miles.

15.2-5

```c
GREEDY-INTERVAL(x,n)
if n = 0
    return 0
let S be a new set
i = 1
while i \le n
    S = S + \{x[i],x[i]+1\}
    j = x[i]+1
    while (i \le n and x[i] \le j)
        i = i + 1
return S
```

15.2-6

Use the linear-time median algorithm to find the median $m$, then partition the items into three parts: Greater, Equal and Less.
Use recursive algorithm to find more. The recurrence is:

$$T(n) = T(n/2) + \Theta(n)$$

So the final $T(n) = \Theta(n)$.

15.2-7

If the two sets are sorted, then we get the max payoff.

When $a_i \le a_j, b_i \le b_j$, we want to prove $a_i^{b_i}a_j^{b_j} \ge a_i^{b_j}a_j^{b_i}$. Use the log function:

$$
b_i \ln a_i + b_j \ln a_j \ge b_j \ln a_i + b_i \ln a_j \\
(b_i - b_j)(\ln a_i - \ln a_j) \ge 0
$$

## Huffman codes

15.3-1

We have

$$
a.freq \le b.freq \\
x.freq \le y.freq \\
x.freq \le a.freq \\
y.freq \le b.freq \\
$$

So

$$
x.freq \le a.freq \le b.freq \\
x.freq \le y.freq \le b.freq
$$

Because $x.freq = b.freq$, so $a.freq = b.freq = x.freq = y.freq$.

15.3-2

If any nonleaf node has only one child, in case 1, the child is a character leaf. Then we can change the nonleaf node to the character leaf, which cost less. In case 2, the child is another nonleaf node. We can move any leaf node in the subtree rooted at the child as the sibling, which cost less. In both way, it's can't be optimal.

15.3-3

h 0 g 10 f 110 e 1110 d 11110 c 111110 b 1111110 a 1111111

The largest one is 0, then add 1 at the front, the first one is n bits 1.

15.3-4

From Lemma 15.3 proof, we have:

$$
B(T) = B(T') + x.freq + y.freq = B(T') + z.freq = \sum z.freq
$$

If we keep merge leaf, finally we get all internal nodes' frequencies sum.

15.3-5

Take a preorder walk, 0 if it's a internal node, 1 if it's a leaf node. There are n leaf nodes and n-1 internal nodes, so it is totally 2n-1 bits. For n character, every character can use $\lceil \lg n \rceil$, so together $2n-1+n\lceil \lg n \rceil$.

15.3-6

```c
TERNARY-HUFFMAN(C)
n = |C|
Q = C
if n%2 == 0 // need apply false element first
    q.freq = 0
    INSERT(Q,z)
for i = 1 to n - 1
    allocate a new node q
    x = EXTRACT-MIN(Q)
    y = EXTRACT-MIN(Q)
    z = EXTRACT-MIN(Q)
    q.first = x
    q.second = y
    q.third = z
    q.freq = x.freq + y.freq + z.freq
    INSERT(Q,z)
```

The Greedy Property and the optimal structure proof is same to huffman.

15.3-7

Because the maximum character frequency is less than twice the minimum character frequencey, so after the first round, it have 128 internal nodes. It have the same property. So the final tree is a complete binary tree, just same as the ordinary fixed-length code tree.

15.3-8

If there exits a lossless compression schema that guarantees that for every input file, the corresponding output file is shorter. The all possible input files of length $n$ is $2^n$, the all possible output files of length less than $n$ is $\sum_1^{n-1} 2^{n-1} = 2^n - 2$. So there are must two different inputs have same output, which can not be true.

## Offline caching

15.4-1

```c
INITIALIZE(C,b)
Let H be a hash table that key is block, value is a stack which stores the block requests positions.
For i = n to 1
    PUSH(H[b_i], i)

Let A be a k max-value heap.
for c in C
    if H[c] = 0
        MAX-HEAP-INSERT(A, (\infty,c), k)
    else
        MAX-HEAP-INSERT(A, (H[c].top(),c),k)


CACHE-MANAGER(C, k, b, i)
for j = 1 to i
    if b_j \in C
        print b_j "cache hit"
    else
        print b_j "cache miss"
        if |C| == k
            c = MAX-HEAP-EXTRACT-MAX(A)
            print c "block evicted"
        POP(H[b_j])
        INSERT(C, b_j)
        if H[b_j] = 0
            MAX-HEAP-INSERT(A, (infty,b_j),k)
        else
            MAX-HEAP-INSERT(A, (H[c].top(),b_j),k)
```

15.4-2

a,b,c,d,a,b,c k = 3. For LRU the cache misses is 4. For furthest-in-future, the cache misses is 2.

15.4-3

When $|D_j| = k - 1$, $b_j \notin D_j$ and $b_j = y$, then for $j+1$, it will not be true.

15.4-4

Change the multiple blocks as a series of block requests. As we know, multiple blocks requests maybe violate furthest-in-future principle (need to replace multiple blocks at once). As we have proved, furthest-in-future method is the optimal strategy, so multiple blocks can't be better.

## Problems

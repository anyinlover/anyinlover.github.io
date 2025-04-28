# Sorting in Linear Time

## Lower bounds for Sorting

8.1-1

Every element need at least one time of sort, so the least comparision times is $n-1$, which is the same to the smallest possible depth of a leaf.

8.1-2

We have:

$$ \int_0^n \lg k dk \le \sum_{k=1}^n \lg k \le \int_1^{n+1} \lg k dk $$

Because we have:

$$
\int \lg k dk = k\lg k - \int k d \lg k = k\lg k - \int \frac{k}{k\ln 2} dk = k\lg k - lg e k + C
$$

So:

$$
\int_1^{n+1} \lg k dk = (n+1)\lg(n+1) - \lg e (n+1) + \lg e = O(n\lg n)
$$

$$
\int_0^n \lg k dk = n \lg n - \lg e n = \Omega(n\lg n)
$$

Finally, we get:

$$
\lg(n!) = \sum_{k=1}^n \lg k = \Theta(n\lg n)
$$

8.1-3

Suppose for every $n \ge k$, half of leaves have linear depth $cn$. For a tree have depth $cn$, it at most have $2^{cn}$ leaves. So the question become $n!/2 \le 2^{cn}$.

We have:
$$
\lg (n!/2) = \Theta(n \lg n) \\
\lg (2^{cn}) = \Theta(n) \\
\lg (n!/n) = \Theta(n \lg n) \\
\lg (n!/2^n) = \Theta(n \lg n)
$$

So in all three condition, it can't be true when $n$ is large enough.

8.1-4

We want to use the partly sorted information, so we filter the position $i$ where $i\ mod\ 4 = 0$, sort the left and insert it back. We need:

$$
\Omega(3n/4 \lg (3n/4)) + \Theta(n/4) = \Omega(n \lg n)
$$

## Counting sort

8.2-1

A = 6,0,2,0,1,3,4,6,1,3,2
C = 2,2,2,2,1,0,2

C = 2,4,6,8,9,9,11

B =  , , , , ,2, , , , , 
C = 2,4,5,8,9,9,11

B =  , , , , ,2, ,3, , ,
C = 2,4,5,7,9,9,11

B =  , , ,1, ,2, ,3, , , 
C = 2,3,5,7,9,9,11

B =  , , ,1, ,2, ,3, , ,6
C = 2,3,5,7,9,9,10

B =  , , ,1, ,2, ,3,4, ,6
C = 2,3,5,7,8,9,10

B =  , , ,1, ,2,3,3,4, ,6
C = 2,3,5,6,8,9,10

B =  , ,1,1, ,2,3,3,4, ,6
C = 2,2,5,6,8,9,10

B =  ,0,1,1, ,2,3,3,4, ,6
C = 1,2,5,6,8,9,10

B =  ,0,1,1,2,2,3,3,4, ,6
C = 1,2,4,6,8,9,10

B = 0,0,1,1,2,2,3,3,4, ,6
C = 0,2,4,6,8,9,10

B = 0,0,1,1,2,2,3,3,4,6,6
C = 0,2,4,6,8,9,9

8.2-2

If $A[i] = A[j]$ where $ i < j $, then based on COUNTING-SORT, $A[j]$ first copy to $B[C[A[j]]]$, then $A[i] copy to $B[C[A[i]]]$, because $C[A[j]]$ decreases, we get $C[A[i]] < C[A[j]]$. So it is stable.

8.2-3

Similar to 8.2-2, but now $A[i]$ copy first, and $C[A[i]] > C[A[j]]$.

```c
INCREASING-COUNTING-SORT
let B[1:n] and C[0:k] be new arrays
for i = 0 to k
    C[i] = 0
for j = 1 to n
    C[A[j]] = C[A[j]] + 1
for i = 1 to k
    C[i] = C[i] + C[i-1]
mv = C[0] - 1
for i = 0 to k
    C[i] = C[i] - mv
for j = 1 to n
    B[C[A[j]]] = A[j]
    C[A[j]] = C[A[j]] + 1
return B
```

8.2-4

Initialization:

Before loop, the last element in $A$ with value i that has not yet been copied into $B$. Bebacuse $C[i]$ is the num of elements that less or equal to $i$, its true.

Maintenance:

In loop, $C[i]$ decreases 1, so for other element before $j$ that equal i, $C[i]$ is still the num of elements that less or equal to $i$.

Termination:

When termination, every element has been copied into $B$.

8.2-5

The key is to exchange when $i < j$.

```c
INARRAY-COUNTING-SORT
let C[0:k] be new arrays
for i = 0 to k
    C[i] = 0
for j = 1 to n
    C[A[j]] = C[A[j]] + 1
for i = 1 to k
    C[i] = C[i] + C[i-1]
j = 1
while j \le n
    v = A[j]
    i = C[v]
    if i \le j
        if i < j
            A[j] = A[i]
            A[i] = v
        else
            j = j + 1
        C[v] = C[v] - 1
    else
        j = j + 1
```

8.2-6

```
PREPROCESS
let C[0:k] be new arrays
for i = 0 to k
    C[i] = 0
for j = 1 to n
    C[A[j]] = C[A[j]] + 1
for i = 1 to k
    C[i] = C[i] + C[i-1]

QUERY(a,b)
if b > k
    b = k
if a < 1
    a = 1

return C[b] - C[a-1]
```

8.2-7

```c
FRACTIONAL-COUNTING-SORT
let B[1:n] and C[0:10^d*k] be new arrays
for i = 0 to k
    C[i] = 0
for j = 1 to n
    C[A[j]] = C[A[j]*10^d] + 1
for i = 1 to k
    C[i] = C[i] + C[i-1]
for j = n downto 1
    B[C[A[j]*10^d]] = A[j]
    C[A[j]] = C[A[j]*10^d] + 1
return B
```

## Radix Sort

8.3-1

SEA
TEA
MOB
TAB
DOG
RUG
DIG
BIG
BAR
EAR
TAR
COW
ROW
NOW
BOX
FOX

----
TAB
BAR
EAR
TAR
SEA
TEA
DIG
BIG
MOB
DOG
COW
ROW
NOW
BOX
FOX
RUG

----
BAR
BOX
BIG
COW
DIG
DOG
EAR
FOX
MOB
NOW
ROW
RUG
SEA
TAB
TAR
TEA

8.3-2

Stable sorting algorithms: insertion sort, merge sort

Change element to tuple, the first one is the element, the second one is the index of same value. the time is same, but the space is double.

8.3-3

When $i$ = 1, sorting one digit mean sorting all. If the last $i-1$ digits have been sorted, now we sort the $i$ digit, for any two element, if their $i$ digit is different, then by sort the $i$ digit, they are sorted, if their $i$ digit is same, then because of the stable sort algorithm, they are sorted by the last $i-1$ digits. Here we need the sort algorithm be stable.

8.3-4

Use another $D[0:k]$ to count for next digit in line 11-13.

8.3-5

Use b bits can express $2^b - 1$, so when $2^b-1 \ge n^3 - 1$, we get $b = 3\lg n = O(n)$, so the running time is $\Theta(n)$.

8.3-6

In the worst case, we need $\sum_{i=0}^d k^i = \frac{k^d - 1}{k-1}$ passes. And We need $k^d$ piles.

## Bucket Sort

8.4-1

/
.13,.16,/
.20,/
.39,/
.42,/
.53,/
.64,/
.79,.71,/
.89,/
/

---
/
.13,.16,/
.20,/
.39,/
.42,/
.53,/
.64,/
.71,.79,/
.89,/
/

8.4-2

Based on 8.1, if all $n$ elements fall into one bucket, Then $T(n) = O(n^2)$.

Change insert sort to heap sort.

8.4-3

It's a binomial distribution. We have:

$$
E[X] = np = 1 \\
V[X] = npq = 0.5 \\
E^2[X] = 1 \\
E[X^2] = V[X] + E^2[X] = 1.5
$$

8.4-4

We can sort it by combinating radix sort and bucket sort. 

```c
SOLUTION-SORT(A, n)
let B[0:n-1] be a new array
for i = 0 to n - 1
    make B[i] an empty list
for i = 1 to n
    insert A[i] into list B[\lfloor n * n * (A[i] - \lfloor A[i]*10 \rfloor /10) \rfloor]
for i = 0 to n - 1
    sort list B[i] by B[i] - \lfloor B[i]*10 \rfloor /10 with insertion sort
concatenate the lists B[0], B[1], ... , B[n-1] together in order and put it in A

COUNT-SORT use the first digit after decimal point
```

8.4-5

Because the points are uniformly distributed, so there are uniformly distributed on radius r too. Divide radius into $n$ parts equally, use it as bucket.

8.4-6

We just need get $n$ buckets that the probability is same. Let $P(x_i) = i/n$, we can get it.

## Problems

8.1

a. Every permutation will reach one exact leaf and none of other leaves would be reached. So there are exactly $n!$ leaves are labeled $1/n!$ and the rest are labeled 0.



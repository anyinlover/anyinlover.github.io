# Medians and Order Statistics

## Minimum and maximum

9.1-1

We can use a binary tree to compare two elements every time. the last left one is the smallest one. And in this method there are $n-1$ comparisions, because every element except for the smallest one just lost once.

In the other hand, only the elements that has been beat by the smallest one could be possible the second smallest one. The height of the tree is $\lceil \lg n \rceil$, and there are at most $\lceil \lg n \rceil$ elements have been compared to the smallest one. It need $\lceil \lg n \rceil - 1$ comparisions to find the smallest one.

So together need $n + \lceil \lg n \rceil - 2$ comparisions.

9.1-2

We find the median one in the first three elements, it could neither be the minimum or the maximum. In best case, it takes just 2 comparisions.

9.1-3

The fastest horse run twice. the second fastest horse must be among the second ones in these two races. the third one must be among the third ones in these two races or the second one in the race in which the 2nd second one is the fastest. together there are 5 horses. We need only one more race to determine it. So together is 7 races.

9.1-4

When $n$ is even, $3n/2-2$ = $\lceil 3n/2 \rceil - 2$.

When $n$ is odd, if $n=1$, then $0 = \lceil 3n/2 \rceil - 2$. If $n \ge 3$, then take first 3 elements to determine the minimum and maximum, then use the same method to compare the rest. it take $3 + 3(n-3)/2 = \lceil 3n/2 \rceil - 2$.

## Selection in expected linear time

9.2-1

If RANDOMIZED-SELECT($A, p, q-1, i$) is empty, which mean $p > q - 1$, so $q \le p$, so $k = q-p+1 \le 1$, so $i < k \le 1$, but $ i \ge 1$, so it can't be true.

Similarly, if RANDOMIZED-SELECT($A, q+1, r, i-k$) is empty, which mean $q+1 > r$, so $q \ge r$, so $k = q-p+1 \ge r - q + 1$, so $ i > k \ge r - q + 1$, but $i \ge r-p+1$, so it can't be true.

9.2-2

```c
ITERATE-RANDOMIZED-SELECT(A, p, r, i)
while true
    if p == r
        return A[p]
    q = RANDOMIZED-PARTITION(A,p,r)
    k = q - p + 1
    if i == k
        return A[q]
    elseif i < k
        r = q - 1
    else
        p = p + 1
        i = i - k
```

9.2-3

When choose the pivot element from large to small, it is the worst case.

2,3,0,5,7,9,1,8,6,4
2,3,0,5,7,4,1,8,6
2,3,0,5,7,4,1,6
2,3,0,5,6,4,1
2,3,0,5,1,4
2,3,0,4,1
2,3,0,1
2,1,0
0,1
0

9.2-4

When $n=i=1$, it is true.

If When $n = j - 1$ is true, if $i < n$, then after the first iteration, the problem is same to $n = j - 1$, the first iteration need $n-1$ comparision for any permutation, so it's true. If $i = n$, suppose the pivot element series is $q_1, q_2, \cdots, q_m$, because pivot element is random choice and not depend on permutation, the   total comparisions is certain when the pivot element series is certain as well. So in both cases, it is not effected by the input permutation.

## Selection in worst-case linear time

9.3-1

We get the running time:

$$
T(n) \le T(n/7) + T(5n/7) + \Theta(n)
$$

By substitution, we get $T(n) = O(n)$, and $T(n) = \Theta(n)$.

9.3-2

We put the non-group elements into line 23 and 24, it have 4 elements at most. We have the running time:

$$
T(n) \le T(n/5) + T(7n/10 + 4) + \Theta(n)
$$

For $n \ge 40$, we have $T(n) \le T(n/5) + T(4n/5) + \Theta(n)$, we get $T(n) = O(n)$.

9.3-3

```c
SELECT-QUICKSORT(A,p,r)
if p < r
    SELECT(A,p,r, \lfloor (p+r)/2 \rfloor)
    SELECT-QUICKSORT(A,p,\lfloor (p+r)/2 \rfloor-1)
    SELECT_QUICKSORT(A,\lfloor (p+r)/2 \rfloor+1, r)
```

9.3-4

If there are one element have not been compared to the ith smallest element, we can't decide that it's the ith smallest element. So every other element have been compared to this element, so we can find $i-1$ smaller elements and $n-i$ larger elements.

9.3-5

If 5 elements are $A,B,C,D,E$, we first compare $A$ and $B$, suppose $A > B$, compare $C$ and $D$, suppose $C > D$, then we compare $B$ and $D$, suppose $B > D$, after 3 comparisions, we can know $D$ can't be the median, because there have been 3 elements larger than it. We then compare $E$ and $C$, suppose $E > C$, then we compare $C$ and $B$, suppose $B > C$, at last we compare $B$ and $E$, the small one is the median. Notice all the suppose is ok because the elements are equal to exchange in that condition.

9.3-6

```c
BLACK-MEDIAN-SELECT(A,p,r,i)
x = BLACK-MEDIAN(A,p,r)
q = PARTITION-AROUND(A,p,r,x)
k = q - p + 1
if i == k
    return A[q]
elseif i < k
    return SELECT(A,p,q-1,i)
else
    return SELECT(A,q+1,r,i-k)
```

9.3-7

Sort the wells by $y$, the problem actually is to find the median $y$, in the median the total length is the smallest. So we could use the SELECT to determine it in linear time.

9.3-8

Use the SELECT algorithm, we first get $\lfloor k/2 \rfloor$ th quantile, and then split the set into two small set, it's a binary tree, and the height is $\lceil \lg k \rceil$. 

We get $T(n) = O(n \lg k)$.

9.3-9

```c
CLOSEST-NUMBERS(A,n,k)
x = SELECT(A,1,n,\lfloor (n+1)/2 \rfloor)
let B = empty tuple list
for i = 1 to n
    B[i] = (|A[i]-x|, A[i])
xb = SELECT(B,1,n,k)
j = PARTITION-AROUND(B,1,n,xb)
return B[1:j]
```

9.3-10

If $X[k]$ is the final median, then we have $Y[n-k] < X[k] < Y[n-k+1]$. When $k=n$, we have $X[k] < Y[1]$. If $X[k']$ is the final median, where $k' < k$, then we have $Y[n-k+1] < X[k]$. If $X[k'']$ is the final median, where $k'' < k$, then we have $Y[n-k] > X[k]$.We can use the binary search to find the suitable one.

```c
TWO-ARRAY-MEDIAN(X,Y,n)
median = FIND-MEDIAN(X,Y,n,1,n)
if median == NOT-FOUND
    median = FIND-MEDIAN(Y,X,n,1,n)
return median

FIND-MEDIAN(A,B,n,low,high)
if low > high
    return NOT-FOUND
else k = \lfloor (low+high)/2 \rfloor
    if k == n and A[n] \le B[1]
        return A[n]
    elseif k < n and B[n-k] \le A[k] \le B[n-k+1]
        return A[k]
    elseif A[k] > B[n-k+1]
        return FIND-MEDIAN(A,B,n,low,k-1)
    else return FIND-MEDIAN(A,B,n,k+1,high)
```

## Problems

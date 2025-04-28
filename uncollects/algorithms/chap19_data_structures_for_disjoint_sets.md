# Data Structures for Disjoint Sets

## Disjoint-set operations

19.1-1

a b c d e f g h i j k
a b c d,i e f g h j k
a b c d,i e f,k g h j
a b c d,g,i e f,k h j
a b,d,g,i c e f,k h j
a,h b,d,g,i c e f,k j
a,h b,d,g,i,j c e f,k
a,h b,d,f,g,i,j,k c e
a,h b,d,f,g,i,j,k c e
a,h b,d,f,g,i,j,k c e
a,h b,d,f,g,i,j,k c e
a,e,h b,d,f,g,i,j,k c

19.1-2

We use loop invariant to prove that if the two vertices is in the same set, they must be connected. If two connected vertices are not in the same set, then on the path between these vertices there must at least one pair of vertices that are not in the same sets. (Otherwise all vertices on this path are in same set.) However, we have deal with all edges, if one edge are not in the same set, in line5 it will be unioned and finally in the same set. So it can't be hold.

19.1-3

FIND-SET $2|E|$

UNION $|V|-k$

## Linked-list represention of disjoint sets

19.2-1

```c
MAKE-SET(x)
let S be new set
S.head = x
S.tail = x
S.len = 1
x.set = S
x.next = NIL
return S

FIND-SET(x)
return x.set.head

UNION(x,y)
if x.set.len < y.set.len
    return UNION(y,x)

x.set.len = x.set.len + y.set.len
u = y.set.head
x.set.tail.next = u
x.set.tail = y.set.tail
while u != NIL
    u.set = x.set
    u = u.next
return x.set
```

19.2-2

1->2->3->4->5->6->7->8->9->10->11->12->13->14->15->16
1
1

19.2-3

Based on Theorem 19.1, If there are l UNION operations, the total running time for UNION is $O(l\lg l)$, So the total running time for m operations is $O(m-l + l\lg l)$

$$
cn + c(m-n-l) + c(\lg n)l = c(m-l+l\lg n)
$$

So when we charge $\Theta(1)$ for MAKE-SET and FIND-SET, charge $\lg n$ for UNION, by counting method it is covered.

19.2-4

$$
T(n) = \Theta(n) + \Theta(n-1) = \Theta(n)
$$

19.2-5

```c
MAKE-SET(x)
let S be new set
S.tail = x
S.len = 1
x.set = S
x.next = x
return S

FIND-SET(x)
return x.set.tail

UNION(x,y)
if x.set.len < y.set.len
    return UNION(y,x)

x.set.len = x.set.len + y.set.len
u = y.set.tail
v = u.next
u.next = x.set.tail.next
x.set.tail.next = v
x.set.tail = u
while v != u.next
    v.set = x.set
    v = v.next
return x.set
```

19.2-6

```c
UNION(x,y)
if x.set.len < y.set.len
    return UNION(y,x)

x.set.len = x.set.len + y.set.len
v = x.set.head.next
u = y.set.head
x.set.head.next = u
w = NIL
while u != NIL
    u.set = x.set
    w = u
    u = u.next
w.next = v
return x.set
```

## Disjoint-set forests

19.3-1

16/4(8/3(4/2(3/0),1/0,5/0,6/0,7/0),12/2(11,0),2/1,4/2,9/0,19/1,13/0,14/0,15/0)

19.3-2

```c
ITERATE-FIND-SET(x)
y = x
while y != y.p
    y = y.p
while x != x.p
    w = x.p
    x.p = y
    x = w
return y
```

19.3-3

In worst case, the tree's length is $O(\lg n)$, If we find the deepest element everytime, and $m$ is more larger than $n$. Then the FIND-SET operations will have the main cost $\Omega(m\lg n)$.

19.3-4

```c
MAKE-SET(x)
x.p = x
x.next = x
x.rank = 0

UNION(x,y)
LINK(FIND-SET(x), FIND-SET(y))

LINK(x, y)
if x.rank > y.rank
    y.p = x
else x.p = y
    if x.rank == y.rank
        y.rank = y.rank + 1
p = x.next
q = y.next
x.next = q
y.next = p

FIND-SET(x)
if x != x.p
    x.p = FIND-SET(x.p)
return x.p
```

19.3-5

Use the counting method, MAKE-SET take 2, 1 for this time, 1 for FIND-SET the node when the node's parent is not the root, as we notice, every node only take at most 1 time when the node's parent is not the root. LINK take 1, and FIND-SET take 1. So the amortized time is $O(1)$. It not effected if not using union by rank.

## Analysis of union by rank with path compression

19.4-1

In MAKE-SET line 2, we get that the value of x.rank is initially 0. FIND-SET don't change ranks. In Union, we find that only the root of tree will change the rank. And the paren's rank is always larger than the child's. And after UNION, internal nodes' ranks does not change.

19.4-2

Use the induction to prove it.

In the base case, we have $\lfloor \lg n \rfloor = \lfloor \lg 1 \rfloor = 0$.

If for $i \le k$ nodes, we have the rank at most $\lfloor \lg k \rfloor$.

For $k+1$ nodes, the rank increase only when $a = b = (k+1)/2$. So we get:

$$
rank = \lfloor \lg a \rfloor + 1 \le \lfloor \lg (k+1)/2 \rfloor + 1 = \lfloor \lg (k+1) \rfloor
$$

19.4-3

We need $\lceil \lg \lfloor \lg n \rfloor \rceil = O(\lg \lg n)$ bits.

19.4-4

MAKE-SET and UNION takes $O(1)$, FIND-SET takes $O(\lg n)$, so the total running time is $O(m\lg n)$.

19.4-5

No，it depends on $x.rank$ and $x.p.rank$. 

1,4,5 -> 1,0

19.4-6

Lemma 19.8 19.9 19.10 19.12 19.13 need to adapt to $c$.

19.4-7

$A_3(1) = 2047$, so we have for all $ n \le 2^{2047} - 1 $, we have $\alpha'(n) \le 3$.

We have:

$$
A_{\alpha'(n)}(x.rank) \ge A_{\alpha'(n)}(1) \ge \lg(n+1) > x.p.rank
$$

So we can replace $\alpha(n)$ with $\alpha'(n)$, and all lemmas are still true.

## Problems

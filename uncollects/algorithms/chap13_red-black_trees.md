# Red-Black Trees

## Properties of red-black trees

13.1-1

Binary Tree: 8(4(2(1,3),6(5,7)),12(10(9,11),14(13,15)))

Black-heights 2: 8(4R(2(1R,3R),6(5R,7R)),12R(10(9R,11R),14(13R,15R)))

Black-heights 3: 8(4R(2(1,3),6(5,7)),12R(10(9,11),14(13,15)))

Black-heights 4: 8(4(2(1,3),6(5,7)),12(10(9,11),14(13,15)))

13.1-2

26(17R(14(10R(7(3R,),12),16(15R,)),21(19(,20R),23)),41(30R(28,38(35R(,36),39R)),47))

If the inserted node is red, then 35 and 36 both are red, which not satisfy property 4.

If the inserted node is black, then for 35, the heights to its leaves are not equal, which not satisfy property 5.

13.1-3

Yes, all five properties are satisfied.

13.1-4

It may have 0,1,2 red children, after absorbing, it may have 2,3,4 degrees. The black-depth of the leaves is not changed.

13.1-5

Because to any leaf, the black-height is same. For the shortest path, it happen when every node on the path is black. No two red nodes can be adjact on the path (otherwise violate proeprty 4), so the longest path happens when black, red interleave one by one. The longest path is at most twice than the shortest path.

13.1-6

The smallest possible number when height is equal black-height: $2^k - 1$

The largest possible number when height is twice of black-height: $2^{2k}-1$

13.1-7

The largest possible ratio is 2, and the smallest possible ratio is 0.

Note: If we consider the specific n like 14, the problem is complex.

The smallest ratio is $\frac{n+1 - 2^{\lfloor\lg(n+1)\rfloor}}{2^{\lfloor\lg(n+1)\rfloor}-1}$.

The largest possible ratio is 

$$
\begin{cases}
\frac{3n-2^{\lfloor\lg(n+1)\rfloor+1}+1}{2^{\lfloor\lg(n+1)\rfloor+1}-1} &\text{if } \lfloor\lg(n+1)\rfloor \text{ is odd}
\frac{3n-2^{\lfloor\lg(n+1)\rfloor+1}-1}{2^{\lfloor\lg(n+1)\rfloor+1}+1} &\text{if } \lfloor\lg(n+1)\rfloor \text{ is even}
\end{cases}
$$

13.1-8

If the red node has only one non-NIL child, then for the red node, its black-height to the NIL leaf is 1, but to the NIL leaf of the non-NIL child is larger than 1, which violates proeprty 5.

## Rotations

13.2-1

```c
RIGHT-ROTATE(T,x)
y = x.left
x.left = y.right
if y.right != T.nil
    y.right.p = x
y.p = x.p
if x.p == T.nil
    T.root = y
elseif x == x.p.left
    x.p.left = y
else x.p.right = y
y.right = x
x.p = y
```

13.2-2

In every n-node binary search tree, there are $n-1$ edges. For every edge, it can left-rotate or right-rotate. So it have exactly $n-1$ possible rotations.

13.2-3

a's depth increase 1.

b's depth keep unchanged.

c's depth decrease 1.

13.2-4

For any left child, we use a right rotate to change it into a right child. Use at most $n-1$ right rotate, we will have the right chain. For the aim binary tree, similarily we could use $n-1$ right rotate to the same right chain. But this time we take the reverse action -- the left rotate. So totally we could take $O(n)$ time to transform.

13.2-5

2(1,) can't right rotate to 2(,1)

By at most $O(n)$ running time, change $T_1$ root node to $T_2$ root node. Recursively do it, in $O(n^2)$ running time we can finish the transform.

## Insertion

13.3-1

The property 5 is violated, and it is harder to deal with it.

13.3-2

38(19R(12(8R,),31),41)

13.3-3

All not change.

13.3-4

The only way to make the $T.nil.color$ to RED is line 7,14,21,28. $z$ must be the root or the child of root. If $z$ is the root, line 1 is false, skip the loop. If $z$ is the child of root, line 1 is also false. So there are no way to change the $T.nil.color$ to RED.

13.3-5

In case 1, 3 red nodes transform to 2 red nodes. In case 2 and 3, 2 red nodes transform to 2 red nodes. After any case, there are at least 2 red nodes. If one of them is the root node which change to black, there are still at least 1 red node.

13.3-6

When insert the $z$, following the path use a link list to storage the parent of nodes.

## Deletion

13.4-1

For case a and b, y is z, which means z is a red node. Removing a red node not effect black-height.

For case c, because y is a red node, x can only be T.nil. After y replacing z, the black-height of l and x is not changed.

For case d, the black-height of l and r is not changed too.

13.4-2

If the while loop not run, then root's color remain unchanged. If the while loop run, then it terminates at case 2 or case 4. In case 4, x is the root, and line 44 change the root color to BLACK. In case 2, if x is the root, the line 44 change the color, otherwise the root color remain unchanged.

13.4-3

If x is red, then skip the while loop and change x to black. The property 4 is restored.

13.4-4

38(19R(12(8R,),31),41)
38(19R(12,31),41)
38(19(,31R),41)
38(31,41)
38(,41R)
41

13.4-5

$x$ may be the T.nil, so line 2 examine it. $w$ can't be the T.nil, otherwise property 5 is not satisfied.

13.4-6

(a)

$$ count(\alpha) = count(\beta) = 3 $$

$$ count(\gamma) = count(\delta) = count(\varepsilon) = count(\zeta) = 2 $$

(b)

$$ count(\alpha) = count(\beta) = count(c) + 2 $$

$$ count(\gamma) = count(\delta) = count(\varepsilon) = count(\zeta) = count(c) + 2 $$

(c)

$$ count(\alpha) = count(\beta) = count(\varepsilon) = count(\zeta) = count(c) + 2 $$

$$ count(\gamma) = count(\delta) = count(c) + 1 $$

(d)

$$ count(\alpha) = count(\beta) = count(c) + 2 $$

$$ count(\gamma) = count(\delta) = count(c) + count(c') + 1 $$

$$ count(\varepsilon) = count(\zeta) = count(c) + 1 $$

13.4-7

If $x.p$ is RED, and $w$ is RED, then two RED are adjacient, which can't be possible.

13.4-8

No, Figure 13.4 is a good example.

13.4-9

A red-black tree is a binary tree, based on 12.2-9, we could first search a, then use successor algorithm for $m$ times.

## Problems

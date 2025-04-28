# Binary Search Trees

## What is a binary search tree

12.1-1

10(4(1,5),17(16,21))

10(5(4(1,),),17(16,21))

16(10(5(4(1,),),),17(,21))

17(16(10(5(4(1,),),),),21)

21(17(16(10(5(4(1,),),),),))

12.1-2

In binary-search-tree, node key value is smaller than right child, and larger than left child. But in min-heap, node key value is larger than both child.

Can't. MIN-HEAPIFY takes $O(\lg n)$, print in sorted order like heapsort, need $O(n\lg n)$.

12.1-3

```c
ITERATE-INORDER-TREE-WALK(x)
x = TREE-MINIMUM(x)
while x != NIL
    print x.key
    x = TREE-SUCCESSOR(x)
```

12.1-4

```c
PREORDER-TREE-WALK(x)
if x !- NIL
    print x.key
    PREORDER-TREE-WALK(x.left)
    PREORDER-TREE-WALK(x.right)

POSTORDER-TREE-WALK(x)
if x != NIL
    POSTORDER-TREE-WALK(x.left)
    POSTORDER-TREE-WALK(x.right)
    print x.key
```

12.1-5

As inorder tree walk takes $\Theta(n)$ running time, if constructing a binary search tree takes less then $\Omega(n\lg n)$ time, then we can first construct a binary search tree then use a inorder tree walk to sort, it totally takes less then $\Omega(n\lg n)$ time to sort, which can't be true.


## Querying a binary search tree

12.2-1

c cannot be true, 912 is in the left subtree of 911.

e cannot be true, 299 is in the right subtree of 347.

12.2-2

```c
RECURSIVE-TREE-MINIMUM(x)
if x.left == NIL
    return x
else return RECURSIVE-TREE-MINIMUM(x.left)

RECURSIVE-TREE-MAXIMUM(x)
if x.right == NIL
    return x
else return RECURSIVE-RIGHT-MAXIMUM(x.right)
```

12.2-3

```c
TREE-PREDECESSOR(x)
if x.left != NIL
    return TREE-MAXIMUM(x.left) // rightmost node in left subtree
else // find the lowest ancestor of x whose right child is an ancestor of x
    y = x.p
    while y != NIL and x == y.left
        x = y
        y = y.p
    return y
```

12.2-4

4(2(1,3),)

The path B is 4-2-1, $ 3 \in C$, however, $ 3 < 4 $.

12.2-5

Because the node has two children, so its successor is the minimum of the right child. if the successor has a left child, it can't be the minimum. Similarly to the predecessor.

12.2-6

If a subtree rooted at x has no right child, then it has finished walking in this subtree and need return to it's parent. Returning from a right child need to keep returning, as the parent node has been walked once until it find a parent that it is it's left child. This parent is its successor.

12.2-7

In TREE-SUCCESSOR algorithm, every edge most walk twice. There are $n-1$ edges in a tree, so the total time is $O(\lg n) + O(2(n-1)) = O(n)$.

12.2-8

We will walk through two kind of nodes, which is a successor or not a successor. If we walk from x to y, and z is the lowest common parent z. For parent nodes in path x->z, if x is in node's right subtree, it just walk up. Otherwise the node and the right subtree of the node will be walked through as successor. Similary, For parent nodes in path z->y, if x is in node's left subtree, it just walk down. Otherwise the node and the left subtree of the node will be walked through as successor. So nonsuccessors can only be in two paths, which at most $2h$. And successors is $k$, because any node can be walk through at most 3 times, so the total time is $O(k+h)$.

12.2-9

Use the TREE-SUCCESSOR and TREE-PREDECESSOR algorithm to solve it. Because x is a leaf node, so it has no child and it's successor and predecessor must in it's parents. If x is y's left child, then y is it's successor (the smallest key in T larger than x.key). else x is y's right child, then y is it's predecessor (the largest key in T smaller than x.key).


## Insertion and deletion

12.3-1
RECURSIVE-TREE-INSERT(T,z)
RECURSIVE-TREE-INSERT-SUBROUTE(T,T.root,NIL,z)

RECURSIVE-TREE-INSERT-SUBROUTE(T,x,y,z)
if x == NIL
    z.p = y
    if y == NIL
        T.root = z
    elseif z.key < y.key
        y.left = z
    else y.right = z
    return
if z.key < x.key
    x = x.left
else x = x.right
RECURSIVE-TREE-INSERT-SUBROUTE(T,x,z)

12.3-2

After inserting, the path from root to the node will be unchange. So the searching path is just the inserting path add the node.

12.3-3

The worst-case running time happen when insert by increasing order or decreasing order. It takes $O(n^2)$ time in insert all elements. The best-case running time happen when the tree is balanced. It takes $O(n\lg n)$ time.

12.3-4

When z have no children.

12.3-5

It's not commutative.

For tree 1(2,4(3,)), if we first delete 1, then delete 2, we get 3(,4). However, if we first delete 2, then delete 1, we get4(3,)

12.3-6

```c
SUCC-TREE-SEARCH(x, k)
while x != NIL and k != x.key
    if k < x.key
        x = x.left
    else x = x.right
return x

SUCC-TREE-INSERT(T, z)
x = T.root
y = NIL
s = NIL
while x != NIL
    y = x
    if z.key < x.key
        s = x
        x = x.left
    else x = x.right
z.succ = s
if y == NIL
    T.root = z
elseif z.key < y.key
    y.left = z
else y.right = z

SUCC-TREE-PARENT(T, x)
if x == T.root
    return NIL
y = TREE-MAXIMUM(x).succ
if y == NIL
    y = T.root
elseif y.left == x
    return y
else y = y.left
while y.right != x
    y = y.right
return y

SUCC-TRANSPLANT(T,u,v)
p = SUCC-TREE-PARENT(T,u)
if p == NIL
    T.root = v
elseif u == p.left
    p.left = v
else p.right = v

SUCC-TREE-PREDECESSOR(T,x)
if x.left != NIL
    return TREE-MAXIMUM(x.left) // rightmost node in left subtree
else // find the lowest ancestor of x whose right child is an ancestor of x
    y = T.root
    p = NIL
    while y != NIL
        if y.key == x.key
            return p
        elseif y.key < x.key
            p = y
            y = y.right
        else
            y = y.left

SUCC-TREE-DELETE(T,z)
p = SUCC-TREE-PREDECESSOR(T,z)
if p != NIL
    p.succ = z.succ
if z.left == NIL
    SUCC-TRANSPLANT(T,z,z.right)
elseif z.right == NIL
    SUCC-TRANSPLANT(T,z,z.left)
else y = TREE-MINIMUM(z.right)
    if y != z.right
        TRANSPLANT(T,y,y.right)
        y.right = z.right
    TRANSPLANT(T,z,y)
    y.left = z.left
```

12.3-7

```c
PREDECESSOR-TREE-DELETE(T,z)
if z.left == NIL
    TRANSPLANT(T,z,z.right)
elseif z.right == NIL
    TRANSPLANT(T,z,z.left)
else y = TREE-MAXIMUM(z.left)
    if y != z.left
        TRANSPLANT(T,y,y.left)
        y.left = z.left
        y.left.p = y
    TRANSPLANT(T,z,y)
    y.right = z.right
    z.right.p = y

FAIR-TREE-DELETE(T,z)
r = RANDOM(0,1)
if r == 0
    TREE-DELETE(T,z)
else PREDECESSOR-TREE-DELETE(T,z)
```


## Problems


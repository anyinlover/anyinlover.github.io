# Minimum Spanning Trees

## Growing a minumum spanning tree

21.1-1

Let $A$ be a subset of $E$ that is included in some minimum spanning tree $T$ and contains only $u$ but not $v$. Let $(S,V-S)$ be any cut of $G$ that respects $A$. Here $(u,v)$ become a light edge cross $(S,V-S)$, by theorem 21.1, $(u,v)$ is a safe edge and is included in some minimum spanning tree.

21.1-2

For $G=(\{a,b,c\}, \{(a,b),(a,c)\})$ and $w(a,b) < w(a,c)$, for cut $(\{a\},\{b,c\})$, we have the two safe edges cross the cut. However, $(a,c)$ is not a light edge.

21.1-3

Cut the edge $(u,v)$. If $(u,v)$ is not the light edge crossing $(S,V-S)$, then there must have another edge $(x,y)$ that $w(x,y) < w(u,v)$. Therefore, replace $(u,v)$ by $(x,y)$

$$ w(T') = w(T) - w(u,v) + w(x,y) < w(T) $$

Then $T$ is not a minimum spanning tree. False.

21.1-4

$G=(\{a,b,c\}, \{(a,b),(a,c),(b,c)\})$ and $w(a,b) = w(a,c) = w(b,c) = 1$.

21.1-5

Let $T$ is a minimum spanning tree containing $e$. (If not, $T$ is the answer.) Let $e = (u,v)$. Cut the edge $e$, because $e$ is on some cycle, so the cut must cross the path $u-v$. Let cut cross $(x,y)$, then we have $w(x,y) \le w(u,v)$. Replacing $(u,v)$ by $(x,y)$, we get another minimum spanning tree.

21.1-6

By GENERIC-MST, in line 3 the edge is always unique. So the final minimum spanning tree is unique.

$G=(\{a,b,c\}, \{(a,b),(a,c),(b,c)\})$ and $w(a,b) = w(a,c) = 1, w(b,c) = 2$.

b,a,c
b-a,c
b-a-c

21.1-7

If it's not a tree, then there is a cycle. remove a edge on the cycle, all vertices are still connected. Because every edge weight is positive, after removing, the total weight decrease, so it can't be true.

$G=(\{a,b,c\}, \{(a,b),(a,c),(b,c)\})$ and $w(a,b) = w(a,c) = w(b,c) = -1$.

21.1-8

If $i$ is the first position that $w(e_i) \neq w(e'_i)$. Suppose $w(e) < w(e')$. $T'$ must not include $e_i$. Cut cross the edge, there are another edge $(x,y)$ that included in $T'$. If $w(x,y) < w(u,v)$, then $T$ is not a minumum spanning tree. If $w(x,y) > w(u,v)$, then $T'$ is not a minumum spanning tree. So $w(x,y) = w(u,v)$ and $w(e) = w(e')$

21.1-9

In GENERIC-MST, first we construct A until $A=T'$, if there is a $T''$ be a minimum spanning tree of $G'$. Replace $T''$ of $T'$, then w(T) can be smaller, and $T$ is not a minumum spanning tree. So it is not true.

21.1-10

By 21.1-3, There is a cut cross edge $(x,y)$, $(x,y)$ must be a light edge. decreasing the edge weight, the edge is still a light edge. So it is a safe edge to add as a minimum spanning tree.

21.1-11

Because $(u,v)$ is not in the $T$, there are must a path $u-v$, by DFS, we find the largest weight edge $(x,y)$. If $w(u,v) >= w(x,y)$, then don't change it. Otherwise, remove $(x,y)$ and add $(u,v)$.

## The algorithms of Kruskal and Prim

21.2-1

By 21.1-8, we know every $T$ has a same sorted list of edge weights. Let $L$ is the sorted list of edge for $T$. The only difference happen when two edges have the same weights. By put the edge in $L$ the first position of same weighted edges in Kruskal sorted list. We can sure the edge will finally be picked up.

21.2-2

```c
MST-MATRIX-PRIM(G,w,r)
for each vertex u \in G.V
    u.key  = \infty
    u.\pi = NIL
r.key = 0
Q = 0
for each vertex u \in G.V
    INSERT(Q,u)
while Q != 0
    u = EXTRACT-MIN(Q)
    for each vertex v in G.V
        if v \in Q and 0 < G.A[u,v] < v.key
            v.\pi = u
            v.key = G.A[u,v]
            DECREASE-KEY(Q,v,G.A[u,v])
```

Here, DECREASE-KEY run $|V|^2$ times, use a Fibonacci heap, it takes $O(1)$, so the total running time is $O(V^2 + V\lg V) = O(V^2)$.

21.2-3

When $|E| = \Theta(V)$, the runing time of binary-heap is $O(E \lg V) = O(V \lg V)$, the running time of Fibonacci heap is $O(E + V \lg V) = O(V + V \lg V) = O(V \lg V)$. There are same.

When $|E| = \Theta(V^2)$, the runing time of binary-heap is $O(E \lg V) = O(V^2 \lg V)$, the running time of Fibonacci heap is $O(E + V \lg V) = O(V^2 + V \lg V) = O(V^2)$. The latter is smaller.

When $|E| = \omega(|V|)$.

21.2-4

We can use the counting method to sort. Here the total running time is $O(V+O(E+V)+E \alpha(V)) = O(E\alpha(V))$.

Using the same way, we have $O(V+O(E+W)+E \alpha(V)) = O(E\alpha(V))$.

21.2-5

It's a hard problem. Refered by Instructor's Manual.

If the edge weights is 1 to $|V|$, we can use van Emde Boas trees which give an upper bound of $O(E + V\lg \lg V)$.

If the edge weights is 1 to $|W|$, we can use an array of doubly linked list. Then EXTRAC-MIN takes only $O(W) = O(1)$, and DECREASE-KEY takes $O(1)$ by removing it and add it with new key in the slot. So the total running time is $O(E)$.

21.2-6

Wrong.

$G=(\{a,b,c\}, \{(a,b),(a,c),(b,c)\})$ and $w(a,b) = w(a,c) = 1, w(b,c) = 2$.

If the partition is a and b,c, then the result is not a minimum spanning tree.

21.2-7

Kruskal, we can use bucket sort to sort it in $O(E)$ time, and make the final running time $O(E\alpha(V))$.

21.2-8

It's a hard problem. Refered by Instructor's Manual.

We want to prove the lemma:

Let $T$ be a minimum spanning tree of $G = (V,E)$, and consider a graph $G' = (V', E')$ for which $G$ is a subgraph. Let $\bar{T} = E - T$ be the edges of $G$ that are not in $T$. Then there is a minimum spanning tree of $G'$ that includes no edges in $\bar{T}$.

We use loop invariant to prove the claim:

For any pair of vertices $u,v \in V$, if these vertices are in the same set after Kruskal's algorithm run on $G$ considers any edge $(x,y) \in E$, then they are in the same set after Kruskal's algorithm run on $G'$ considers $(x,y)$.

So if some edge $(u,v) \in \bar{T}$ is placed into $T'$, it contradicts the claim. So in $T'$ there are no edges in $\bar{T}$.

Using the lemma, we could build the graph $G'' = (V', E'')$, where $E''$ consists of the edges of $T$ and the edges in $E' - E$. Then find a minimum spanning tree $T'$ for $G''$, which is the result of $G'$ too.

Using Prim's algorithm with a Fibonacci-heap, we have $|V'| = |V|+1$, $|E''| \le 2|V| - 1$. So the running time is $O(E'' + V' \lg V') = O(V \lg V)$.

## Problems

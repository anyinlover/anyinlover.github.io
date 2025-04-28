# Elementary Graph Algorithms

## Representations of graphs

20.1-1

out-degree: $\Theta(|V|+|E|)$

in-degree: $\Theta(|V|+|E|)$

20.1-2

1: 2->3
2: 1->4->5
3: 1->6->7
4: 2
5: 2
6: 3
7: 3

  1 2 3 4 5 6 7
1 0 1 1 0 0 0 0
2 1 0 0 1 1 0 0
3 1 0 0 0 0 1 1
4 0 1 0 0 0 0 0
5 0 1 0 0 0 0 0
6 0 0 1 0 0 0 0
7 0 0 1 0 0 0 0

20.1-3

```c
TRANSPOSE-LIST(G)
let G'.Adj a new |V| array of empty lists.
for u = 1 to |V|:
    v = G.Adj[u]
    while v != NIL
        Insert(G'.Adj[v],u)
        v = v.next
return G'
```

It takes $\Theta(|V|+|E|)$ running time.

```c
TRANSPOSE-MATRIX(G)
let G'.A a empty |V|*|V| matrix
for i = 1 to |V|
    for j = 1 to |V|
        if G.A[i,j] = 1
            G'.A[j,i] = 1
return G'
```

It takes $\Theta(|V|^2)$ running time.

20.1-4

```c
SIMPLE-GRAPH(G)
let G'.Adj a new |V| array of empty lists.
let T[1:|V|] be a new list.
for u = 1 to |V|:
    for v = 1 to |V|:
        T[v] = false
    v = G.Adj[u]
    while v != NIL
        if v != u and !T[v]
            Insert(G'.Adj[u],v)
            T[v] = true
return G'
```

20.1-5

```c
SQUARE-GRAPH-LIST(G)
let G^2.Adj a new |V| array of empty lists.
for u = 1 to |V|:
    v = G.Adj[u]
    while v != NIL
        Insert(G^2.Adj[u],v)
        w = G.Adj[v]
        while w != NIL
            Insert(G^2.Adj[u],w)
            w = w.next
        v = v.next
return G^2
```

The running time is $O(|V||E|+|V|) = O(|V||E|)$.

```c
SQUARE-GRAPH-MATRIX(G)
let G^2.A a empty |V|*|V| matrix
for i = 1 to |V|
    for j = 1 to |V|
        if G.A[i,j] == 1
            G^2.A[i,j] = 1
            for k = 1 to |V|
                if G.A[j,k] == 1
                    G^2.A[i,k] = 1
return G^2
```

The running time is $O(|V|^3)$, as a dense matrix mulptily, we can optimize it using STRASSEN. As a sparse matrix mulptily, maybe there are better ways.

20.1-6

```c
IS-SINK(A, k)
    let A be |V| × |V|
    for j = 1 to |V|    // check for a 1 in row k
        if a[k][j] == 1
            return FALSE
    for i = 1 to |V|    // check for an off-diagonal 0 in column k
        if a[i][k] == 0 and i ≠ k
            return FASLE
    return TRUE

UNIVERSAL-SINK(A)
    let A be |V| × |V|
    i = j = 1
    while i ≤ |V| and j ≤ |V|
        if a[i][j] == 1
            i = i + 1
        else j = j + 1
    if i > |V|
        return "there is no universal sink"
    else if IS-SINK(A, i) == FASLE
        return "there is no universal sink"
    else return i "is a universal sink"
```

20.1-7

It's a $|V|*|V|$ matrix, where diagonal represents the vertex's sum of in degrees and out degrees, -1 in other positions represents two vertexs are linked, 0 represents are not linked.

20.1-8

Based on Theorem 11.2, the search time is:

$$
\Theta(1+\alpha) = \Theta(1+|E|/|V|)
$$

## Breadth-first Search

20.2-1

3 0 NIL
5 1 3
6 1 3
4 2 5
2 3 4
1 $\infty$ NIL

20.2-2

u 0 NIL
s 1 u
t 1 u
y 1 u
r 2 s
v 2 s
x 2 y
w 3 r
z 3 x

20.2-3

```c
BFS(G,s)
for each vertex u \in G.V - {s}
    u.color = true
    u.d = \infty
    u.\pi = NIL
s.color = false
s.d = 0
s.\pi = NIL
Q = 0
ENQUEUE(Q,s)
while Q != 0
    u = DEQUENUE(Q)
    for each vertex v in G.Adj[u]
        if v.color
            v.color = false
            v.d = u.d + 1
            v.\pi = u
            ENQUEUE(Q,v)

BFS(G,s)
for each vertex u \in G.V - {s}
    u.d = \infty
    u.\pi = NIL
s.d = 0
s.\pi = NIL
Q = 0
ENQUEUE(Q,s)
while Q != 0
    u = DEQUENUE(Q)
    for each vertex v in G.Adj[u]
        if v.d == \infty
            v.d = u.d + 1
            v.\pi = u
            ENQUEUE(Q,v)
```

20.2-4

```c
BFS(G,s)
let D[1:|V|] be a new list
let P[1:|V|] be a new list
for i = 1 to |V|
    D[i] = \infty
    P[i] = NIL
D[s] = 0
Q = 0
ENQUEUE(Q,s)
while Q != 0
    u = DEQUENUE(Q)
    for v = 1 to |V|
        if G.A[u,v] == 1 and D[v] == \infty
            D[v] = D[u] + 1
            P[v] = u
            ENQUEUE(Q,v)
```

20.2-5

Based on Theorem 20.5, after termination, we have $u.d = \delta(s,u)$, the discovery order will not effect $\delta(s,u)$, so as to $u.d$.

In the book, the breadth-first tree is:

s(r(t,w(x,z)),u(y),v)

Using another discovery order, the tree is:

s(v(y(x),w(z)),u(t),r)

20.2-6

s: u->v
u: w->x
v: w->x

$E_\pi = \{(s,u),(u,w),(s,v),(v,x)\}$ can't get from BFS.

20.2-7

First we may need to find the groups of connected graphs, then for each group, use the BFS.

```c
BFS(G,s)
for each vertex u \in G.V - {s}
    u.d = 0
s.d = 1
Q = 0
ENQUEUE(Q,s)
while Q != 0
    u = DEQUENUE(Q)
    for each vertex v in G.Adj[u]
        if v.d == 0
            v.d = -u.d
            ENQUEUE(Q,v)
        elseif v.d == u.d
            return false
return true
```

20.2-8

```c
FIND-ROOT(T)
let R[1:|V|] be a new list
for each vertex in R
    R[u] = true
for each vertex u \in G.V
    for each vertex v in G.Adj[u]
        R[v] = false

for each vertex in R
    if R[u] == true
        return u

BFS(G,s)
for each vertex u \in G.V - {s}
    u.d = \infty
s.d = 0
max = 0
Q = 0
ENQUEUE(Q,s)
while Q != 0
    u = DEQUENUE(Q)
    for each vertex v in G.Adj[u]
        if v.d == \infty
            v.d = u.d + 1
            max = v.d
            ENQUEUE(Q,v)

return max

DIAMETER(T)
r = FIND-ROOT(T)
return BFS(T,r)
```

## Depth-first Search

20.3-1

| DIRECTED | WHITE | GRAY | BLACK |
| -- | -- | -- | -- |
| WHITE | T/B/F/C | B/C | C |
| GRAY | T/F | T/B/F | T/F/C |
| BLACK | N | B | T/B/F/C |

| UNDIRECTED | WHITE | GRAY | BLACK |
| -- | -- | -- | -- |
| WHITE | T/B/ | T/B | N |
| GRAY | T/B | T/B | T/B |
| BLACK | N | T/B | T/B |

20.3-2

q 1/16
r 17/20
s 2/7
t 8/15
u 18/19
v 3/6
w 4/5
x 9/12
y 13/14
z 10/11

T: (q,s) (s,v) (v,w) (q,t) (t,x) (x,z) (t,y) (r,u)
B: (w,s) (z,x) (y,q)
F: (q,w)
C: (r,y) (u,y)

20.3-3

(u (v (y (x x) y) v) u) (w (z z) w)

20.3-4

```c
DFS(G)
for each vertex u \in G.V
    u.color = true
    u.\pi = NIL
time = 0
for each vertex u \in G.V
    if u.color
        DFS-VISIT(G,u)

DFS-VISIT(G,u)
time = time + 1
u.d = time
u.color = false
for each vertex v in G.Adj[u]
    if v.color
        v.\pi = u
        DFS-VIST(G,v)
time = time + 1
u.f = time
```

20.3-5

a.

In four edge types, only tree edges and forward edges $v$ is a descendant of $u$. By Corollary 20.8 we know that $v$ is a descendant of $u$ if and only if $u.d < v.d < v.f < u.f$. So we prove it.

b.

In four edge types, only back edges $v$ is an ancestor of $u$. By Theorem 20.7 we know that $v$ is an ancestor of $u$ if and only if $v.d < u.d < u.f < v.f$.

c.

First we prove the if. $[u.d,u.f]$ and $[v.d,v.f]$ are disjoint, so v is neither a descendant nor an ancestor of $u$. It must be Cross edge.
Then we prove the only if. If it is a cross edge, then v is neither a descendant nor an ancestor of $u$, and $[u.d,u.f]$ and $[v.d,v.f]$ are disjoint. Because $v$ is BLACK, so it has finished. So we have $v.d < v.f < u.d < u.f$.

20.3-6

```c
DFS-Stack(G)
for each vertex u in G.V
    u.color = false
    u.pi = NIL

time = 0

for each vertex u in G.V
    if u.color == false
        stack.push(u)
        u.color = true
        u.d = time
        time = time + 1
        while stack is not empty
            current = stack.top()
            allAdjacentVisited = true
            for each v in G.Adj[current]
                if v.color == false
                    stack.push(v)
                    v.color = true
                    v.pi = current
                    v.d = time
                    time = time + 1
                    allAdjacentVisited = false
                    break
            if allAdjacentVisited
                stack.pop()
                current.f = time
                time = time + 1
            
```

20.3-7

w: u->v
u: w

DFS from $w$, we have $u.d < v.d$, however, $v$ is not $u$'s descendant.

20.3-8

w: u->v
u: w

DFS from $w$, we can have $v.d > u.f$.

20.3-9

```c
DIRECTED-PRINT-EDGES(G)
for each vertex u \in G.V
    u.color = WHITE
    u.\pi = NIL
time = 0
for each vertex u \in G.V
    if u.color == WHITE
        DFS-VISIT(G,u)

DFS-VISIT(G,u)
time = time + 1
u.d = time
u.color = GRAY
for each vertex v in G.Adj[u]
    if v.color == WHITE
        print Tree(u,v)
        v.\pi = u
        DFS-VIST(G,v)
    elseif v.color == GRAY
        print BACK(u,v)
    elseif v.d < u.d
        print CROSS(u,v)
    else print FORWARD(u,v)

time = time + 1
u.f = time
u.color = BLACK
```

```c
UNDIRECTED-PRINT-EDGES(G)
for each vertex u \in G.V
    u.color = WHITE
    u.\pi = NIL
time = 0
for each vertex u \in G.V
    if u.color == WHITE
        DFS-VISIT(G,u)

DFS-VISIT(G,u)
time = time + 1
u.d = time
u.color = GRAY
for each vertex v in G.Adj[u]
    if v.color == WHITE
        print Tree(u,v)
        v.\pi = u
        DFS-VIST(G,v)
    else print BACK(u,v)

time = time + 1
u.f = time
u.color = BLACK
```

20.3-10

It depends on the first vertex choice.

v->u->w

If we first choose w, then u, then v, finally every dfs tree has only one vertex.

20.3-11

A DFS algorithm traverses each edge exactly once in each direction.

Use a DFS algorithm start from entrance, at every visited edge start point use a penny to indicate it. Never go the edge which start point has a penny. Stop the finding until we get the outdoor.

20.3-12

```c
DFS(G)
for each vertex u \in G.V
    u.color = true
    u.\pi = NIL
time = 0
cc = 0
for each vertex u \in G.V
    if u.color
        cc = cc + 1
        DFS-VISIT(G,u)

DFS-VISIT(G,u)
time = time + 1
u.d = time
u.color = false
u.cc = cc
for each vertex v in G.Adj[u]
    if v.color
        v.\pi = u
        DFS-VIST(G,v)
time = time + 1
u.f = time
```

20.3-13

1. Use Tarjan's algorithm to find the strongly connected components (SCCs) of the directed graph.
2. For each SCC, check if it has only one entry point and one exit point. If any SCC has more than one entry or exit point, then the graph is not singly connected.

## Topological sort

20.4-1

p 27/28
n 21/26
o 22/25
s 23/24
m 1/20
r 6/19
y 9/18
v 10/17
x 15/16
w 11/14
z 12/13
u 7/8
q 2/5
t 3/4

20.4-2

```c
COUNT-PATH(G,a,b)
DFS(G)
for each vertex u \in G.V
    u.count = 0
    u.color = true
count = 0
b.count = 1
return DFS-VISIT(G,a)

DFS-VIST(G,u)
u.color = false
for each vertex v in G.Adj[u]
    if v.color
        u.count = u.count + v.count
    elseif v.f > b.f 
        u.count = u.count + DFS-VISIT(G,v)
return u.count
```

20.4-3

We can run DFS, if we find a back edge, there's a cycle.

The running time is $O(V)$ because If we have seen $|V|$ distinct edges. There is must a back edge. Because by theorem B.2, in an acyclic forest, $|E| \le |V| - 1$.

20.4-4

False

For graph

a: b->d
b: c
c: a
d: c

DFS from c, we have:

c 1/8
a 2/7
d 5/6
b 3/4

bad edges are (b,c) (d,c)

DFS from a, we have:

a 1/8
d 6/7
b 2/5
c 3/4

bad edge is (c,a)

So DFS not always minimizes the number of bad edges.

20.4-5

```c
TOPOLOGICAL-SORT-DEGREE(G)
for each vertex u \in G.V
    u.degree = 0
for each vertex u \in G.V
    for each vertex v \in G.Adj[u]
        v.degree = v.degree + 1
Q = 0
for each vertex u \in G.V
    if u.degree == 0
        ENQUEUE(Q,u)
while Q != 0
    u = DEQUEUE(Q)
    print u
    for each vertex v \in G.Adj[u]
        v.degree = v.degree - 1
        if v.degree == 0
            ENQUEUE(Q,v)
```

If G has cycles, then the vertex in cycles will never go into the queue and never been printed.

## Strongly connected components

20.5-1

Keep unchanged or decrease one.

If the edge is in one scc, then nothing happen.

If the edge is between two sccs and there is already an edge from A to B, then nothing happen.

If the edge is between two sccs and there is already an edge from B to A, then the number decrease 1.

20.5-2

q 1 16
r 17 20
s 2 7
t 8 15
u 18 19
v 3 6
w 4 5
x 9 12
y 13 14
z 10 11

r
u
q t y
x z
s w v

20.5-3

False.

w: u->v
u: w

When we start from w, we have u,v,w, then start from u, we have only one tree, which is not right.

20.5-4

If there is a edge in $(G^T)^{SCC}$ from $C$ to $C'$, which mean there are vertices $u \in C$ and $v \in C'$ that $u->v$. So in $G$ there is a path $v->u$. $G$ and $G^T$ have same sccs. So the edge become a edge connect from $C'$ to $C$ in $G^{SCC}$.

20.5-5

First add a $u.scc$ to represent the scc id.

Then construct a multiset $T = \{u.scc: u\in V\}$ and sort it by using counting sort. Select distinct element and add it to $V^{SCC}$. It takes $O(V)$ time.

Then construct a multiset of ordered pairs $S = \{(x,y): \text{there is an edge }(u,v) \in E, x=u.scc,\text{ and }y = v.scc\text{ and }u.scc \neq v.scc\}$. Sort it by using radix sort. Select distinct pairs and add it to $E^{SCC}$. It takes $O(E+V)$ time.

So the total running time is $O(E+V)$.

20.5-6

We want to make edges in a scc become a simple cycle. And remove redundant edges between sccs.

1. Identify all sccs, takes $\Theta(V+E)$.
2. Form the component graph $G^{SCC}$, takes $\Theta(V+E)$.
3. For each scc, let the vertices in the scc be $v_1,v_2,\cdots,v_k$. Add the direct edges $(v_1,v_2),(v_2,v_3),\cdots,(v_{k-1}, v_k), (v_k, v_1)$, takes $O(V)$.
4. For each edge $(u,v)$ in the component graph $G^{SCC}$, select any vertex $x$ in $u$ and any vertex $y$ in $v$, add the directed edge $(x,y)$ to $E'$, takes $O(E)$.

So the total time is $\Theta(V+E)$.

20.5-7

The basic idea here is that the sequence of vertices given by topological sort forms a linear chain in the component graph.

1. Call STRONGLY-CONNECTED-COMPONENTS
2. Form the component graph.
3. Topologically sort the component graph.
4. Verify that the sequence of vertices $<v_1,v_2,\cdots,v_k>$ given by topological sort forms a linear chain in the component graph. That is ,verify that the edges $(v_1,v_2),(v_2,v_3),\cdots,(v_{k-1},v_k)$ exist in the component graph.

20.5-8

1. Form the component graph with min and max in every scc
2. Use dynamic programming find the longest path in the component graph
3. Traverse the longest path, for every other branch, calculate the processers's min value until reach the longest path node
4. Traverse the longest path, and get the processers' min value.
5. Traverse all nodes, and get the largest difference.

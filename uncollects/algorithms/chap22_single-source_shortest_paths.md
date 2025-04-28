# Single-source shortest paths

## The Bellman-Ford algorithm

22.1-1

s 2 z
t 8 s
x 7 z
y 9 s
z 0 NIL
s 2 z
t 5 x
x 6 y
y 9 s
z 0 NIL
s 2 z
t 4 x
x 6 y
y 9 s
z 0 NIL
s 2 z
t 4 x
x 6 y
y 9 s
z 0 NIL

s 0 NIL
t 6 s
x $\infty$ NIL
y 7 s
z $\infty$ NIL
s 0 NIL
t 6 s
x 4 y
y 7 s
z 2 t
s 0 NIL
t 2 x
x 4 y
y 7 s
z 2 t
s 0 NIL
t 2 x
x 2 z
y 7 s
z -2 t

Return false, because $t.d > x.d + w(x,t)$.

22.1-2

If there is a simple path from $s$ to $v$, let $s=v_0$. $v=v_k$, the path is $<v_0,v_1,\cdots,v_k>$, we know $|PATH| \le |V| - 1$. So after run BELLMAN-FORD, we have:

$$
v.d \le \sum_{i=1}^k w(v_{i-1}, v_i) < \infty
$$

If $v.d < \infty$, then it must be available from it's predecessor $v_{k-1}$ which has the property that $v_{k-1}.d < \infty$ too. By recursion, we know that $v$ is available from $s$, so there is a path from $s$ to $v$.

22.1-3

```c
BELLMAN-FORD(G,w,s)
INITIALIZE-SINGLE-SOURCE(G,s)
change = true
while change
    change = false
    for each edge (u,v) \in G.E
        RELAX(u,v,w)

RELAX(u,v,w)
if v.d > u.d + w(u,v)
    v.d = u.d + w(u,v)
    v.\pi = u
    change = true
```

22.1-4

```c
BELLMAN-FORD(G,w,s)
INITIALIZE-SINGLE-SOURCE(G,s)
for i = 1 to |G.V| - 1
    for each edge (u,v) \in G.E
        RELAX(u,v,w)
for each edge (u,v) \in G.E
    if v.d > u.d + w(u,v)
        LABEL_NEG(G,u)
        return FALSE
return TRUE

LABEL_NEG(G,u,s)
    u.d = -\infty
    for each vertex v \in G.Adj[u]
        if v != s and v.d != -\infty
            LABEL_NEG(G,v,s)
```

22.1-5

In BELLMAN-FORD algorithm, line 3 takes $O(|V|)$, and line 4 takes $O(|E|)$, line 5 takes $O(1)$, so the total running time is $O(|V||E|)$.

```c
BELLMAN-FORD(G,w,s)
INITIALIZE-SINGLE-SOURCE(G,s)
Create a list of |E| edges.
for i = 1 to |G.V| - 1
    for each edge (u,v) \in edge list
        RELAX(u,v,w)
for each edge (u,v) \in G.E
    if v.d > u.d + w(u,v)
        return FALSE
return TRUE
```

22.1-6

```c
BELLMAN-FORD(G,w)
for each vertex v \in G.V
    V.d = \infty
for i = 1 to |G.V|
    for each edge (u,v) \in G.E
        if v.d > min\{w(u,v),u.d + w(u,v)\}
            v.d = min\{w(u,v),u.d + w(u,v)\}
for each edge (u,v) \in G.E
    if v.d > min\{w(u,v),u.d + w(u,v)\}
    return false
return true
```

22.1-7

```c
BELLMAN-FORD(G,w,s)
INITIALIZE-SINGLE-SOURCE(G,s)
Create a list of |E| edges.
for i = 1 to |G.V| - 1
    for each edge (u,v) \in edge list
        RELAX(u,v,w)
for each edge (u,v) \in G.E
    if v.d > u.d + w(u,v)
        DFS-FIND(G,v,u)
return TRUE

DFS-FIND(G,v,u)
for each vertex w \in G.V
    w.color = true
    w.\pi = NIL
DFS-VISIT(G,v,u)
PRINT-CIRCLE(u)

DFS-VISIT(G,v,u)
    v.color = false
    for each vertex w in G.Adj[v]
        if w.color
            w.\pi = v
            if w == u
                return
            DFS-VIST(G,w)

PRINT-CIRCLE(u)
    w = u
    while w != NIL
        print w
        w = w.\pi
```

## Single-source shortest paths in directed acyclic graphs

22.2-1

r 0 NIL
s 5 r
t 3 r
x $\infty$ NIL
y $\infty$ NIL
z $\infty$ NIL
r 0 NIL
s 5 r
t 3 r
x 11 s
y $\infty$ NIL
z $\infty$ NIL
r 0 NIL
s 5 r
t 3 r
x 10 t
y 7 t
z 5 t
r 0 NIL
s 5 r
t 3 r
x 10 t
y 7 t
z 5 t
r 0 NIL
s 5 r
t 3 r
x 10 t
y 7 t
z 5 t
r 0 NIL
s 5 r
t 3 r
x 10 t
y 7 t
z 5 t

22.2-2

The last vertex has no out degrees.

22.2-3

In fact, If the weights are non-negative, the largest path is the link-list return by topological-sort.

Following considering that weights might be negative.

```c
DAG-LONGEST-PATHS(G,s)
topologically sort the vertices of G
INITIALIZE-SINGLE-SOURCE(G,s)
for each vertex u \in G.V, taken in topologically sorted order
    c = u.c + u.d
    for each vertex v in G.Adj[u]
        RELEX(u,v,c)

INITIALIZE-SINGLE-SOURCE(G,s)
for each vertex v \in G.V
    v.d = -\infty
    v.\pi = NIL
s.d = 0

RELAX(u,v,c)
if v.d < c
    v.d = c
    v.\pi = u

```

22.2-4

```c
COUNT-PATHS(G)
topologically sort the vertices of G
for each vertex u \in G.V
    u.p = 0

sum = |V|(|V|-1)/2 // the vertex pairs with 0 edges' path.
for each vertex u \in G.V, taken in reversed topologicaly sorted order
    for each vertex v in G.Adj[u]
        u.p = u.p + v.p + 1
    sum = sum + u.p

return sum
```

The running time is $\Theta(|V|+|E|)$

## Dijkstra's algorithm

22.3-1

s 0 NIL
t 3 s
x $\infty$ NIL
y 5 s
z $\infty$ NIL
s 0 NIL
t 3 s
x 9 t
y 5 s
z $\infty$ NIL
s 0 NIL
t 3 s
x 9 t
y 5 s
z 11 y
s 0 NIL
t 3 s
x 9 t
y 5 s
z 11 y
s 0 NIL
t 3 s
x 9 t
y 5 s
z 11 y

s 3 z
t $\infty$ NIL
x 7 z
y $\infty$ NIL
z 0 NIL
s 3 z
t 6 s
x 7 z
y 8 s
z 0 NIL
s 3 z
t 6 s
x 7 z
y 8 s
z 0 NIL
s 3 z
t 6 s
x 7 z
y 8 s
z 0 NIL

22.3-2

s: t(1) -> x(2)
t:
x: t(-3)

Because negative weight exists, we not have $\delta(s,y) \le \delta(s,u)$.

22.3-3

Yes. For every vertex $u$, $u.d = \delta(s,u)$,

22.3-4


22.3-5

```c
CHECK(G)
for each vertex u \in G.V
    for each vertex v \in G.Adj[u]
        if v.d < 0 or (v.d > 0 and v.d != v.\pi.d + w(v.\pi, v))
            return false
return true
```

22.3-11

In round 1, negative-weight not effect the vertex adjacient to $z$ get the shortest path. In after rounds, the Dijkastra still work.


22.3-12

```c
BFS(G,w,s)
for each vertex u \in G.V
    u.color = true
    u.d = \infty
    u.\pi = NIL
s.d = 0
Q = 0
ENQUEUE(Q,s)
while Q != 0
    u = DEQUEUE(Q)
    for each vertex v in G.Adj[u]
        RELAX(u,v,w)
        if v.color
            v.color = false
            ENQUEUE(Q,v)
```



## Difference constraints and shortest paths

## Proofs of shortest-paths properties

## Problems

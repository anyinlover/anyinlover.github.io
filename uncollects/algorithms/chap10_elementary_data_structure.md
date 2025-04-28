# Elementary Data Structure

## Simple array-based data structure

10.1-1

We can use the first 2 bits to represent the block, the next $\lg m - 1$ bits to represent the row in the block, and the next $\lg n - 1$ bits to represent the column in the block.

10.1-2

4,
4,1
4,1,3
4,1
4,1,8
4,1

10.1-3

We can build two stacks, one from left to right, and another one from right to left. Be sure to check if the left stack top is adjact to the right stack top.

10.1-4

4,
4,1
4,1,3
1,3
1,3,8
3,8

10.1-5

```c
ENQUEUE(Q,x)
if Q.head == Q.tail + 1 or (Q.head == 1 and Q.tail == Q.size)
    error "overflow"
Q[Q.tail] = x
if Q.tail == Q.size
    Q.tail = 1
else Q.tail = Q.tail + 1

DEQUEUE(Q)
if Q.head == Q.tail
    error "underflow"
x = Q[Q.head]
if Q.head == Q.size
    Q.head = 1
else Q.head = Q.head + 1
return x
```

10.1-6

```c
APPEND(D,x)
if D.head == D.tail + 1 or (D.head == 1 and D.tail == D.size)
    error "overflow"
D[D.tail] = x
if D.tail == D.size
    D.tail = 1
else D.tail = D.tail + 1

APPENDLEFT(D,x)
if D.head == D.tail - 1 or (D.head == 1 and D.tail == D.size)
    error "overflow"
if D.head == 1
    D.head = D.size
else D.head = D.head - 1
D[D.head] = x

POP(D)
if D.head == D.tail
    error "underflow"
if D.tail == 1
    D.tail = D.size
else D.tail = D.tail - 1
return D[D.tail]

POPLEFT(D)
if D.head == D.tail
    error "underflow"
x = D[D.head]
if D.head == D.size
    D.head = 1
else D.head = D.head + 1
return x
```

10.1-7

The queue ENQUEUE is like the stack PUSH, need $O(1)$ running time. The queue DEQUEUE can be done as following,which take $O(n)$ running time.

Keep popping elements and push them one by one into another stack until the first stack is empty. Then pop one element from the second stack as the dequeue element, and keep popping other elements and push them back into the first stack.

10.1-8

It is similarly to 10.1-7.

The stack PUSH is like the queue ENQUEUE, need $O(1)$ running time.
The stack POP can be done as following, which take $O(n)$ running time.

Keep dequeuing elements, if it's the last one then return it, else enqueue them into another queue.

## Linked lists

10.2-1

```c
SINGLE-LIST-INSERT(x,y)
x.next = y.next
y.next = x

SINGLE-LIST-DELETE(L,x)
if x == L.head
    L.head = x.next
i = L.head
while i.next != x
    i = i.next
i.next = x.next
```

10.2-2

No other infomation needed.

```c
STACK-EMPTY(S)
if S.head == NIL
    return TRUE
else return FALSE

PUSH(S,x)
x.next = S.head
S.head = x

POP(S)
if STACK-EMPTY(S)
    error "underflow"
else x = S.head
    S.head = x.next
    return x
```

10.2-3

```c
ENQUEUE(Q,x)
Q.tail.next = x
Q.tail = x

DEQUEUE(Q,x)
if Q.head == NIL
    return "underflow"
else x = Q.head
    Q.head = x.next
    return x
```

10.2-4

Use a circular, doubly linked list with a sentinel to represent a set, we can do union as following:

```c
UNION(S1,S2)
S1.nil.prev.next = S2.nil.next
S2.nil.next.prev = S1.nil.prev
S2.nil.prev.next = S1.nil
S1.nil.prev = S2.nil.prev
```

10.2-5

```c
REVERSE(L)
if L.head == NIL
    return
l = L.head
while l.next != NIL
    n = l.next
    l.next = n.next
    n.next = L.head
    L.head = n
```

10.2-6

```c
XOR-LIST-SEARCH(L,k)
x = L.head
px = NIL
while x != NIL and x.key != k
    tmp = px
    px = x
    x = x.np XOR tmp
return x

XOR-LIST-INSERT(L,x,y)
z = L.head
pz = NIL
while z != y
    tmp = pz
    pz = z
    z = z.np XOR tmp
ny = y.np XOR pz
x.np = y XOR ny
y.np = pz XOR x
if ny != NIL
    ny.np = ny.np XOR y XOR x

XOR-LIST-DELETE(L,x)
z = L.head
pz = NIL
while z != x
    tmp = pz
    pz = z
    z = z.np XOR tmp

nx = x.np XOR pz
if pz == NIL
    L.head = nx
else pz.np = pz.np XOR x XOR nx

if nx != NIL
    nx.np = nx.np XOR x XOR pz

XOR-LIST-REVERSE(L)
tmp = L.head
L.head = L.tail
L.tail = tmp
```

## Representing rooted trees

10.3-1

15(17(22,13(12,28)),20(25(,33(14,)),))

10.3-2

```c
BINARY-TREE-RECURSIVE-PRINT(T)
x = T.root
RECURSIVE-PRINT(x)

RECURSIVE-PRINT(x)
if x != NIL
    print(x.key)
    RECURSIVE-PRINT(x.left)
    RECURSIVE-PRINT(x.right)
```

10.3-3

```c
BINARY-TREE-NONRECURSIVE-PRINT(T)
Initialize empty stack S
S.push(T.root)
while not STACK-EMPTY(S)
    x = POP(S)
    if x != NIL
        print(x.key)
        S.push(x.left)
        S.push(x.right)
```

10.3-4

```c
TREE-RECURSIVE-PRINT(T)
x = T.root
RECURSIVE-PRINT(x)

RECURSIVE-PRINT(x)
if x != NIL
    print(x.key)
    RECURSIVE-PRINT(x.left-child)
    RECURSIVE-PRINT(x.right-sibling)
```

10.3-5

```c
TREE-RECURSIVE-PRINT(T)
x = T.root
up = FALSE
while x != T.root or not up
    if not up
        print(x.key)
        if x.left-child != NIL
            x = x.left-child
            up = FALSE
    if x.right-sibling != NIL
        x = x.right-sibling
        up = FALSE
    else x = x.p
        up = TRUE   
```

10.3-6

```c
PARENT(x)
while not x.is-sibling-parent
    x = x.right-sibling
return x.right-sibling

CHILD(x,i)
child = x.left-child
while i > 1 
    child = child.right-sibling
    i = i - 1
return child
```

## Problems

# Augmenting Data Structures

## Dynamic order statistics

17.1-1

26 17 21 19 20

17.1-2

1 + 2 + 13 = 16

17.1-3

```c
NONRECURSIVE-OS-SELECT(x,i)
r = x.left.size + 1
while i != r
    if i < r
        x = x.left
    else
        x = x.right
        i = i - r
    r = x.left.size + 1
return x
```

17.1-4

```c
OS-KEY-RANK(T,k)
x = TREE-SEARCH(T.root,k)
OS-RANK(T,x)
```

17.1-5

```c
SUCCESSOR-OS-SELECT(T,x,i)
k = OS-RANK(T,x)
return OS-SELECT(T.root, k+i)
```

17.1-6


## How to augment a data structure

## Interval trees

## Problems

# The Role of Algorithms in Computing

## Algorithms

What are algorithms?

An algorithm is any well-defined computational procedure that takes value as input and produces some value as output in a finite amount of time. It can be viewed as a tool for solving a well-specified computational problem.

There are many practical applications of algorithms, for example, The Human Genome Project, Internet, electronic commerce and manufacturing.

A data structure is a way to store and organize data in order to facilitate access and modifications. Choosing the appropriate data structure or structures is an important part of algorithm design.

Except addressing specific problems such as computing minimum spanning trees, the book discusses techniques of algorithm design and analysis like divide-and-conquer, dynamic programming and amortized analysis. These techniques can be used to develop new algorithms.

NP-complete problems are a set of problems that no efficient algorithm has ever been found. If we can show a problem is NP-complete, we can spend time developing an efficient approximation algorithm rather than find the best one.

For many years, we use computing models based on one processor. Nowadays multi-cores parallel computers are mainstream, and so parallel algorithms have been developed. Another kind of computing models are online algorithms, where not all data are available when an algorithm begins running, for example, task scheduling.

### Exercises

1.1-1

A real-world example that requires sorting: Search Engine

A read-world example that requires finding the shortest distance between two points: map navigation

1.1-2

the storage

1.1-3

vector, it's easy to read and append, but not efficient to insert

1.1-4

shortest-path and traveling-salesperson problems are both need to find the shortest path, when the first one need find only one point to another point, but the second need find the sum in a net.

1.1-5

In dense retrieval, the best solution is to find the most close embedding to the query embedding, but is it time-cost and we use ANN as an approximate algorithm.

1.1-6

Training task schedule, sometimes we need to train a batch of tasks, but other times the tasks are coming in continuesly.

## Algorithms as a technology

We should choose algorithms that use the resoures of time and space efficiently. Different algorithms often differ dramatically in their efficency, which can be much more significant than differences due to hardware and software.

Algorithms can be base for other technologies, like hardware designs and network routing.

### Exercises

1.2-1

notion, it can be used to search personal note, which are about string matching and relevance sorting.

1.2-2

n<=43

```python
from math import log2

n = 2
while n < 8 * log2(n):
    n += 1

print(n-1)
```

1.2-3

15

```python

n = 1
while 2**n <= 100 * n**2:
    n += 1

print(n)
```

## Problems

1.1

|           | 1 second | 1 minute | 1 hour   | 1 day      | 1 month     | 1 year        | 1 century     |
| --------- | -------- | -------- | -------- | ---------- | ----------- | ------------- | ------------- |
| $\lg n$   | 1.07e301 | inf      | inf      | inf        | inf         | inf           | inf           |
| $\sqrt n$ | 1e6      | 3.6e9    | 1.296e13 | 7.46496e15 | 6.718464e18 | 9.94519296e20 | 9.95827587e24 |
| $n$       | 1e3      | 6e4      | 3.6e6    | 8.64e7     | 2.592e9     | 3.1536e10     | 3.1556736e12  |
| $n \lg n$ | 140      | 4895     | 204094   | 3943234    | 97659289    | 1052224334    | 86842987896   |
| $n^2$     | 31       | 244      | 1897     | 9295       | 50911       | 177583        | 1776421       |
| $n^3$     | 10       | 39       | 153      | 442        | 1373        | 3159          | 14667         |
| $2^n$     | 9        | 15       | 21       | 26         | 31          | 34            | 41            |
| $n!$      | 6        | 8        | 9        | 11         | 12          | 13            | 15            |

```python
from scipy.special import factorial
from scipy.optimize import fsolve
import numpy as np
from functools import partial

bases = [1000] * 7
bases[1] = bases[0] * 60
bases[2] = bases[1] * 60
bases[3] = bases[2] * 24
bases[4] = bases[3] * 30
bases[5] = bases[3] * 365
bases[6] = bases[3] * 36524

print(np.power(2.0, bases))
print(np.power(bases, 2.0))
print(bases)

nlog = lambda n, base: n * np.log2(n) - base
print([int(fsolve(partial(nlog, base=base), 1000000)) for base in bases])

print(np.power(bases, 1/2).astype('int64'))
print(np.power(bases, 1/3).astype('int64'))
print(np.log2(bases).astype('int64'))
fact = lambda n, base: factorial(n) - base
print([int(fsolve(partial(fact, base=base), 15)) for base in bases])
```

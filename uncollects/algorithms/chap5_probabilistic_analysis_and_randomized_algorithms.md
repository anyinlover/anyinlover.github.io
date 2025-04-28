# Probabilistic Analysis and Randomized Algorithms

## The hiring problem

We have the following hiring problem:

```c
HIRE-ASSISTANT(n)
best = 0
for i = 1 to n
    interview candidate i
    if candidate i is better than candidate best
        best = i
        hire candidate i
```

Letting $m$ be the number of people hired, the total cost associated with this algorithm is $O(c_in + c_hm)$. $m$ depends on the order in which you interview candidates.

Probabilistic analysis is the use of probability in the analysis of problems. Most commonly, we use Probabilistic analysis to analyze the running time of an algorithm. In order to perform a probabilistic analysis, we must use knowledge of the distribution of the inputs. Then we analyze our algorithm, computing an average-case running time, where we take the average, or expected value over the distribution of the possible inputs. We refer to it as the average-case running time.

For the hiring problem, the input form a uniform random permutation.

We call an algorithm randomized if its behavior is determined not only by its input but also by values produced by a random-number generator. By this way, probability and randomness often serve as tools for algorithm design and analysis, by making part of the algorithm behave randomly. We refer to it as the expected running time.

### Exercises

5.1-1

```c
HIRE-ASSISTANT(n)
best = 0
for i = 1 to n
    interview candidate i
    if candidate i is the best one
        best = i
        hire candidate i
```

5.1-2

```c
RANDOM(a,b)
n = \lceil \lg (b-a+1) \rceil
Initialize an array A of length n
for i = 1 to n do
    A[i] = RANDOM(0, 1)
if A holds the binary representation of number from a through b then
    return number represents by A
else
    RANDOM(a,b)
```

Because every time A holds the binary representation of number from a through b is $p = (b-a+1)/2^n$, so the expected running time is:

$$
n \sum_{i=0}^\infty i(1-p)^i p = n/p = \Theta(\lg(b-a))
$$

5.1-3

```
UNBIASED-RANDOM
while TRUE
    x = BIASED-RANDOM
    y = BIASED-RANDOM
    if x \ne y
        return x
```

By these way, $P(0) = (1-p)p$, $P(1) = p(1-p)$. So its same chance. The running time is the time call BIASED-RANDOM.

For any two BIASED-RANDOM, we get the result probability is $2p(1-p)$, every unbiased-random trial is a Bernoulli trial, and together is a geometric distribution. By C.36. we have the expection $1/p = 1/2p(1-p)$, so the expect running time is $\Theta(1/2p(2-p))$.

## Indicator random variables

Indicator random variables provide a convenient method for converting between probabilities and expectations. Given a sample space $S$ and an event $A$, the indicator random variable $I\{A\}$ associated with event $A$ is defined as:

$$
I\{A\} = \begin{cases}
1 \text{ if } A \text{occurs} \\
0 \text{ if } A \text{does not occur}
\end{cases}
$$

Lemma 5.1

Given a sample space $S$ and an event $A$ in the sample space $S$, let $X_A = I\{A\}$. Then $E[X_A] = Pr\{A\}$.

Lemma 5.2

Assuming that the candidates are presented in a random order, algorithm HIRE-ASSISTANT has an average-case total hiring cost of $O(c_h \ln n)$.

### Exercises

5.2-1

Only when the best one is the first one, we hire exactly one time. $p = (n-1)!/n! = 1/n$.

When better one is the latter one, we hire exactly n times. $p = 1/n!$

5.2-2

If the first candidate is rank $i$, then candidate whose rank is $i+1$ to $n$ must behind the rank $n$ candidate. The first probability is $1/n$, the second probability is $1/{n-i}$, so we have the together probability is:

$$
Pr\{A\} = \sum_{i=1}^{n-1} \frac{1}{n-i} \frac{1}{n} = \frac{1}{n} \sum_{i=1}^{n-1} \frac{1}{n-i} = \frac{1}{n} H_{n-1}
$$

5.2-3

Letting $X_j$ be the indicator of a dice coming up $j$. So the expected value of a single dice roll $X$ is:

$$
E[X] = \sum_{j=1}^6 j Pr(X_j) = 7/2
$$

So the sum of n dice is:

$$
E[nX] = nE[X] = 3.5n
$$

5.2-4

When two dice are rolled independently, the expected value of the sum is 7.
When the second dice is rolled same as the first one, the expected value of the sum is 7.
When the second dice is rolled as 7 minus the first one, the expected value of the sum is 7.

5.2-5

Let $X_i$ is the indicator random variable that customer $i$ gets back his own hat. And let $X$ equals the number of customers that get back their own hat. So we have:

$$
X = X_1 + X_2 + \cdots + X_n
$$

Each customer has a probability of $1/n$ of getting back his hat, so $Pr\{X_i = 1\} = 1/n$. So we have:

$$
E[X] = E[\sum_1^n X_i] = \sum_1^nE[X_i] = \sum_1^n 1/n = 1
$$

5.2-6

Let $X_{ij}$ is the indicator random variable that pair $(i,j)$ is an inversion. we have $Pr\{X_{ij} = 1\} = 1/2$. Let $X$ equals the number of inversions. Then we have:

$$
X = \sum_{i=1}^{n-1} \sum_{j=i+1}^n X_{ij} = \frac{1}{2} \frac{n(n-1)}{2} = \frac{n(n-1)}{4}
$$

## Randomized Algorithms

```c
RANDOMIZED-HIRE-ASSSISTANT

randomly permute the list of candidates
HIRE-ASSISTANT(n)
```

Lemma 5.3
The expected hiring cost of the procedure RANDOMIZED-HIRE-ASSISTANT is $O(c_h \ln n)$

Following is a method to generate a random permutation permutes the array in place:

```
RANDOMLY-PERMUTE(A, n)
for i = 1 to n
    swap A[i] with A[RANDOM(i,n)]
```

Lemma 5.4
Procedure RANDOMLY-PERMUTE computes a uniform random permutation.

We can use loop invariant that the subarray $A[1:i-1]$ contains this $(i-1)$ permutation with probability $(n-i+1)!/n!$.

### Exercises

5.3-1

```
RANDOMLY-PERMUTE(A, n)
swap A[1] with A[RANDOM(1,n)]
for i = 2 to n
    swap A[i] with A[RANDOM(i,n)]
```

We just need change the Initialization of the loop invariant. the subarray $A[1:1]$ contains this 1-permutation with probability $(n-2+1)!/n! = 1/n$. Because we swap $A[1]$ with $A[RANDOM(1,n)]$, this 1-permutation indeed is $1/n$.

5.3-2

No, in this way $A[i]$ always put behind, and its not a permutation.

5.3-3

It's not a uniform random permutation. When $n=3$, The PERMUTE-WITH-ALL has 27 possible outcomes. But there are $3!=6$ permutations. If each permutation occurs $m$ times, so $ m/27 = 1/6$. It can't be solved.

5.3-4

Two near members are still near. So it's not true.

5.3-5

We can use the induction to solve it. It equals to prove that every num have same probability of $m/n$.

In base case, $m=1$, it's true. If every num have same probability in $m-1/n-1$, for $j < n$, we have:

$$
Pr(j \in S) = \frac{m-1}{n-1} + (1 - \frac{m-1}{n-1})*\frac{1}{n} = \frac{m}{n}
$$

For $n$, we have:

$$
Pr(n \in S) = \frac{1}{n} + (1 - \frac{1}{n})*\frac{m-1}{n-1} = \frac{m}{n}
$$

## Probabilistic analysis and further uses of indicator random variables

### Exercises

5.4-1

The first question is a geometric distribution, $p = 1/365$, so that:

$$
(1-p)^k \le 1/2
$$

We get $k \ge 253$

$$
(1-p)^k + k*p*(1-p)^(k-1) \le 1/2
$$

We get $k \ge 613$.

5.4-2

$$ Pr\{B_k\} = \prod_{i=0}^{k-1}(1-p*i) \le 0.01 $$

We have $k \ge 57$.

From the indicator random variable method, we have:

$$
E[X] = \frac{k(k-1)}{2n} = 4
$$

5.4-3

The problem is same to birthday paradox, we can use indicator random variable to solve it.

Which mean:

$$
E[X] = \frac{k(k-1)}{2b} \ge 1
$$

So the answer is $k = \lfloor \sqrt {2b} \rfloor + 1 $.

5.4-4

Pairwise independence is enough. In indicator random variable method, we just need $X_{ij}$ to be independent.

5.4-5

We can use indicator random variable method to solve it.

For any three people, they have the same birthday is $\frac{1}{n^2}$. So, 

$$
E[X] = \binom{k}{3} \frac{1}{n^2} = \frac{k(k-1)(k-2)}{6n^2}
$$

Let $E[X] \ge 1$, we have $k \ge 94$.

5.4-6

$Pr = \frac{n!}{(n-k)! n^k}$, it's the same as $Pr\{B_k\}$.

5.4-7

$$
E[X] = \sum_{i=1}^n E[X_i] = n(\frac{n-1}{n})^n
$$

$$
E[X] = \sum_{i=1}^n [X_i] = n \binom{n}{1}(\frac{n-1}{n})^{n-1} \frac{1}{n} = n(\frac{n-1}{n})^{n-1}
$$

5.4-8

We can use method in book, this time we use $s = \lg n - 2\lg \lg n$, so we have:

$$
Pr\{A_{i,\lg n - 2\lg \lg n}\} = \frac{1}{2^{\lg n - 2\lg \lg n}} = \frac{\lg^2 n}{n}
$$

There are $ m = \lfloor \frac{n}{\lg n - 2\lg \lg n} \rfloor $ groups, the probability that all groups fail to have the trail is:

$$
(1- \frac{\lg^2 n}{n})^m \le (1- \frac{\lg^2 n}{n})^{\frac{n}{\lg n - 2\lg \lg n}} \le e^{-\frac{\lg^2 n}{\lg n - 2\lg \lg n}} < \frac{1}{n}
$$

## Problems


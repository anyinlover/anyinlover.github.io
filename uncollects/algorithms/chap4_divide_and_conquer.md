# Divide and conquer

A recurrence $T(n)$ is algorithmic if, for every sufficiently large threshold constant $n_0 > 0$, the following two properties hold.

1. For all $n < n_0$, we have $T(n) = \Theta(1)$.
2. For all $n \le n_0$, every path of recursion terminates in a defined base case within a finite number of recursive invocations.

A recurrence $T(n)$ that represents a divide-and-conquer algorithm's worst-case running time satisfy these properties for all sufficiently large threshold constants.

We adopt the following convention: when a recurrence is stated without an explicit base case, we assume that the recurrence is algorithmic. And we often drop any floors or ceilings in a recurrence. We will prove for most cases it is safe for ignoring.

When recurrences are not equations, but rather inequalities, such as $T(n) \le 2T(n/2) + \Theta(n)$, we express its solution using $O$-notation. Similarly, we use $\Omega$-notation to solve $T(n) \ge 2T(n/2) + \Theta(n)$.

Sometimes it maybe divide a problem into subproblems of different sizes. Such as $T(n) = T(n/3) + T(2n/3) + \Theta(n)$, $T(n) = T(n/5) + T(7n/10) + \Theta(n)$, $T(n) = T(n-1) + \Theta(1)$.

## Multiplying square matrices

```c
MATRIX-MULTIPLY-RECURSIVE(A,B,C,n)
if n == 1
// Base case.
    c_{11} = c_{11} + a_{11}*b_{11}
    return
// Divide
partition A, B, and C into n/2 * n/2 submatrices
    A_{11}, A_{12}, A_{21}, A_{22}; B_{11}, B_{12}, B_{21}, B_{22};
    and C_{11}, C_{12}, C_{21}, C_{22}; respectively
// Conquer.
MATRIX-MULTIPLY-RECURSIVE(A_{11}, B_{11}, C_{11}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{11}, B_{12}, C_{12}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{21}, B_{11}, C_{21}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{21}, B_{12}, C_{22}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{12}, B_{21}, C_{11}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{12}, B_{22}, C_{12}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{22}, B_{21}, C_{21}, n/2)
MATRIX-MULTIPLY-RECURSIVE(A_{22}, B_{22}, C_{22}, n/2)
```

Use divide_and_conquer to solve the matrix multiply, we have the running time:

$$ T(n) = 8T(n/2) + \Theta(1) $$

Using the master method, we get $T(n) = \Theta(n^3)$.

### Exercises

4.1-1

When $n$ is odd, we can add one row and one column filled with 0 to calculate.

$$ T(n) = 8T(\lceil n/2 \rceil) + \Theta(1) $$

We can use the substitution method to prove it. First we want to prove $ T(n) = \Omega(n^3)$. If $T(n) \ge cn^3$ for all $n \ge n_0$, then:

$$
\begin{align*}
T(n) &= 8T(\lceil n/2 \rceil) + \Theta(1) \\
&\ge 8T(n/2) + \Theta(1) \\
&\ge cn^3 + \Theta(1) \\
&\ge cn^3
\end{align*}
$$

next we prove that $ T(n) = O(n^3). If $T(n) \le c(n-3)^3$ for all $n \ge n_0$, then when n is large enough:

$$
\begin{align*}
T(n) &= 8T(\lceil n/2 \rceil) + \Theta(1) \\
&\le 8T(n/2+1) + \Theta(1) \\
&\le 8c(n/2-2)^3 + \Theta(1) \\
&= c(n-4)^3 + \Theta(1) \\
&\le c(n-3)^3
\end{align*}
$$

So $T(n) \le (n-3)^3 < n^3 = O(n^3)$, and $T(n) = \Theta(n^3)$.

4.1-2

When multiply $kn * n$ matrix by an $n * kn$ matrix, it need $k^2 n*n$ matrix multiplies. So the worst running time is $\Theta(k^2 n^3)$, When multiply $n * kn$ matrix by an $kn * n$ matrix, it need $k n*n$ matrix multiplies. So the worst running time is $\Theta(k n^3)$, the latter is faster, by $k$,

4.1-3

$$ T(n) = 8T(n/2) + \Theta(n^2) $$

The solution is same, $ T(n) = \Theta(n^3) $.

4.1-4

```c
MATRIX-ADD-RECURSIVE(A,B,C,n)
if n == 1
// Base case.
    c_{11} = c_{11} + a_{11} + b_{11}
    return
// Divide
partition A, B, and C into n/2 * n/2 submatrices
    A_{11}, A_{12}, A_{21}, A_{22}; B_{11}, B_{12}, B_{21}, B_{22};
    and C_{11}, C_{12}, C_{21}, C_{22}; respectively
// Conquer.
MATRIX-ADD-RECURSIVE(A_{11}, B_{11}, C_{11}, n/2)
MATRIX-ADD-RECURSIVE(A_{12}, B_{12}, C_{12}, n/2)
MATRIX-ADD-RECURSIVE(A_{21}, B_{21}, C_{21}, n/2)
MATRIX-ADD-RECURSIVE(A_{22}, B_{22}, C_{22}, n/2)
```

$$ T(n) = 4T(n/2) + \Theta(1) $$

The solution is $ T(n) = \Theta(n^2) $.

When $ T(n) = 4T(n/2) + \Theta(n^2) $, The solution is $T(n) = n^2\lg n$.

## Strassen's algorithm for matrix multiplication

Strassen's algorithm runs in $\Theta(n^{\lg 7})$, which is better than $\Theta(n^3)$. By making the recursion tree less bushy, Strassen's algorithm performs only seven subproblem.

The trick is that matric multiplication is complex than addition. like $x^2 - y^2 = (x+y)(x-y)$, Strassen's algorithm is following:

1. If $n=1$, the matrices each contain a single element. Perform a calculation as MATRIX-MULTIPLY-RECURSIVE. Otherwise, partition into 4 submatrices.
2. Create $n/2*n/2$ matrices $S_1,S_2,\cdots,S_{10}$, and create and zero $n/2*n/2$ matrices $P_1,P_2,\cdots,P_7$ to hold matrix products. All can be done in $\Theta(n^2)$ time.
    $$
    S_1 = B_{12} - B_{22} \\
    S_2 = A_{11} + A_{12} \\
    S_3 = A_{21} + A_{22} \\
    S_4 = B_{21} - B_{11} \\
    S_5 = A_{11} + A_{22} \\
    S_6 = B_{11} + B_{22} \\
    S_7 = A_{12} - A_{22} \\
    S_8 = B_{21} + B_{22} \\
    S_9 = A_{11} - A_{21} \\
    S_{10} = B_{11} + B_{12}
    $$

3. Recursively compute each of the 7 matrix products taking $7T(n/2)$ time.

$$
P_1 = A_{11} S_1 (= A_{11} B_{12} - A_{11} B_{22}) \\
P_2 = S_2 B_{22} (= A_{11} B_{22} + A_{12} B_{22}) \\
P_3 = S_3 B_{11} (= A_{21} B_{11} + A_{22} B_{11}) \\
P_4 = A_{22} S_4 (= A_{22} B_{21} - A_{22} B_{11}) \\
P_5 = S_5 S_6 (= A_{11} B_{11} + A_{11} B_{22} + A_{22}B_{11} + A_{22} B_{22}) \\
P6 = S_7 S_8 (= A_{12} B_{21} + A_{12} B_{22} - A_{22}B_{21} - A_{22}B_{22}) \\
P_7 = S_9 S_{10} (= A_{11} B_{11} + A_{11} B_{12} - A_{21}B_{11} - A_{21} B_{12})
$$

4. Update the four submatrices $C_{11}, C_{12}, C_{21}, C_{22}$ by adding or substracting $P_i$, which takes $\Theta(n^2)$ time.

$$
C_{11} = P_5 + P_4 - P_2 + P_6 \\
C_{12} = P_1 + P_2 \\
C_{21} = P_3 + P_4 \\
C_{22} = P_5 + P_1 - P_3 - P_7
$$

Understanding is easy, getting it is amazing.

We obtain the following recurrence:

$$ T(n) = 7T(n/2) + \Theta(n^2) $$

By the master method, we get $T(n) = \Theta(n^{\lg7}) = O(n^{2.81})$

### Exercises

4.2-1

$$
\begin{align*}
S_1 &= B_{12} - B_{22} = 6 \\
S_2 &= A_{11} + A_{12} = 4 \\
S_3 &= A_{21} + A_{22} = 12 \\
S_4 &= B_{21} - B_{11} = -2 \\
S_5 &= A_{11} + A_{22} = 6 \\
S_6 &= B_{11} + B_{22} = 8 \\
S_7 &= A_{12} - A_{22} = -2 \\
S_8 &= B_{21} + B_{22} = 6 \\
S_9 &= A_{11} - A_{21} = -6 \\
S_{10} &= B_{11} + B_{12} = 14 \\
P_1 &= A_{11} S_1 = 6 \\
P_2 &= S_2 B_{22} = 8 \\
P_3 &= S_3 B_{11} = 72 \\
P_4 &= A_{22} S_4 = -10 \\
P_5 &= S_5 S_6 = 48 \\
P6 &= S_7 S_8 = -12 \\
P_7 &= S_9 S_{10} = -84 \\
C_{11} &= P_5 + P_4 - P_2 + P_6 = 18 \\
C_{12} &= P_1 + P_2 = 14 \\
C_{21} &= P_3 + P_4 = 62 \\
C_{22} &= P_5 + P_1 - P_3 - P_7 = 66
\end{align*}
$$

4.2-2

```c
STRASSEN(A,B,C,n)
if n == 1
// Base case.
    c_{11} = c_{11} + a_{11}*b_{11}
    return
// Divide
partition A, B, and C into n/2 * n/2 submatrices
    A_{11}, A_{12}, A_{21}, A_{22}; B_{11}, B_{12}, B_{21}, B_{22};
    and C_{11}, C_{12}, C_{21}, C_{22}; respectively
Create n/2*n/2 matrices S_1,S_2,...,S_{10}, and create and zero n/2*n/2 matrices P_1,P_2,...,P_7 to hold matrix products
S_1 = B_{12} - B_{22}
S_2 = A_{11} + A_{12}
S_3 = A_{21} + A_{22}
S_4 = B_{21} - B_{11}
S_5 = A_{11} + A_{22}
S_6 = B_{11} + B_{22}
S_7 = A_{12} - A_{22}
S_8 = B_{21} + B_{22}
S_9 = A_{11} - A_{21}
S_{10} = B_{11} + B_{12}
// Conquer.
STRASSEN(A_{11}, S_1, P_1, n/2)
STRASSEN(S_2, B_{22}, P_2, n/2)
STRASSEN(S_3, B_{11}, P_3, n/2)
STRASSEN(A_{22}, S_4, P_4, n/2)
STRASSEN(S_5, S_6, P_5, n/2)
STRASSEN(S_7, S_8, P_6, n/2)
STRASSEN(S_9, S_{10}, P_7, n/2)

C_{11} = P_5 + P_4 - P_2 + P_6
C_{12} = P_1 + P_2
C_{21} = P_3 + P_4
C_{22} = P_5 + P_1 - P_3 - P_7
```

4.2-3

We have $T(n) = kT(n/3) + O(1)$, using the master theorem, k need be $\lg_3 k < \lg 7$, so $k \le 3^{\lg 7}$, and the largest $k=21$. The running time is $\Theta(n^{\lg_3 21})$

4.2-4

$$
\lg_{68} 132464 = 2.795128 \\
\lg_{70} 143640 = 2.795123 \\
\lg_{72} 155424 = 2.795147 \\
$$

70 is the best one, they all faster than Strassen.

4.2-5

We can have three multiplies $(a+b)(c+d) = ac+ad+bc+bd, ac, bd$, Then $ad+bc = (a+b)(c+d) - ac - bd$ and $ac - bd$, so we get the answer.

4.2-6

Refer to [here](https://math.stackexchange.com/questions/4535345/reducing-matrix-multiplication-to-squaring).

$$
\begin{pmatrix}
 A & B \\
 0 & 0
\end{pmatrix}^2 = 
\begin{pmatrix}
 A^2 & AB \\
 0 & 0
 \end{pmatrix}
$$

By construct a new matrix, we can get $AB$.

## The substitution method for solving recurrences

The substitution method comprises two step:

1. Guess the form of the solution using symbolic constants.
2. Use mathematical induction to show that the solution works, and find the constants.

Rather than trying to prove a $\Theta$-bound directly, it is often best to first prove an $O$-bound, then prove an $\Omega$-bound.

Learning some recurrence-solving heuristics as well as playing around with recurences to gain experience and using recursion trees can help you become a good guesser.

when using substitution method, it's best to avoid asymptotic notation, because the constatns matter. But we can use the trick by subtracting a low-order term to make the inqualities work.

### Exercises

4.3-1

a.

If there exist $c, n_0$ that $T(n) \le cn^2$ when $ n \ge n_0$. We assume $n_0 = 1, c \ge 1$, We have:

$$
\begin{align*}
T(n) &= T(n-1) + n \\
&\le c(n-1)^2 + n \\
&= cn^2 + (1-2c)n + c \\
&\le cn^2
\end{align*}
$$

b.

If there exist $c, n_0$ that $T(n) \le c\lg n$ when $ n \ge n_0$. When $c$ is large enough, we have:

$$
\begin{align*}
T(n) &= T(n/2) + \Theta(1) \\
&\le c(\lg (n/2)) + \Theta(1) \\
&= c\lg n + (\Theta(1) - c) \\
&\le c\lg n
\end{align*}
$$

c.

If there exist $c_1, c_2, n_0$ that $c_1n(\lg n) \le T(n) \le c_2n\lg n$ when $ n \ge n_0$. Assume that $c_1 \le 1, c_2 \ge 1$, we have:

$$
\begin{align*}
T(n) &= 2T(n/2) + n \\
&\le c_2 n \lg(n/2) + n \\
&= c_2 n \lg n + (1 - c_2)n \\
&\le c_2 n \lg n
\end{align*}
$$

If there exist $c_1, n_0$ that $c_1n(\lg n) \le T(n) $ when $ n \ge n_0$. we have:

$$
\begin{align*}
T(n) &= 2T(n/2) + n \\
&\ge c_1 n (\lg(n/2)) + n \\
&= c_1 n \lg n + (1-c_1)n \\
&\ge c_1 n \lg n
\end{align*}
$$

d.

If there exist $c, n_0$ that $T(n) \le c(n-34)\lg (n-34)$ when $ n \ge n_0$. When $c > 1, n$ is large enough, we have:

$$
\begin{align*}
T(n) &= 2T(n/2+17) + n \\
&\le 2c (n/2 - 17) \lg(n/2 - 17) + n \\
&= c (n-34) \lg (n-34) + (1-c)n + 34c \\
&\le c (n-34) \lg (n-34)
\end{align*}
$$

e. 

If there exist $c_1, c_2, n_0$ that $c_1n \le T(n) \le c_2n$ when $ n \ge n_0$. Assume that $c_1$ small enough, $c_2$ large enough, we have:

$$
\begin{align*}
T(n) &= 2T(n/3) + \Theta(n) \\
&\le 2c_2 (n/3) + \Theta(n) \\
&= c_2n + (\Theta(n) - 1/3c_2n) \\
&\le c_2n
\end{align*}
$$

$$
\begin{align*}
T(n) &= 2T(n/3) + \Theta(n) \\
&\ge 2c_1 (n/3) + \Theta(n) \\
&= c_1n + (\Theta(n) - 1/3c_1n) \\
&\ge c_1n
\end{align*}
$$

f.

If there exist $c_2, n_0$ that $T(n) \le c_2(n^2-n)$ when $ n \ge n_0$. Assume that  $c_2$ large enough, we have:

$$
\begin{align*}
T(n) &= 4T(n/2) + \Theta(n) \\
&\le 4c_2 ((n/2)^2 - n/2) + \Theta(n) \\
&= c_2(n^2-n) + (\Theta(n) - c_2n) \\
&\le c_2(n^2-n)
\end{align*}
$$

If there exist $c_1, n_0$ that $c_1n^2 \le T(n)$ when $n \ge n_0$. we have:

$$
\begin{align*}
T(n) &= 4T(n/2) + \Theta(n) \\
&\ge 4c_2 (n/2)^2 + \Theta(n) \\
&= c_2n^2 + \Theta(n) \\
&\ge c_2n^2
\end{align*}
$$

4.3-2

If there exist $c_1, c_2, n_0$ that $c_1n^2 \le T(n) \le c_2n^2$ when $ n \ge n_0$. We have:

$$
\begin{align*}
T(n) &= 4T(n/2) + n \\
&\le 4c_2 (n/2)^2 +n \\
&= c_2 n^2 + n
\end{align*}
$$

We can't prove $T(n) \le c_2n^2$. If $T(n) \le c_2(n^2 - n)$, let $ c_2 \ge 1$, then we have:

$$
\begin{align*}
T(n) &= 4T(n/2) + n \\
&\le 4c_2 ((n/2)^2 - n/2) +n \\
&= c_2 (n^2-n) + (1- c_2)n \\
&\le c_2 (n^2-n)
\end{align*}
$$

$$
\begin{align*}
T(n) &= 4T(n/2) + n \\
&\ge 4c_1 (n/2)^2 +n \\
&= c_1 n^2 + n \\
&\ge c_1 n^2
\end{align*}
$$

4.3-3

If there exist $c, n_0$ that $T(n) \le c2^n$ when $n \ge n_0$. We have:

$$
\begin{align*}
T(n) &= 2T(n-1) + 1 \\
&\le 2 =2*c2^{n-1} +1 \\
&= c2^n + 1 \\
\end{align*}
$$

We can't prove $T(n) \le c2^n$, If $T(n) \le c(2^n-1)$, let $c>1$, then:

$$
\begin{align*}
T(n) &= 2T(n-1) + 1 \\
&\le 2*c(2^{n-1}-1) +1 \\
&= c(2^n-1) + 1-c \\
&\le c(2^n-1)
\end{align*}
$$

## The recursion-tree method for solving recurrences

In a recursion tree, each nodde represents the cost of a single subproblem. We typically sum the costs within each level of the tree to obtain the per-level costs, then sum all the per-level costs to determine the total cost.

A recursion tree is best used to generate intuition for a good guess.

It's wise to verify any bound obtained with a recursive tree by using the substitution method, or using a mre-powerful mathmatics like master method or Akra-Bazzi method.

### Exercises

4.4-1

a. 

$T(n) = O(n^3)$.

If there exist $c, n_0$ that $T(n) \le cn^3$ when $ n \ge n_0$. Assume $c \ge 8/7$, We have:

$$
\begin{align*}
T(n) &= T(n/2) + n^3 \\
&\le c(n/2)^3 + n^3 \\
&= cn^3 + (1-7/8c)n^3 \\
&\le cn^3
\end{align*}
$$

b.

$T(n) = O(n^{\lg_34})$.

If there exist $c, n_0$ that $T(n) \le c (n^{\lg_34} - n)$ when $ n \ge n_0$. Assume $c \ge 3$, We have:

$$
\begin{align*}
T(n) &= 4T(n/3) + n \\
&\le 4c((n/3)^{\lg_34}-n/3) + n \\
&= c(n^{\lg_34} - n) + (1-1/3c)n \\
&\le c(n^{\lg_34} - n)
\end{align*}
$$

c.

$T(n) = O(n^2)$

If there exist $c, n_0$ that $T(n) \le c (n^2-n)$ when $ n \ge n_0$. Assume $c \ge 1$, We have:

$$
\begin{align*}
T(n) &= 4T(n/2) + n \\
&\le 4c(n^2/4 - n/2) + n \\
&=c(n^2 - n) + (1 - c)n \\
&\le c(n^2 - n)
\end{align*}
$$

d. 

$T(n) = O(3^n)$

If there exist $c, n_0$ that $T(n) \le c (3^n-1)$ when $ n \ge n_0$. Assume $c \ge 0.5$, We have:

$$
\begin{align*}
T(n) &= 3T(n-1) + 1 \\
&= 3c(3^{n-1} - 1) + 1 \\
&=c(3^n - 1) + (1 - 2c)n \\
&\le c(3^n - 1)
\end{align*}
$$

4.4-2

If there exist $c, n_0$ that $L(n) \ge cn$ when $ n \ge n_0$. We have:

$$
\begin{align*}
L(n) &= L(n/3) + L(2n/3) \\
&\ge cn/3 + 2cn/3 \\
&= cn
\end{align*}
$$

So we get $L(n) = \Omega(n)$.

4.4-3

If there exist $c, n_0$ that $T(n) \ge cn \lg n$ when $ n \ge n_0$. We have:

$$
\begin{align*}
T(n) &= T(n/3) + T(2n/3) + \Theta(n) \\
&\ge c(n/3 \lg (n/3)) + c(2n/3 \lg (2n/3)) \\
&\ge c(n/3\lg n) + c(2n/3 \lg n) \\
&= cn\lg n
\end{align*}
$$

4.4-4

Like the example in book, it's $n\lg n$.

## The master method for solving recurrences

The master theorem

Let $a > 0$ and $b > 1$ be constants, and let $f(n)$ be a driving function that is defined and nonnegatie on all sufficiently large reals. Define the recurence $T(n)$ on $n \in \N$ by

$$
T(n) = aT(n/b) + f(n)
$$

where $aT(n/b)$ actually means $a'T(\lfloor n/b \rfloor) + a''T(\lceil n/b \rceil)$ for some constants $a' \ge 0$ and $a'' \ge 0$ satisfying $a = a' + a''$. Then the asymptotic behavior of $T(n)$ can be characterized as follows:

1. If there exists a constant $\epsilon > 0$ such that $f(n) = O(n^{\log_b{a-\epsilon}})$, then $T(n) = \Theta(n^{\log_ba})$.
2. If there exists a constant $k \ge 0$ such that $f(n) = \Theta(n^{\log_ba}\lg^k n)$, then $T(n) = \Theta(n^{\log_ba}\lg^{k+1}n)$.
3. If there exists a constant $\epsilon > 0$ such that $f(n) = \Omega(n^{\log_b{a+\epsilon}})$, and if $f(n)$ additionally satisfies the regularity condition $af(n/b) \le cf(n)$ for some constant $ c < 1$ and all sufficiently large $n$, then $T(n) = \Theta(f(n))$.

The function $n^{\log_ba}$ is called the watershed function. In each of the three cases, we compare the driving function $f(n)$ to the watershed function. Intuitively, if the watershed function grows asymptotically faster than the driving function, then case 1 applies, Case 2 applies if the two functions grow at nearly the same asymptotic rate. Case 3 is the opposite of case 1 where the driving functin grows asymptotically faster than the watershed function.

In case 1, the watershed function grow polynomially faster than the driving function. If we look at the recursion tree for the recurrence, the cost per level grows at least geometrically from root to leaves, and the total cost of leaves dominates the total cost of the internel nodes.

In case 2, the driving function grows faster than the watershed function by a factor of $\Theta(\lg^k n)$. In this ase, each level of the recursion tree costs approximately the same $\Theta(n^{\log_ba} \lg^k n)$, and there are $\Theta(\lg n)$ levels.

In case 3, the driving function grow polynomially faster than watershed function. Moreover, the driving function must satisfiy the regularity condition, which is satisfied by most of the polynomially bounded functions. It might not be satisfied if the driving function grows slowly in local areas, yet relatiely quickly overall. If we look at the recursion tree, the cost per leel drops at least geometrically from the root to the leaves, and the root cost dominates the cost of all other nodes.

There are situations where we can't use the master theorem. For example, it can be that the watershed function and the driving function cannot be asymptotically compared. Even when they can be compared, There is a gap between cases 1 and 2 when $f(n) = o(n^{\log_b a})$, yet the watershed function does not grow polynomially faster than the driving function. Similarly, there is a gap between cases 2 and 2 when $f(n) = \omega(n^{\log_b a})$, and the driving function grows more than polynomially faster htan the watershed function, but it does not grow polynomially faster.

### Exercises

4.5-1

$n^{\log_ba} = \sqrt n$

a. $T(n) = \sqrt n $
b. $T(n) = \sqrt n \lg n$
c. $T(n) = \sqrt n \lg^3 n$
d. $T(n) = n$
e. $T(n) = n^2$

4.5-2

$T(n) = aT(n/4) + \Theta(n^2)$

If $a < 16$, it's case 3, and $T(n) = n^2 < n^{\lg 7}$, ok.

If $ a = 16$, it's case 2, and $T(n) = n^2\lg n < n^{\lg 7}$, ok.

If $ a > 16$, it's case 1, and $T(n) = n^{\log_4a} < n^{\lg 7}$, we get $a < 49$, the largest is 48.

4.5-3

It's case2, solved.

4.5-4

$$
\lg (n/2) \le c \lg n \\
(1-c)\lg n \le 1
$$

When $n > 2^{\frac{1}{1-c}}$, the above inequality can't hold.

If there exist $c, n_0$ that $f(n) \ge c n^{\epsilon}$ when $ n \ge n_0$. We have:

$$
\lg n \ge cn^{\epsilon} \\
\lg \lg n \ge \lg c + \epsilon \lg n
$$

When $n$ is large enough ,the above inequality can't hold.

4.5-5

$$
n^2 = 2^{\lg n + 1} > 2^{\lceil \lg n \rceil} \ge 2^{\lg n} = n
$$

let $a=1, b=3/2, \epsilon=1$, we have $n^{\log_b{a+\epsilon}} = n$, so $T(n) = \Omega(n)$, because $b < 2$, when $n$ is large enough, there must exist $[\lceil \lg (n/(3/2)) \rceil = \lceil \lg n \rceil]$, so $af(n/b) = f(n)$. 

## Proof of the continuous master

This section states and proves a variant of the master theorem, called the continuous master theorem in which the master recurrence is defined over sufficiently large positive real numbers.

Lemma 4.2

Let $a > 0$ and $b > 1$ be constants, and let $f(n)$ be a function defined over real numbers $n \ge 1$. Then the recurrence

$$
T(n) = \begin{cases}
\Theta(1) \text{      if } 0 \le n < 1 \\
aT(n/b) + f(n)  \text{ if } n \ge 1 
\end{cases}
$$

it has solution

$$
T(n) = \Theta(n^{\log_ba}) + \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f(n/b^j)
$$

By constructing a recursion tree, we can prove it.

Lemma 4.3

Let $a > 0$ and $b > 1$ be constants, and let $f(n)$ be a function defined over real numbers $ n\ge 1$. Then the asymptotic behavior of the function

$$
g(n) = \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f(n/b^j)
$$

defined for $ n \ge 1$, can be characterized as follows:

1. If there exists a constant $ \epsilon > 0$ such that $f(n) = O(n^{\log_ba - \epsilon})$, then $g(n) = O(n^{\log_ba})$.
2. If there exists a constant $ k \ge 0$ such that $f(n) = \Theta(n^{\log_ba}\lg^k n)$, then $g(n) = \Theta(n^{\log_ba}\lg^{k+1} n)$.
3. If there exists a constant $c$ in the range $0 < c < 1$ such that $0 < af(n/b) \le cf(n)$ for all $n \ge 1$, then $g(n) = \Theta(f(n))$.

For case 1, we have:

$$
\begin{align*}
g(n) &= \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f(n/b^j) \\
&= O(\sum_{j=0}^{\lfloor log_b n \rfloor}a^j (\frac{n}{b^j})^{\log_b{a-\epsilon}}) \\
&= O(n^{\log_b{a-\epsilon}} \sum_{j=0}^{\lfloor log_b n \rfloor} (\frac{ab^\epsilon}{b^{\log_ba}})^j) \\
&= O(n^{\log_b{a-\epsilon}} \sum_{j=0}^{\lfloor log_b n \rfloor} (b^\epsilon)^j) \\
&= O(n^{\log_b{a-\epsilon}} (\frac{b^{\epsilon(\lfloor \log_b n \rfloor +1)} - 1}{b^\epsilon - 1})) \\
\end{align*}
$$

Sine $b^{\epsilon(\lfloor \log_b n \rfloor +1)} \le (b^{\log_b n + 1})^\epsilon = b^\epsilon n^\epsilon = O(n^\epsilon)$, so we get $g(n) = O(n^{\log_b{a-\epsilon}} O(n^\epsilon)) = O(n^{\log_ba})$.

For case 2, we have:

$$
\begin{align*}
g(n) &= \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f(n/b^j) \\
&= \Theta(\sum_{j=0}^{\lfloor log_b n \rfloor} a^j (\frac{n}{b^j})^{\log_b a} \lg^k(\frac{n}{b^j})) \\
&= \Theta(n^{\log_ba} \sum_{j=0}^{\lfloor log_b n \rfloor}\lg^k(\frac{n}{b^j})) \\
&= \Theta(n^{\log_ba} \sum_{j=0}^{\lfloor log_b n \rfloor}(\frac{\log_bn -j}{\log_b2})^k) \\
&= \Theta(\frac{n^{\log_ba}}{{\log_b2}^k} \sum_{j=0}^{\lfloor log_b n \rfloor}(\log_bn -j)^k) \\
&= \Theta(n^{\log_ba} \sum_{j=0}^{\lfloor log_b n \rfloor}(\log_bn -j)^k) \\
\end{align*}
$$

The summation from above can be bounded as follows:

$$
\begin{align*}
\sum_{j=0}^{\lfloor log_b n \rfloor}(\log_bn -j)^k &\le \sum_{j=0}^{\lfloor log_b n \rfloor}(\lfloor \log_bn \rfloor + 1 -j)^k \\
&= \sum_{j=1}^{\lfloor log_b n \rfloor + 1} j^k \\
&= O((\lfloor log_b n \rfloor + 1)^{k+1}) \\
&= O(\log_b^{k+1} n)
\end{align*}
$$

For case 3, based on $g(n)$ definition, we have $g(n) = \Omega(f(n))$. We have:

$$
\begin{align*}
g(n) &= \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f(n/b^j) \\
&\le \sum_{j=0}^{\lfloor log_b n \rfloor} c^j f(n) \\
&\le f(n) \sum_{j=0}^\infty c^j \\
&= f(n) (\frac{1}{1-c}) \\
&= O(f(n))
\end{align*}
$$

So we get $g(n) = \Theta(f(n))$.

Theorem 4.4 continuous master theorem

Let $a > 0$ and $b > 1$ be constants, and let $f(n)$ be a driving function that is defined and nonnegatie on all sufficiently large reals. Define the recurence $T(n)$ on the positive real numbers by

$$
T(n) = aT(n/b) + f(n)
$$

Then the asymptotic behavior of $T(n)$ can be characterized as follows:

1. If there exists a constant $\epsilon > 0$ such that $f(n) = O(n^{\log_b{a-\epsilon}})$, then $T(n) = \Theta(n^{\log_ba})$.
2. If there exists a constant $k \ge 0$ such that $f(n) = \Theta(n^{\log_ba}\lg^k n)$, then $T(n) = \Theta(n^{\log_ba}\lg^{k+1}n)$.
3. If there exists a constant $\epsilon > 0$ such that $f(n) = \Omega(n^{\log_b{a+\epsilon}})$, and if $f(n)$ additionally satisfies the regularity condition $af(n/b) \le cf(n)$ for some constant $ c < 1$ and all sufficiently large $n$, then $T(n) = \Theta(f(n))$.

The idea is to bound the summation from Lemma 4.2 by applying Lemma 4.3. But we must account for Lemma 4.2 using a base case for $ 0 < n < 1$, whereas this theorem uses an implicit base case for $ 0 < n < n_0$.

For $ n > 0$, let us define two auxiliary functions $T'(n) = T(n_0 n)$ and $f'(n) = f(n_0n)$. We have:

$$
\begin{align*}
T'(n) &= T(n_0 n) \\
&= \begin{cases}
\Theta(1) \text{ if } n_0n < n_0 \\
aT(n_0n/b) + f(n_0n) \text{ if } n_0n \ge n_0
\end{cases} \\
&= \begin{cases}
\Theta(1) \text{ if } n < 1 \\
aT'(n/b) + f'(n) \text{ if } n \ge 1
\end{cases} \\
\end{align*}
$$

By Lemma 4.2, the solution is:

$$
T'(n) = \Theta(n^{\log_ba}) + \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f'(n/b^j)
$$

For case 1, $f(n) = O(n^{\log_b{a-\epsilon}})$, we have:

$$
\begin{align*}
f'(n) &= f(n_0n) \\
&= O((n_0n)^{\log_b{a-\epsilon}}) \\
&= O(n^{\log_b{a-\epsilon}})
\end{align*}
$$

So it's case 1 of Lemma 4.3, the summation is $O(n^{\log_b{a-\epsilon}})$. Then we have:

$$
\begin{align*}
T(n) &= T'(n/n_0) \\
&= \Theta((n/n_0)^{\log_ba}) + O((n/n_0)^{\log_b{a-\epsilon}}) \\
&= \Theta(n^{\log_ba}) + O(n^{\log_b{a-\epsilon}}) \\
&= \Theta(n^{\log_ba})
\end{align*}
$$

For case 2, $f(n) = \Theta(n^{\log_ba}\lg^k n)$ for some constant $ k \ge 0$, we have:

$$
\begin{align*}
f'(n) &= f(n_0n) \\
&= \Theta((n_0n)^{\log_ba}\lg^k (n_0n)) \\
&= \Theta(n^{\log_ba}\lg^k n)
\end{align*}
$$

So it's case 2 of Lemma 4.3, the summation is $\Theta(n^{\log_ba}\lg^{k+1}n)$.Then we have:

$$
\begin{align*}
T(n) &= T'(n/n_0) \\
&= \Theta((n/n_0)^{\log_ba}) + \Theta((n/n_0)^{\log_ba}\lg^{k+1}(n/n_0)) \\
&= \Theta(n^{\log_ba}) + \Theta(n^{\log_ba}\lg^{k+1}n) \\
&= \Theta(n^{\log_ba}\lg^{k+1}n)
\end{align*}
$$

For case 3, $f(n) = \Omega(n^{\log_b{a+\epsilon}})$ and $f(n)$ satisfies the regularity condition, we have:

$$
\begin{align*}
f'(n) &= f(n_0n) \\
&= \Omega((n_0n)^{\log_b{a+\epsilon}}) \\
&= \Omega(n^{\log_b{a+\epsilon}})
\end{align*}
$$

$$
\begin{align*}
af'(n/b) &= af(n_0n/b) \\
&\le cf(n_0n)  \\
&= cf'(n)
\end{align*}
$$

So it's case 3 of Lemma 4.3, the summation is $\Theta(f'(n))$. Then we have:

$$
\begin{align*}
T(n) &= T'(n/n_0) \\
&= \Theta((n/n_0)^{\log_ba}) + \Theta(f'(n/n_0)) \\
&= \Theta(f'(n/n_0)) \\
&= \Theta(f(n))
\end{align*}
$$

### Exercises

4.6-1

$$
\begin{align*}
\sum_{j=0}^{\lfloor log_b n \rfloor}(\log_bn -j)^k &\ge \sum_{j=0}^{\lfloor log_b n \rfloor}(\lfloor \log_bn \rfloor -j)^k \\
&= \sum_{j=0}^{\lfloor log_b n \rfloor} j^k \\
&= \Omega((\lfloor log_b n \rfloor)^{k+1}) \\
&= \Omega(\log_b^{k+1} n)
\end{align*}
$$

4.6-2

For $c < 1, n > n_0$, we have $af(n/b) \le cf(n)$. So we have $f(n/b) \le \frac{c}{a}f(n)$, we have $f(n/b^i) \le (\frac{c}{a})^if(n)$, let $i = \log_b(n/n'_0)$ which $n'_0 > n_0$. Then we have:

$$
f(n'_0) \le (c/a)^{\log_b(n/n'_0)} f(n) \\
f(n) \ge (c/a)^{log_b n'_0}f(n'_0) n^{\log_b {a/c}}
$$

4.6-3

$$
\begin{align*}
g(n) &= \sum_{j=0}^{\lfloor log_b n \rfloor} a^j f(n/b^j) \\
&= \Theta(\sum_{j=0}^{\lfloor log_b n \rfloor} a^j (\frac{n}{b^j})^{\log_b a} /\lg(\frac{n}{b^j})) \\
&= \Theta(n^{\log_ba} \sum_{j=0}^{\lfloor log_b n \rfloor}\ 1/lg(\frac{n}{b^j})) \\
&= \Theta(n^{\log_ba} \sum_{j=0}^{\lfloor log_b n \rfloor}\frac{\log_b2}{\log_bn -j}) \\
&= \Theta(n^{\log_ba} \sum_{j=0}^{\lfloor log_b n \rfloor}\frac{1}{\log_bn -j}) \\
\end{align*}
$$

Notice $\log_bn -j \ne 0$.

The summation from above can be bounded as follows:

$$
\begin{align*}
\sum_{j=0}^{\lfloor log_b n \rfloor}\frac{1}{\log_bn -j} &\le \sum_{j=0}^{\lfloor log_b n \rfloor}\frac{1}{\lfloor log_b n \rfloor -j} \\
&= \sum_{j=1}^{\lfloor log_b n \rfloor} 1/j \\
&\le \ln\lfloor log_b n \rfloor + 1  \\
&= O(\lg\lg n)
\end{align*}
$$

$$
\begin{align*}
\sum_{j=0}^{\lfloor log_b n \rfloor}\frac{1}{\log_bn -j} &\ge \sum_{j=0}^{\lfloor log_b n \rfloor}\frac{1}{\lfloor log_b n \rfloor + 1 -j} \\
&= \sum_{j=1}^{\lfloor log_b n \rfloor + 1} 1/j \\
&\ge \ln(\lfloor log_b n \rfloor + 1) \\
&= \Omega(\lg\lg n)
\end{align*}
$$

So we have $\sum_{j=0}^{\lfloor log_b n \rfloor}\frac{1}{\log_bn -j} = \Theta(\lg\lg n)$, and $g(n) = n^{\log_ba} \lg\lg n$

We use the same method in book:

$$
\begin{align*}
f'(n) &= f(n_0 n) \\
&= \Theta((n_0n)^{\log_b a}/\lg(n_0n)) \\
&= \Theta(n^{\log_b a}/\lg n)
\end{align*}
$$

So the summation is still $n^{\log_ba} \lg\lg n$, so we have:

$$
\begin{align*}
T(n) &= T'(n/n_0) \\
&= \Theta((n/n_0)^{\log_ba}) + \Theta((n/n_0)^{\log_ba}\lg\lg(n/n_0)) \\
&= \Theta(n^{\log_ba}) + \Theta(n^{\log_ba}\lg\lg n) \\
&= \Theta(n^{\log_ba}\lg\lg n)
\end{align*}
$$

## Akra-Bazzi recurrences

The Akra-Bazzi recurrences take the form

$$
T(n) = f(n) + \sum_{i=1}^k a_i T(n/b_i)
$$

The master theorem allows you to ignore floors and ceilings, but the Akra-Bazzi method needs an additional requirement to deal with floors and ceilings.

If the driving funtion $f(n)$ is well behaved in the following sense, it's okay t drop floors and ceilings.

A function $f(n)$ defined on all sufficiently large positive reals satisfies the polynomial-growth condition if there exists a constant $ \hat n > 0$ such that the following holds: for every constant $\phi \ge 1$, there exists a constant $ d > 1 $ (depending on $\phi$) such that $ f(n)/d \le f(\psi n) \le df(n)$ for all $ 1 \le \psi \le \phi$ and $ n \ge \hat n$.

Examples of functions that satisfy the polynomial-growth conditin include any function of the form $ f(n) = \Theta(n^\alpha \lg^\beta n \lg \lg^\gamma n)$. Exponentials and superexponentials do not satisfy the condition.

The Akra-Bazzi method

The method involves first determining the unique real number $p$ such that $\sum_{i=1}^k a_i / b_i^p = 1$, then we have:

$$
T(n) = \Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx))
$$

### Exercises

4.7-1

$$
T'(n) = \Theta(n^p ( 1 + \int_1^n \frac{cf(x)}{x^{p+1}} dx)) = \Theta(n^p ( 1 + c\int_1^n \frac{f(x)}{x^{p+1}} dx))
$$

All we need is to prove $\int_1^n \frac{f(x)}{x^{p+1}} dx = \omega(1)$. Because $\frac{f(x)}{x^{p+1}} > 0$, so for any $c > 0$, if $n$ is large enough, it always hold $\int_1^n \frac{f(x)}{x^{p+1}} dx > c$.

So we have:

$$
T'(n) = c\Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx)) = cT(n)
$$

4.7-2

Let $d = \phi^2$, we have $f(\psi n) = \psi^2 n^2 \le \phi^2 n^2 = df(n)$ and $f(\psi n) = \psi^2 n^2 \ge n^2 \ge f(n)/d$

So $f(n) = n^2$ satisfies the polynomial-growth condition.

$(2^n)^\psi \le d 2^n$, when $\psi > 1$ and $n$ is large enough, it can't hold.

4.7-3

Let $n_0 = \hat n$, based on definition, we have $f(n)/d \le df(n)$, because $d > 1$, there must $f(n) \ge 0$.

4.7-4

#TODO

$f(n) = n^{\lg\lg n}$. we want to prove:

$$
c_1 n^{\lg\lg n} \le \Theta(n)^{\lg\lg \Theta(n)} \le c_2 n^{\lg\lg n}
$$

which mean:

$$
\lg c_1 + \lg\lg n \lg n^ \le \lg\lg \Theta(n) \lg \Theta(n) \le \lg c_2 + \lg\lg n \lg n
$$

when $n$ is large enough, we have:

$$
\begin{align*}
\lg\lg \Theta(n) \lg \Theta(n) &\ge \lg\lg k_1n \lg k_1n \\
&= \lg(\lg k_1 + \lg n) (\lg k_1 + \lg n) \\
&\ge \lg\lg n (\lg k_1 + \lg n) \\
&= \lg k_1 \lg\lg n + \lg\lg n \lg n \\
&\ge \lg c_1 + \lg\lg n \lg n
\end{align*}
$$

$$
\begin{align*}
\lg\lg \Theta(n) \lg \Theta(n) &\le \lg\lg k_2n \lg k_2n \\
&= \lg(\lg k_2 + \lg n) (\lg k_2 + \lg n) \\
&\le \lg(2\lg n) (\lg k_2 + \lg n) \\
&= \lg k_1 \lg\lg n + \lg\lg n \lg n \\
&\ge \lg c_1 + \lg\lg n \lg n
\end{align*}
$$

4.7-5
a.

$$
(\frac{1}{2})^p + (\frac{1}{3})^p + (\frac{1}{6})^p = 1
$$

we get $p = 1$, we have:

$$
\begin{align*}
T(n) &= \Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx)) \\
&= \Theta(n ( 1 + \int_1^n \frac{x\lg x}{x^2} dx)) \\
&= \Theta(n (1 + \frac{1}{\ln2} \int_1^n \ln x d(\ln x))) \\
&= \Theta(n (1 + \frac{1}{2\ln2} \ln^2 n)) \\
&= \Theta(n\lg^2n)
\end{align*}
$$

b. 

$$
3(\frac{1}{3})^p + 8(\frac{1}{4})^p = 1
$$

we get $ 1 < p < 2$, we have:

$$
\begin{align*}
T(n) &= \Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx)) \\
&= \Theta(n^p ( 1 + \int_1^n \frac{x^2\lg x}{x^{p+1}} dx)) \\
&= \Theta(n^p (1 + \frac{1}{2\ln2}(\frac{1}{2-p} n^{2-p} (\ln n - \frac{1}{2-p}) + \frac{1}{(2-p)^2}))) \\
&= \Theta(n^p n^{2-p} \lg n) \\
&= \Theta(n^2 \lg n)
\end{align*}
$$

Let $ u = \ln x, v = \frac{x^{2-p}}{2-p}$, we have:

$$
\begin{align*}
\int_1^n x^{1-p} \ln x dx &= \int_1^n u dv \\
&= uv - \int_1^n v du \\
&= \frac{n^{2-p} \ln n}{2-p} - \int_1^n \frac{x^{1-p}}{2-p} dx \\
&= \frac{n^{2-p} \ln n}{2-p} - \frac{n^{2-p} - 1}{(2-p)^2}
\end{align*}
$$

c.

$$
\frac{2}{3}(\frac{1}{3})^p + \frac{1}{3}(\frac{2}{3})^p = 1
$$

We get $ p = 0 $, we have:


$$
\begin{align*}
T(n) &= \Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx)) \\
&= \Theta( 1 + \int_1^n \frac{\lg x}{x} dx) \\
&= \Theta(\lg^2 n)
\end{align*}
$$

d.

$$
\frac{1}{3}(\frac{1}{3})^p = 1
$$

We get $ p = -1$, we have:

$$
\begin{align*}
T(n) &= \Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx)) \\
&= \Theta(n^{-1}(1 + \int_1^n \frac{1}{x} dx)) \\
&= \Theta(n^{-1}(1-\frac{1}{2}(n^{-2}-1))) \\
&= \Theta(1/n)
\end{align*}
$$

e.

$$
3(\frac{1}{3})^p + 3(\frac{2}{3})^p = 1
$$

We get $ p = 3 $, we have:

$$
\begin{align*}
T(n) &= \Theta(n^p ( 1 + \int_1^n \frac{f(x)}{x^{p+1}} dx)) \\
&= \Theta(n^3(1 + \int_1^n \frac{1}{x^2} dx)) \\
&= \Theta(n^3(1-\frac{1}{3}(n^{-3}-1))) \\
&= \Theta(n^3)
\end{align*}
$$

4.7.6

First we have:

$$
a(\frac{1}{b})^p = 1
$$

So we get $ p = \log_b a$, and based on Akra-Bazzi method, we have:

$$
T(n) = \Theta(n^{\log_b a} ( 1 + \int_1^n \frac{f(x)}{x^{\log_b a+1}} dx))
$$

For case 1, $f(n) = O(n^{\log_b a - \epsilon})$, so there exist $c > 0, n_0 > 0$ that for $n \ge n_0$, $f(n) \le cn^{\log_b a - \epsilon}$, so we have:

$$
\begin{align*}
\int_1^n \frac{f(x)}{x^{\log_b a+1}} dx &= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \int_{n_0}^n \frac{f(x)}{x^{\log_b a+1}} dx \\
&\le \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \int_{n_0}^n \frac{cx^{\log_b a - \epsilon}}{x^{\log_b a+1}} dx \\
&\le \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + c\int_{n_0}^n x^{-1 - \epsilon} dx \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \frac{c}{\epsilon}(x^{-n_0} -  x^{-n}) \\
&\le \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \frac{c}{\epsilon}x^{-n_0} \\
&= O(1)
\end{align*}
$$

So we get $T(n) = \Theta(n^{\log_b a})$.

For case 2, $f(n) = \Theta(n^{\log_ba}\lg^k n)$, so we have:

$$
\begin{align*}
\int_1^n \frac{f(x)}{x^{\log_b a+1}} dx &= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \int_{n_0}^n \frac{f(x)}{x^{\log_b a+1}} dx \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \int_{n_0}^n \frac{\Theta(x^{\log_ba}\lg^k x)}{x^{\log_b a+1}} dx \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \Theta(\int_{n_0}^n \frac{\lg^k x}{x} dx) \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \Theta(\lg^{k+1}n) \\
&= \Theta(\lg^{k+1}n)
\end{align*}
$$

So we get $T(n) = \Theta(n^{\log_b a}\lg^{k+1}n)$

For case 3, $f(n) = \Omega(n^{\log_b{a+\epsilon}})$, so we have: 

#TODO

$$
\begin{align*}
\int_1^n \frac{f(x)}{x^{\log_b a+1}} dx &= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \int_{n_0}^n \frac{f(x)}{x^{\log_b a+1}} dx \\
&\ge \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \int_{n_0}^n \frac{cx^{\log_b a + \epsilon}}{x^{\log_b a+1}} dx \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + c\int_{n_0}^n x^{-1 + \epsilon} dx \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \frac{c}{\epsilon}(x^n -  x^{n_0}) \\
&= \omega(1)
\end{align*}
$$

$$
\begin{align*}
\int_1^n \frac{f(x)}{x^{\log_b a+1}} dx &= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx - \frac{1}{\log_ba}\int_{n_0}^n f(x) d(x^{-\log_ba}) \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx - \frac{1}{\log_ba}((f(n)n^{-\log_ba} - f(n_0)n_0^{-\log_ba}) - \int_{n_0}^n x^{-\log_ba} d(f(x))) \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + c\int_{n_0}^n x^{-1 + \epsilon} dx \\
&= \int_1^{n_0-1} \frac{f(x)}{x^{\log_b a+1}} dx + \frac{c}{\epsilon}(x^n -  x^{n_0}) \\
&= \omega(1)
\end{align*}
$$

$$
\begin{align*}
\int_{n_0}^n x^{-\log_ba} d(f'(x)) &= \Omega(\int_{n_0}^n x^{-\log_ba} x^{\log_ba+\epsilon-1} d(x)) \\
&= \Omega(\frac{1}{\epsilon}(n^\epsilon - n_0^\epsilon)) \\
&= \Omega(n^\epsilon)
\end{align*}
$$

## Problems

4.1

a. $T(n) = \Theta(n^3)$
b. $T(n) = \Theta(n)$
c. $T(n) = \Theta(n^2\lg n)$
d. $T(n) = \Theta(n^2\lg^2 n)$
e. $T(n) = \Theta(n^2)$
f. $T(n) = \Theta(n^{7/2})$
g. $T(n) = \Theta(n^{1/2}\lg n)$
h. $T(n) = \Theta(n^3)$

4.2

a.

$$
T_{a1}(N, n) = T_{a1}(N, n/2) + \Theta(1)
$$

We get $T_{a1}(N) = \lg N$

$$
T_{a2}(N, n) = T_{a2}(N, n/2) + \Theta(N)
$$

We get $T_{a2}(N) = N \lg N$

$$
T_{a2}(N, n) = T_{a2}(N, n/2) + \Theta(n/2)
$$

We get $T_{a3}(N) = N$

b.

$$
T_{b1}(N, n) = 2T_{b1}(N, n/2) + \Theta(n)
$$

We get $T_{b1}(N) = N\lg N$

$$
T_{b2}(N, n) = 2T_{b2}(N, n/2) + 2\Theta(N) + \Theta(n)
$$

We get $T_{b2}(N) = N^2$

$$
T_{b2}(N, n) = 2T_{b2}(N, n/2) + 2\Theta(n/2) + \Theta(n)
$$

We get $T_{b1}(N) = N\lg N$

c.

$$
T_{c1}(N, n) = 8T_{c1}(N, n/2) + \Theta(1)
$$

We get $T_{c1}(N) = N^3$

$$
\begin{align*}
T_{c2}(N, n) &= 8T_{c2}(N, n/2) + 8\Theta(N) + \Theta(1) \\
&= 8^2T{c2}(N, n/4) + 8^2\Theta(N) + 8\Theta(N) \\
&= \sum_1^{\lg n} 8^i \Theta(N) \\
\end{align*}
$$

We get $T_{c2}(N) = N^4$

$$
T_{c3}(N, n) = 8T_{c3}(N, n/2) + 8\Theta(n) + \Theta(1)
$$

We get $T_{c1}(N) = N^3$

4.3

a.

$$
S(m) = 2S(m/2) + \Theta(m)
$$

b.

$$
S(m) = \Theta(m\lg m)
$$

c.

$$
T(n) = S(m) = \Theta(m\lg m) = \Theta(\lg n \lg\lg n)
$$

d.
level sum is $\lg n$, and height is $\lg\lg n$.

e.

Let $n = 2^m, S(m) = T(n)$, we have:

$$
S(m) = 2S(m/2) + \Theta(1)
$$

We get $S(m) = \Theta(m)$, So 

$$
T(n) = S(m) = \Theta(m) = \Theta(\lg n)
$$

f.

Let $n = 2^m, S(m) = T(n)$, we have:

$$
S(m) = 3S(m/3) + \Theta(2^m)
$$

We get $S(m) = \Theta(2^m)$, So

$$
T(n) = S(m) = \Theta(2^m) = \Theta(n)
$$

4.4

a. $T(n) = \Theta(n^{5/3})$
b. $T(n) = \Theta(n)$
c. $T(n) = \Theta(n^{7/2})$
d. $T(n) = \Theta(n\lg n)$
e. $T(n) = \Theta(n)$
f. $T(n) = \Theta(n)$
g. $T(n) = \Theta(\lg n)$
h. $T(n) = \Theta(\lg (n!)) = \Theta(n\lg n)$

j. 

Let $n = 2^m, S(m)=T(n)$, we get $S(m)=2^{m/2} S(m/2) + 2^m$


4.4

a.

$$
\begin{align*}
z + zF(z) + z^2F(z) &= z + \sum_0^\infty F_i z^{i+1} + \sum_0^\infty F_i z^{i+2} \\
&= z + \sum_1^\infty F_{i-1}z^i + \sum_2^\infty F_{i-2}z^i \\
&= \begin{cases}
0 \text{ if } i = 0 \\
1 + F_0 = F_i \text{ if } i = 1 \\
F_{i-1} + F_{i-2} = F_i \text{ if } i \ge 2 \\
\end{cases}
\end{align*}
$$

b.

based on a, we have:

$$
(1-z-z^2)F(z) = z
$$

Then we get $F(z) = \frac{z}{1-z-z^2}$. We have:

$$
1 - z - z^2 = (1 - \phi z)(1 - \hat \phi z)
$$

We can solve it.

c.

From exercise 3.3-8, we have proved it.

d.

From chap3, we have:

$$
F_i = \lfloor \frac{\phi^i}{\sqrt5} + \frac{1}{2} \rfloor
$$

e.

$$
F_2 = 1 \ge \phi^0
$$

$$
F_{i+2} = F_i + F_{i+1} \ge \phi^i + \phi^{i+1} = \phi^i(1+\phi) =\phi^{i+2}
$$

4.6

a.

If $m < n/2$ are good, then we can let $m$ pretend good, when it meet good chip, it give good to pretended bad chip, give bad to other bad chip, then these $m$ pretended good chips is mirror to those $m$ good chips, and we can't judge it.

b. For every two chips, if in case 1, random give one chip to group 1, one to group 2. If in case 2 or case 3, give good to group 1, bad to group 2. If in case 4, we don't put them in groups. After $\lfloor n/2 \rfloor$ tests, We have two groups. If no item in group, if no one left, then it's not true that more than $n/2$ are good. If one left, then it must be the good one. For case 1,2,3, group 2 bad chips are same or more than group 1; If no one left, then group 1 must have more than half good chips. otherwise together have less than half good chips. If one left, If every group have even chips, then add the left one to group 1, because in group 1, good chips must larger or euqal to bad chips, otherwise together have less than half good chips. And if it's euqal to bad chips, the left one must be good one. If its larger then bad chips (at least two more), add one is ok. If every group have odd chips, then good chips must more than bads. So in any case, we have group 1 which keep the property. The group 1 size is at most $\lceil n/2 \rceil$.

c.
    recursively until the group 1 has 1 chip, which must be the good one.

$$
    T(n) = T(\lceil n/2 \rceil) + \lfloor n/2 \rfloor
$$

we get $T(n) = \Theta(n)$.

d.

We have the golden one, pair every other chip to the good one, based on golden chip result, we can have all good chips.

4.7

a.

If for some $i$ and $j$, we have:

$$
A[i,j] + A[i+1,j+1] > A[i,j+1] + A[i+1,j]
$$

Let $k=i+1,l=j+1$, then we have:

$$
A[i,j] + A[k,l] > A[i,l] + A[k,j]
$$

It's not a monge array.

$$
A[i,j] + A[i+1,j+1] \ge A[i,j+1] + A[i+1,j] \\
A[i+1,j] + A[i+2,j+1] \ge A[i+1,j+1] + A[i+2,j]
$$

together we have:

$$
A[i,j] + A[i+2,j+1] \ge A[i,j+1] + A[i+2,j]
$$

Similarly we have:

$$
A[i,j] + A[i+p,j+1] \ge A[i,j+1] + A[i+p,j]
$$

$$
A[i,j] + A[i+p,j+q] \ge A[i,j+q] + A[i+p,j]
$$

So we get the definition.

b.

$$
\begin{matrix}
37 & 23 & 22 & 32 \\
21 & 6 & 4 & 10 \\
53 & 34 & 30 & 31 \\
32 & 13 & 9 & 6 \\
43 & 21 & 15 & 8
\end{matrix}
$$

c.

For $p<q$, if $f(p) > f(q)$, then we have $A[p, f(q)] > A[p, f(p)]$ and $A[q, f(p)] \ge A[q, f(q)]$, so we have $A[p, f(q)] + A[q, f(p)] >  A[p, f(p)] + A[q, f(q)]$, it's not a monge array.

d.

Based on c, we have:

$$
f(1) \le f(2) \\
f(2) \le f(3) \le f(4) \\
\cdots \\
$$

so the total time is:

if $m$ is odd:

$$
f(1) + f(3) + \cdots = f(2) + f(4) - f(2) + 1 + \cdots + n - f(m-1) + 1 = n + (m-1)/2
$$

If $m$ is even:

$$
f(1) + f(3) + \cdots = f(2) + f(4) - f(2) + 1 + \cdots + f(m) - f(m-2) + 1 \le n + m/2 - 1
$$

Any way, the cost time is $O(m+n)$.

e.

$$
T(m,n) = T(\lfloor m/2 \rfloor,n) + O(m) + O(m+n)
$$

Based on substitution method, we get $T(m,n) = O(m+n\lg m)$.

# Sets

## Sets

B.1-1

Just draw like Figure B.1

B.1-2

We can use induction to prove it.

When $n=2$, $\overline{A_1 \cap A_2} = \overline A_1 \cup \overline A_2$,
If it holds when $n=i$, then for $n=i+1$, we have:

$$
\begin{align*}
\overline A_1 \cup \overline A_2 \cup \cdots \cup \overline A_i \cup \overline A_{i+1} &= \overline{A_1 \cap A_2 \cap \cdots \cap A_i} \cup \overline A_{i+1} \\
&= \overline{A_1 \cap A_2 \cap \cdots \cap A_{i+1}} 
\end{align*}
$$

Similarly, when $n=2$, $\overline{A_1 \cup A_2} = \overline A_1 \cap \overline A_2$.

If it holds when $n = i$, then for $n = i + 1$, we have:

$$
\begin{align*}
\overline A_1 \cap \overline A_2 \cap \cdots \cap \overline A_i \cap \overline A_{i+1} &= \overline{A_1 \cup A_2 \cup \cdots \cup A_i} \cap \overline A_{i+1} \\
&= \overline{A_1 \cup A_2 \cup \cdots \cup A_{i+1}} 
\end{align*}
$$

B.1-3

When $n=2$ it holds, if $n=i$ it holds, then:

$$
\begin{align*}
|A_1 \cup A_2 \cup \cdots \cup A_i \cup A_{i+1}| &= |A_1 \cup A_2 \cup \cdots \cup A_i| + |A_{i+1}| - |(A_1 \cup A_2 \cup \cdots \cup A_i) \cap A_{i+1}| \\
&= |A_1 \cup A_2 \cup \cdots \cup A_i| + |A_{i+1}| - |(A_1 \cap A_{i+1}) \cup (A_2 \cap A_{i+1}) \cup \cdots \cup (A_i \cap A_{i+1})| \\
&= |A_1| + |A_2| + \cdots + |A_i| + |A_{i+1}|
- |A_1 \cap A_2| - |A_1 \cap A_3| - \cdots -|A_1 \cap A_{i+1}| - |A_2 \cap A_{i+1}|
+ |A_1 \cap A_2 \cap A_3| + |A_1 \cap A_2 \cap A_{i+1}|
+ (-1)^i|A_1 \cap A_2 \cap \cdots \cap A_{i+1}|
\end{align*}
$$

B.1-4

For every odd natural number, $n = 2k+1$, where k is in $\N$, so the odd natural number is countable too.

B.1-5

For every element, it have two choose, in a subset or not in a subset, so the total subsets is $2^{|S|}$.

B.1-6

$$
(a_1,a_2,a_3,\cdots,a_n) = \{a_1,\{a_1,a_2\},\{a_1,a_2,a_3\},\{a_1,a_2,a_3,\cdots,a_n\}\}
$$

## Relations

## Functions

## Graphs

## Trees

## Problems

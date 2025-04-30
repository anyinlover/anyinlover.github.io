---
title: DP and Policy Iteration
description: Using policy and value Iteration to solve bellman optimality equation
sidebar_position: 3
---

In general, RL algorithms iterate over two steps:

1. **Policy evaluation**: For a given policy $\pi$, the value of all states $V^\pi(s)$ or all state-action pairs $Q^\pi(s, a)$ is calculated or estimated.

2. **Policy improvement**: From the current estimated values $V^\pi(s)$ or $Q^\pi(s, a)$, a new **better** policy $\pi$ is derived.


After enough iterations, the policy converges to the **optimal policy** (if the states are Markov).

This alternation between policy evaluation and policy improvement is called **generalized policy iteration** (GPI).
One particular form of GPI is **dynamic programming**, where the Bellman equations are used to evaluate a policy.

## Policy Evaluation

When the environment model is known, the Bellman equations for the state-value function $v_\pi(s)$ become a system of $|\mathcal{S}|$ linear equations in the variables $v_\pi(s)$. We can use iterative methods to solve these equations. This algorithm is known as iterative policy evaluation.

$$
\begin{align*}
& \text{Loop:} \\
& \quad \Delta \gets 0 \\
& \quad \text{Loop for each } s \in \mathcal{S}: \\
& \qquad v \gets V(s) \\
& \qquad V(s) \gets \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')] \\
& \qquad \Delta \gets \max(\Delta, | v - V(s) |) \\
& \text{until } \Delta < \theta
\end{align*}
$$

In practice, we consider the process converged when the maximum change $\Delta$ is less than a small positive threshold $\theta$.

## Policy Improvement

Given an existing policy $\pi$, how can we find a better policy $\pi'$? Consider a deterministic policy. Suppose for a given state $s$, we change the action to $\pi'(s) = a \neq \pi(s)$, while keeping the policy unchanged for all other states $s' \neq s$. If $q_\pi(s,a) > v_\pi(s)$, then we consider the new policy $\pi'$ to be better than $\pi$. This is actually a special case of the policy improvement theorem stated below. If for all $s \in \mathcal{S}$, the following condition holds:

$$q_\pi(s,\pi'(s)) \ge v_\pi(s)$$

Then the policy $\pi'$ must be as good as, or better than, $\pi$. That is, for all $s \in \mathcal{S}$:

$$v_{\pi'}(s) \ge v_\pi(s)$$

The policy improvement theorem can be derived as follows:

$$
\begin{align*}
v_\pi(s) &\le q_\pi(s, \pi'(s)) \\
&= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t=\pi'(s)] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s] \\
&\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) | S_t = s] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma \mathbb{E}_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) | S_{t+1}, A_{t+1} = \pi'(S_{t+1})] | S_t = s] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) | S_t=s] \\
&\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_\pi(S_{t+3}) | S_t=s] \\
& \vdots \\
&\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots | S_t=s] \\
&= v_{\pi'}(s)
\end{align*}
$$

With the policy improvement theorem, we can use a greedy approach to obtain a new, better policy. The theorem guarantees that choosing actions greedily with respect to the value function of the current policy leads to an improved (or equal) policy.

$$
\begin{align*}
\pi'(s) &\doteq \argmax_a q_\pi(s,a) \\
&= \argmax_a \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t=s, A_t=a] \\
&= \argmax_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_\pi(s')]
\end{align*}
$$

When the greedy policy $\pi'$ is no longer strictly better than the previous policy $\pi$ (i.e., $v_{\pi'} = v_\pi$), then $v_\pi$ must satisfy the Bellman optimality equation:

$$v_{\pi'}(s) = \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_{\pi'}(s')]$$

This means $v_{\pi'} = v_\ast$, and thus $\pi'$ is an optimal policy $\pi_\ast$.

Note that the specific calculation shown above is for deterministic policies, but the derivation of the policy improvement theorem holds for stochastic policies as well.

## Policy Iteration

Combining policy evaluation and policy improvement gives us the policy iteration algorithm. Since the policy improvement step guarantees improvement until the optimal policy is reached (unless already optimal), and a finite Markov Decision Process (MDP) has a finite number of policies, policy iteration is guaranteed to converge to an optimal policy in a finite number of iterations.

Note that during the policy evaluation phase, initializing $V(s)$ with the value function from the previous policy can speed up convergence.

$$
\begin{align*}
& \text{1. Initialization} \\
& \quad V(s) \in \mathbb{R} \text{ and } \pi(s) \in \mathcal{A}(s) \text{ arbitrarily for all } s \in \mathcal{S} \\
& \text{2. Policy Evaluation} \\
& \quad \text{Loop:} \\
& \qquad \Delta \gets 0 \\
& \qquad \text{Loop for each } s \in \mathcal{S}: \\
& \qquad \quad v \gets V(s) \\
& \qquad \quad V(s) \gets \sum_{s',r} p(s',r|s,\pi(s))[r + \gamma V(s')] \quad \text{// Using current policy } \pi \\
& \qquad \quad \Delta \gets \max(\Delta, | v - V(s) |) \\
& \quad \text{until } \Delta < \theta \quad \text{// } V \approx v_\pi \\
& \text{3. Policy Improvement} \\
& \quad \text{policy-stable} \gets \text{true} \\
& \quad \text{For each } s \in \mathcal{S}: \\
& \qquad \text{old-action} \gets \pi(s) \\
& \qquad \pi(s) \gets \argmax_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')] \\
& \qquad \text{If } \text{old-action} \neq \pi(s), \text{ then policy-stable} \gets \text{false} \\
& \quad \text{If policy-stable, then stop and return } V \approx v_\ast \text{ and } \pi \approx \pi_\ast; \text{ else go to 2}
\end{align*}
$$

## Value Iteration

A potential drawback of policy iteration is that each iteration involves a full policy evaluation phase, which can be computationally expensive as it requires multiple sweeps through the state space until convergence. In fact, we can truncate the policy evaluation step after just one sweep (one update for each state). This leads to the value iteration algorithm:

$$v_{k+1}(s) \doteq \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_k(s')]$$

This update rule can also be seen as applying the Bellman optimality equation directly as an update. It can be proven that this iterative method converges to the optimal value function $v_\ast$. To show this, we define the Bellman optimality operator $\mathcal{T}$:

$$\mathcal{T} v_k(s) \doteq \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_k(s')]$$
So the update rule is $v_{k+1} = \mathcal{T} v_k$.

We introduce the concept of a contraction mapping. An operator $O$ mapping a Banach space (like the space of value functions) to itself is a $\gamma$-contraction if there exists a $\gamma \in [0, 1)$ such that for any two elements $V, V'$ in the space, $||OV - OV'|| \le \gamma || V - V' ||$, where $||\cdot||$ is a norm. We will use the max norm (or infinity norm), $||x||_\infty = \max_s |x(s)|$.

We prove that the Bellman optimality operator $\mathcal{T}$ is a $\gamma$-contraction in the max norm:

$$
\begin{align*}
||\mathcal{T} v - \mathcal{T} v'||_\infty &= \max_{s \in \mathcal{S} }| \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v(s')] - \max_{a'} \sum_{s',r} p(s',r | s,a')[r + \gamma v'(s')]| \\
&\le \max_{s \in \mathcal{S} } \max_a | \sum_{s',r}p(s',r | s,a)[r + \gamma v(s')] - \sum_{s',r} p(s',r | s,a)[r + \gamma v'(s')] | \\
&= \max_{s \in \mathcal{S} } \max_a | \sum_{s',r}p(s',r | s,a)[\gamma v(s') - \gamma v'(s')] | \\
&= \gamma \max_{s \in \mathcal{S} } \max_a |\sum_{s',r}p(s',r | s,a)(v(s')-v'(s'))| \\
&\le \gamma \max_{s \in \mathcal{S} } \max_a \sum_{s',r}p(s',r | s,a) |v(s')-v'(s')| \\
&\le \gamma \max_{s \in \mathcal{S} } \max_a \sum_{s',r}p(s',r | s,a) \max_{s''}|v(s'')-v'(s'')| \\
&= \gamma ||v - v'||_\infty \max_{s \in \mathcal{S} } \max_a \sum_{s',r}p(s',r | s,a) \\
&= \gamma||v - v'||_\infty \quad (\text{since } \sum_{s',r}p(s',r | s,a) = 1)
\end{align*}
$$
The key step uses the inequality $\max_x f(x) - \max_y g(y) \le \max_x (f(x) - g(x))$.

Since $\mathcal{T}$ is a $\gamma$-contraction and the optimal value function $v_\ast$ is its unique fixed point ($\mathcal{T} v_\ast = v_\ast$), the Banach fixed-point theorem guarantees that the sequence $v_k$ defined by $v_{k+1} = \mathcal{T} v_k$ converges to $v_\ast$ for any starting $v_0$. Specifically, when $v' = v_\ast$:

$$||v_{k+1} - v_\ast||_\infty = ||\mathcal{T} v_k - \mathcal{T} v_\ast ||_\infty \le \gamma ||v_k - v_\ast||_\infty \le \cdots \le \gamma^{k+1}||v_0 - v_\ast||_\infty$$

Therefore, when $\gamma < 1$, $\lim_{k\rightarrow \infty}v_k = v_\ast$.

The overall algorithm is as follows:

$$
\begin{align*}
& \text{Initialize } V(s) \in \mathbb{R} \text{ arbitrarily for all } s \in \mathcal{S} \\
& \text{Loop:} \\
& \quad \Delta \gets 0 \\
& \quad \text{Loop for each } s \in \mathcal{S}: \\
& \qquad v \gets V(s) \\
& \qquad V(s) \gets \max_a\sum_{s',r} p(s',r|s,a)[r + \gamma V(s')] \\
& \qquad \Delta \gets \max(\Delta, | v - V(s) |) \\
& \text{until } \Delta < \theta \quad \text{(a small positive number determining accuracy)} \\
& \\
& \text{Output a deterministic policy, } \pi \approx \pi_\ast, \text{ such that} \\
& \quad \pi(s) = \argmax_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]
\end{align*}
$$

## Asynchronous Dynamic Programming

A major drawback of the dynamic programming methods discussed (policy iteration and value iteration) is that they involve iterative sweeps through the entire state set $\mathcal{S}$. If the state set is very large, even a single sweep can be prohibitively expensive.

Asynchronous dynamic programming algorithms are in-place iterative DP algorithms that are not organized in terms of systematic sweeps over the state set. They update the values of states in any order, using whatever values of other states are available. This approach offers great flexibility in selecting which states to update. It can significantly improve efficiency, especially if some state values converge faster than others, and allows computation to be focused on states that are most relevant to the agent's current behavior or situation (e.g., updating only states visited in an ongoing episode). For convergence, the condition is typically that all states continue to be updated eventually.


## Efficiency of Dynamic Programming

Dynamic programming methods can be computationally efficient for solving MDP planning problems when a perfect model is available. Their worst-case time complexity is polynomial in the number of states $|\mathcal{S}|$ and actions $|\mathcal{A}|$. This compares favorably to direct search in policy space (which is exponential) or linear programming methods, especially for large state spaces (though DP still faces challenges with extremely large state spaces, the "curse of dimensionality"). Furthermore, in practice, DP algorithms like value iteration often converge much faster than their theoretical worst-case time bounds suggest.

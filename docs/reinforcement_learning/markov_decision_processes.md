---
title: Markov Decision Process
description: Markov Decision Process and Bellman Equations
sidebar_position: 2
---

A Markov Decision Process (MDP) is a mathematical abstraction for sequential decision-making. Specifically, a Finite Markov Decision Process effectively characterizes the mathematical properties of reinforcement learning, where "finite" refers to the state, action, and reward sets all being finite. We say a system possesses the Markov property if its next state depends solely on the current state; all historical information is implicitly contained within the current state.

In reinforcement learning, we distinguish between the agent and the environment. The agent is the decision-maker, and everything else constitutes the environment. The agent receives a state from the environment and makes a decision (an action). Consequently, it receives a reward and the next state. Note that sometimes it's necessary to differentiate between the state and the observation. The agent might not perceive all aspects of the environment's state, only a partial observation.

## Definition of Markov Decision Process

In a Finite Markov Decision Process, the state and reward follow a fixed probability distribution based on the previous state and action:

$$p(s',r | s,a) \doteq \Pr\{S_t = s', R_t = r | S_{t-1} = s, A_{t-1} = a\}$$

where $s',s \in \mathcal{S}$, $r \in \mathcal{R}$, and $a \in \mathcal{A}(s)$. Here, $p$ characterizes the dynamics of this Markov Decision Process.

From the four-argument dynamics function above, other variables describing the environment can be derived.

For example, the state transition probability distribution is:

$$p(s'|s,a) \doteq \Pr\{S_t = s' | S_{t-1} = s, A_{t - 1}=a\} = \sum_{r\in\mathcal{R}}p(s',r|s,a)$$

The expected reward for a state-action pair is:

$$r(s,a) \doteq \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a] = \sum_{r\in\mathcal{R}}r \sum_{s'\in\mathcal{S}} p(s',r|s,a)$$

The expected reward given the resulting state is:

$$r(s,a, s') \doteq \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a, S_t = s'] = \sum_{r\in\mathcal{R}}r \frac{p(s',r|s,a)}{p(s'|s,a)}$$

## Reward and Return

In reinforcement learning, the agent's objective is to maximize cumulative reward. We define cumulative reward as the return. In its simplest form:

$$G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T$$

However, the return defined above is only suitable for episodic tasks, i.e., tasks with a natural terminal state. It is not applicable to continuous tasks. Generally, for continuous tasks, a discount factor is introduced, meaning future rewards are discounted more heavily.

$$ G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

where $0 \le \gamma \le 1$ is the discount rate. The above equation can be written in a recursive form:

$$ G_t \doteq R_{t+1} + \gamma G_{t+1}$$

Both forms can be unified into one equation (assuming $T=\infty$ or $\gamma=1$ for episodic tasks ending at step $T$):

$$ G_t \doteq \sum_{k=t+1}^T \gamma^{k-t-1}R_k$$

## Policy and Value Function

Almost all reinforcement learning algorithms involve the evaluation of value functions. Value functions measure the expected return for a given state or a given state-action pair. The value function is related to a specific sequence of decision actions, and the corresponding distribution of decision actions is called the policy. A policy $\pi(a|s)$ can be viewed as a probability distribution over actions given a state.

The definition of the state-value function is given as:

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \bigg| S_t = s\right]$$

The definition of the action-value function is similar:

$$q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \bigg| S_t = s, A_t = a\right]$$

State-value and action-value functions can be estimated from experience. One only needs to maintain the average return for each state (or state-action pair) under a specific policy. According to the law of large numbers, this average will eventually converge to the expected value. This is the main idea behind Monte Carlo methods. However, when the number of states is very large, this approach may not be feasible. In such cases, we need to use function approximation to fit the value function, and the effectiveness depends on the choice of the function. Deep learning falls into this category.

Corresponding to the recursive nature of the return function, the state-value function can also be written in a recursive form, known as the Bellman equation.

$$
\begin{align*}
 v_\pi(s) &\doteq \mathbb{E}_\pi[G_t | S_t = s] \\
 &= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
 &= \sum_a \pi(a|s)\sum_{s'}\sum_r p(s',r|s,a)[r + \gamma \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s']] \\
 &= \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]
\end{align*}
$$

A similar Bellman equation can be derived for the action-value function.

$$
\begin{align*}
q_\pi(s,a) &\doteq \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \sum_{s'}\sum_r p(s',r|s,a)[r + \gamma \sum_{a'}\pi(a'|s')\mathbb{E}_\pi[G_{t+1}|S_{t+1}=s', A_{t+1}=a']] \\
&= \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'}\pi(a'|s')q_\pi(s',a')]
\end{align*}
$$

## Optimal Policy and Optimal Value Function

If a policy yields a higher expected value than any other policy for all states, we consider it an optimal policy. It is defined as follows, for all $s \in \mathcal{S}$:

$$ v_\ast(s) \doteq \max_\pi v_\pi(s)$$

There might be multiple optimal policies, but there is only one optimal value function (otherwise, it would violate the definition of an optimal policy).

Correspondingly, there is also an optimal action-value function.

$$q_\ast(s,a) \doteq \max_\pi q_\pi(s,a) $$

They have the following relationship:

$$q_\ast(s,a) = \mathbb{E}[R_{t+1} + \gamma v_\ast(S_{t+1}) | S_t=s, A_t=a] $$

Based on the properties of the optimal policy, the Bellman optimality equation can be derived:

$$
\begin{align*}
v_\ast(s) &= \max_{a \in \mathcal{A}(s)} q_{\pi_\ast}(s, a) \\
&= \max_a \mathbb{E}_{\pi_\ast}[G_t | S_t=s, A_t=a] \\
&= \max_a \mathbb{E}_{\pi_\ast}[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a] \\
&= \max_a \mathbb{E}[R_{t+1} + \gamma v_\ast(S_{t+1}) | S_t=s, A_t=a] \\
&= \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_\ast(s')]
\end{align*}
$$

Similarly, the Bellman optimality equation for the action-value function can be obtained:

$$
\begin{align*}
q_\ast(s,a) &= \mathbb{E}[R_{t+1} + \gamma \max_{a'}q_\ast(S_{t+1}, a')|S_t=s, A_t=a] \\
&= \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'}q_\ast(s',a')]
\end{align*}
$$

Mathematically, it can be shown that the Bellman optimality equation has a unique solution. It amounts to one equation for each state. If there are $|\mathcal{S}|$ states, there are $|\mathcal{S}|$ equations. When the dynamics function is known, we can use any method for solving systems of non-linear equations.

Deriving the optimal policy from the optimal state-value function requires only a one-step search, i.e., finding the action that maximizes the expression in the Bellman optimality equation. For the optimal action-value function, it is even more convenient: the action corresponding to the maximum value for a given state is the best action.

Although the Bellman equation has a theoretical solution, it can rarely be computed directly. Direct computation relies on the following three conditions:

1.  Accurate knowledge of the environment's dynamics function.
2.  Sufficient computational resources.
3.  Satisfaction of the Markov property.

Conversely, many reinforcement learning methods provide corresponding approximate solutions.

## Approximate Optimal Solutions

Besides the conditions mentioned above, memory is also a significant limitation in practice. If the task's state set is small enough, approximate value functions can be stored in memory in tabular form; this is known as the tabular method. However, in many practical examples, the value function cannot fit into memory. In such cases, function approximation must be used to fit the value function.

Approximation might also bring unforeseen benefits. For instance, suboptimal actions in low-probability states do not significantly affect the overall return. The nature of online learning happens to make calculations for high-probability states more accurate. Therefore, computation can be saved without introducing excessive error.

---
title: 蒙特卡洛方法
description: 蒙特卡洛方法-强化学习
updateDate: June 23 2024
---

对于动态规划方法而言，前提是我们完全掌握环境知识。但很多时候，环境知识是未知的或者非常的复杂。这时候蒙特卡洛方法就非常有用了，因为它只需要从与环境交互中采样，不要求有显式的状态转义函数分布。

在蒙特卡洛方法中，我们仍然套用的是GPI框架。只是其中策略评估的这一步被蒙特卡洛预测所取代。

## 蒙特卡洛预测

蒙特卡洛策略一般用于分幕式任务。在给定策略$\pi$下，如何来估计$v_\pi(s)$？一般来说，我们有两种蒙特卡洛采样方式。第一种是首次访问采样法，即每幕中只对第一次访问到此状态的采样。整体算法如下：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Generate an episode following }\pi:\ S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T \\
& \quad G \gets 0 \\
& \quad \text{Loop for each step of episode, }t = T-1,T-2,\cdots,0: \\
& \qquad G \gets \gamma G + R_{t+1} \\
& \qquad \text{Unless } S_t \text{ appears in } S_0,S_1,\cdots,S_{t-1}: \\
& \qquad \quad \text{Append } G \text{ to }Returns(S_t) \\
& \qquad \quad V(S_t) \gets average(Returns(S_t)) \\
\end{align*}
$$

由于分幕式任务中幕之间是相互独立的，因此根据大数定理，当采样足够多时，平均值就是实际值的无偏估计。

另一种采样方式是每次访问采样。这种方式的采样利用率更高，但是收敛性不太直观，实际可以证明它也可以收敛。

蒙特卡洛算法只需要几次幕的计算就可以拟合未知或者复杂的环境是它突出的特色。

蒙特卡洛算法对每个状态的估计是完全独立的，这里面不存在动态规划自举的思想。因此即使只有整体状态集合的一个子集，也可以进行有效计算。

## 蒙特卡洛对动作价值函数的估计

由于缺乏环境模型，在蒙特卡洛方法中，对动作价值函数的估计是更有用的，因为没有显式的状态转移函数，虽然有状态价值函数，也无法确定策略。

将对状态价值函数的估计更改为对动作价值函数的估计，其实现基本一致。唯一需要注意的是，对于某些动作-状态二元组，蒙特卡洛方法可能永远没有访问到。这对于估算$q_\pi(s,a)$是致命的。 没有估计，那么策略就不会提升。这里的问题和K臂老虎机一致，算法需要保持探索。

一种简单的方法是试探性出发。每一组动作-状态二元组都有非零的概率作为初始状态出发。这种方式简单直接，但也有其局限性。比如现实环境不允许任意指定初始状态。

## 蒙特卡洛控制

使用蒙特卡洛方法进行控制即估计近似最优策略的框架仍然是GPI，在基于动作价值函数的估计完成之后，我们可以很容易的选择更优的策略。

即对任意$s \in \mathcal{S}$

$$\pi(s) \doteq \argmax_a(q,a)$$

我们可以证明这是满足策略提升定理的。对于所有$s \in \mathcal{S}$：

$$
\begin{align*}
q_{\pi_k}(s, \pi_{k+1}(s)) &= q_{\pi_k}(s, \argmax_a q_{\pi_k}(s,a)) \\
&= \max_a q_{\pi_k}(s,a) \\
&\ge q_{\pi_k}(s,\pi_k(s)) \\
&\ge v_{\pi_k}(s)
\end{align*}
$$

上述方法的其中一个问题在于对动作价值函数的估计需要大量幕模拟。实际上在动态规划中有类似问题存在，策略评估需要大量迭代，实际上在价值迭代算法中，策略评估可以被剪枝。类似的思想也可以使用在动态规划上。我们不需要等到动作价值函数被精确估计后再进行策略提升。相反，我们可以在每一幕结束后就开始进行策略提升。整体算法如下：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Choose } S_0 \in \mathcal{S}, A_0 \in \mathcal{A}(S_0) \text{ randomly such that all pairs have probability } \ge 0 \\
& \quad \text{Generate an episode from }S_0, A_0,\text{ following }\pi:\ S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T \\
& \quad G \gets 0 \\
& \quad \text{Loop for each step of episode, }t = T-1,T-2,\cdots,0: \\
& \qquad G \gets \gamma G + R_{t+1} \\
& \qquad \text{Unless the pair }S_t, A_t \text{ appears in } S_0,S_1,\cdots,S_{t-1}: \\
& \qquad \quad \text{Append }G \text{ to }Returns(S_t, A_t) \\
& \qquad \quad Q(S_t,A_t) \gets average(Returns(S_t,A_t)) \\
& \qquad \quad \pi(S_t) \gets \argmax_a Q(S_t,a)
\end{align*}
$$

上面这个算法理论收敛性还没有完全被证明，有勇气的可以试试。

## 不对初始点探索的蒙特卡洛控制

如果不对初始点进行探索，我们怎么保证每个状态-动作二元组都被访问到呢？另外有两类方法可以解决这个问题，分别称为同轨策略方法和离轨策略方法。在同轨策略中，采样数据的策略与评估提升的策略是一致的，而离轨策略中则是不一致的。

对于同轨策略方法中，策略一般是软性的，也就是说对所有$s \in \mathcal{S}, a \in \mathcal{A}(s)$，但是最终会接近一个确定性策略。下面介绍的$\varepsilon$-贪心策略就是其中一种。它的思想和K臂老虎机中的$\varepsilon$-贪心方法基本一致。也就是在迭代策略时，大概率贪心的选择当前最好的动作，小概率随机选择所有动作之一，这样可以保证所有状态-动作二元组都有机会被选中。在这种设定下，所有非贪心动作被选中的概率是$\frac{\varepsilon}{|\mathcal{A}(s)|}$，而贪心动作被选中的概率是$1-\varepsilon+\frac{\varepsilon}{|\mathcal{A}(s)|}$。整体算法如下：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Generate an episode from }S_0, A_0,\text{ following }\pi:\ S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T \\
& \quad G \gets 0 \\
& \quad \text{Loop for each step of episode, }t = T-1,T-2,\cdots,0: \\
& \qquad G \gets \gamma G + R_{t+1} \\
& \qquad \text{Unless the pair }S_t, A_t{ appears in }S_0,S_1,\cdots,S_{t-1}: \\
& \qquad \quad \text{Append }G \text{ to }Returns(S_t, A_t) \\
& \qquad \quad Q(S_t,A_t) \gets average(Returns(S_t,A_t)) \\
& \qquad \quad A^\ast \gets \argmax_a Q(S_t,a) \text{(with ties broken arbitrarily)} \\
& \qquad \quad \text{For all a} \in \mathcal{A}(S_t): \\
& \qquad \qquad \pi(a | S_t) \gets
\begin{cases}
1 - \varepsilon+\frac{\varepsilon}{|\mathcal{A}(s)|} &\text{if } a = A^\ast \\
\frac{\varepsilon}{|\mathcal{A}(s)|} &\text{if } a \ne A^\ast
\end{cases}
\end{align*}
$$

我们可以证明$\varepsilon$-贪心策略是满足策略提升定理的。对于任意$s \in \mathcal{S}$:

$$
\begin{align*}
q_\pi(s, \pi'(s)) &= \sum_a \pi'(a|s)q_\pi(s,a) \\
&= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1 - \varepsilon)\max_aq_\pi(s,a) \\
&\ge \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1 - \varepsilon) \sum_a \frac{\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}(s)|}}{1 - \varepsilon}\pi(s,a) \\
&= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) - \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + \sum_a \pi(a|s)q_\pi(s,a) \\
&= v_\pi(s)
\end{align*}
$$

因此有$ \pi' \ge \pi $。下面证明，只有当$\pi'$和$\pi$都为最优$\varepsilon$-软性策略时等号才能成立。

设想一个与原来环境基本相同的新环境，只是将$\varepsilon$-软性移入新环境中，即新环境有$1-\varepsilon$的概率和旧环境表现一致，而有$\varepsilon$的概率会随机等概率选择一个动作，因此新环境的最优策略表现跟旧环境中的最优$\varepsilon$-软性策略效果是一致的。令$\tilde{v_\ast}$和$\tilde{q_\ast}$为新环境中的最优价值函数，当且仅当$v_\pi=\tilde{v_\ast}$是，策略$\pi$是最优$\varepsilon$-软性策略。

根据$\tilde{v_\ast}$的定义，我们知道其是下式的唯一解。

$$
\begin{align*}
\tilde{v_\ast}(s) &= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a \tilde{q_\ast}(s,a) + (1 - \varepsilon)\max_a \tilde{q_\ast}(s,a) \\
&= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a \sum_{s',r} p(s',r|s,a) [r+\tilde{v_\ast}(s')] + (1 - \varepsilon)\max_a \sum_{s',r} p(s',r|s,a) [r+\tilde{v_\ast}(s')] \\
\end{align*}
$$

对应的，当$\pi$无法改进时，下式成立。

$$
\begin{align*}
v_\pi(s) &= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1 - \varepsilon)\max_a q_\pi (s,a) \\
&= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a \sum_{s',r} p(s',r|s,a) [r+v_\pi(s')] + (1 - \varepsilon)\max_a \sum_{s',r} p(s',r|s,a) [r+v_\pi(s')] \\
\end{align*}
$$

两式完全一致，因为$\tilde{v_\ast}$是为一街，因此有$v_\pi=\tilde{v_\ast}$

因此这种同轨策略方法去除了对初始点进行探索，但它只能得到$\varepsilon$-软性策略中的最优策略，不一定是全局最优策略。

## 基于重要度采样的离轨策略

同轨策略为了保持探索性做了折中处理，一个更彻底的思路是将目标策略$\pi$和行动策略$b$分开。通过行动策略的分布去学习目标策略的分布。离轨策略方差更大，收敛更慢，但同时也更强大和通用。

离轨策略需要用到重要度采样的概念，这是一种在给定其他分布的样本下去估计当前样本期望值的通用方法。对于强化学习，在给定起始状态$S_t$，后续的状态-动作轨迹$A_t,S_{t+1},A_{t+1},\cdots,S_T$在策略$\pi$下发生的概率是：

$$
\begin{align*}
& \Pr \{A_t,S_{t+1},A_{t+1},\cdots,S_T | S_t, A_t:T-1 \sim \pi\} \\
&= \pi(A_t|S_t) p(S_{t+1}|S_t, A_t)\pi(A_{t+1}|S_{t+1}) \cdots p(S_T|S_{T-1}, A_{T-1}) \\
&= \prod_{k=t}^{T-1} \pi(A_k|S_k) p(S_{k+1}|S_k, A_k)
\end{align*}
$$

进而可以得到重要性采样比，其中状态转移概率函数上下可以抵消：

$$\rho_{t:T-1} \doteq \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k) p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k) p(S_{k+1}|S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

有了重要性采样比，我们可以正确的在行动策略的数据分布下计算目标策略的期望：

$$ \mathbb{E}[\rho_{t:T-1} G_t | S_t = s] = v_\pi(s) $$

令$\mathcal{T}(s)$为状态$s$被访问到的集合，$T(t)$是$t$之后第一次幕终止的时间。在普通重要度采样算法下，我们得到：

$$ V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1} G_t}{|\mathcal{T}(s)|}$$

另外一种采样方式是加权重要度采样：

$$ V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1}}$$

这两种采样方法在数学上的性质不同。对于首次访问下的普通重要度采样方法是无偏的，但其方差很大。相反的，加权重要度采样是有偏的，其偏差收敛到0，但其方差很小，因此在实践中更多的使用后者。

## 离轨蒙特卡洛控制

有了基于离轨策略的估计之后，我们就可以给出一版基于离轨策略的控制。仍然基于的是GPI的框架，只是此时存在目标策略和行动策略两个策略。目标策略永远是贪心的，而行动策略为了保持探索，使用的是一个$\varepsilon$-软性策略。具体算法如下：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad b \gets \text{any soft policy} \\
& \quad \text{Generate an episode using }b: \ S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T \\
& \quad G \gets 0 \\
& \quad W \gets 1 \\
& \quad \text{Loop for each step of episode, }t = T-1,T-2,\cdots,0: \\
& \qquad G \gets \gamma G + R_{t+1} \\
& \qquad C(S_t,A_t) \gets C(S_t,A_t) + W \\
& \qquad Q(S_t,A_t) \gets Q(S_t,A_t) + \frac{W}{C(S_t,A_t)}[G - Q(S_t,A_t)] \\
& \qquad \pi(S_t) \gets \argmax_a Q(S_t,a) \text{(with ties broken arbitrarily)} \\
& \qquad \text{ If } A_t \ne \pi(S_t) \text{ then exit inner Loop (proceed to next episode)} \\
& \qquad W \gets W \frac{1}{b(A_t|S_t)}
\end{align*}
$$

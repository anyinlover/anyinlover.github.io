---
title: '动态规划'
description: '动态规划-强化学习'
pubDate: 'May 05 2024'
---

如果环境模型是完全已知的，我们就可以用动态规划的思想来求解强化学习问题。但是在现实中很少存在环境模型完全已知的情况。

## 策略评估

当环境模型是已知的，状态价值函数的贝尔曼方程组变成了针对$v_\pi(s)$变量的$|\mathcal{S}|$个线性等式。我们可以使用迭代法来求解，这个算法被称为迭代策略评估。

$$
\begin{align*}
& Loop: \\
& \quad\bigtriangleup \gets 0 \\
& \quad Loop\ for\ each\ s \in \mathcal{S}: \\
& \qquad v \gets V(s) \\
& \qquad V(s) \gets \sum_a\pi(a|s)\sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')] \\
& \qquad \bigtriangleup \gets \max(\bigtriangleup, | v - V(s) |) \\
& until\ \bigtriangleup < \theta
\end{align*}
$$

实际操作中差值小于一个小的$\theta$时我们就认为已经收敛了。

## 策略提升

在已有策略的基础上，我们如何能够找到更好的策略呢？假设在确定性策略下，对于状态$s$，$\pi'(s) = a \neq \pi(s)$，其他状态下的策略保持一致。如果此时$q_\pi(s,a) > v_\pi(s)$，那么我们认为策略$\pi'$比$\pi$更好。这实际上是下面这个策略提升定理的一种特殊情况，如果对于所有$s \in \mathcal{S}$，都满足：

$$q_\pi(s,\pi'(s)) \ge v_\pi(s)$$

此时策略$\pi'$一定比$\pi$同样好或者更好。也就是对于所有$s \in \mathcal{S}$，都满足：

$$v_{\pi'}(s) \ge v_\pi(s)$$

策略提升定理可以推导如下：

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

有了策略提升定理，我们可以用贪心算法得到一个新的更好的策略，由策略提升定理保证了我们贪心解全局也最优。

$$
\begin{align*}
\pi'(s) &\doteq \argmax_a q_\pi(s,a) \\
&= \argmax_a \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t=s, A_t=a] \\
&= \argmax_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_\pi(s')]
\end{align*}
$$

当新的策略效果和老策略一致时，此时：

$$ \pi'(s) = \argmax_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_{\pi'}(s')] $$

这就是贝尔曼最优方程，此时的$\pi'$即$\pi_\ast$。

注意上面的计算是针对确定性策略的，但策略提升定理的推导在概率策略上同样成立。

## 策略迭代

将策略评估和策略提升结合起来，我们就获得了策略迭代算法。由于策略提升算法在达到最优策略前保证稳步提升，同时有限马尔可夫决策过程又保证解空间有限，因此，最后必然可以收敛到最优策略。

注意在策略评估阶段，此时我们将前一个策略的价值函数作为初始值，这样可以更快的收敛。

$$
\begin{align*}
& 1.\ Initialization \\
& \quad V(s) \in \R \ and \ \pi(s) \in \mathcal{A}(s)\ arbitrarily\ for\ all\ s \in \mathcal{S} \\
& 2.\ Policy\ Evaluation \\
& \quad Loop: \\
& \qquad  \bigtriangleup \gets 0 \\
& \qquad Loop\ for\ each\ s \in \mathcal{S}: \\
& \qquad \quad v \gets V(s) \\
& \qquad \quad V(s) \gets \sum_{s',r} p(s',r|s,\pi(s))[r + \gamma V(s')] \\
& \qquad \quad \bigtriangleup \gets \max(\bigtriangleup, | v - V(s) |) \\
& \quad until\ \bigtriangleup < \theta \\
& 3. Policy\ Improvement \\
& \quad stable \gets true \\
& \quad For\ each\ s \in \mathcal{S}: \\
& \qquad old \gets \pi(s) \\
& \qquad \pi(s) \gets \argmax_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')] \\
& \qquad If\ old \ne \pi(s), then\ stable \gets false \\
& \quad If\ stable,\ then\ stop\ and\ return\ V \approx v_\ast\ and\ \pi \approx \pi_\ast;\ else\ go\ to\ 2
\end{align*}
$$

## 价值迭代

策略迭代的问题在于每一次迭代都需要经过一次完整的策略评估，非常的耗时。事实上，可以对策略评估算法进行剪枝，只进行一次状态价值函数的更新。即：

$$ v_{k+1}(s) \doteq \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_k(s')] $$

这也能被看作贝尔曼方程的一种应用。可以证明，这种迭代方式最终可以收敛到$v_\ast$。我们定义一个贝尔曼最优算子$\Tau$：

$$ v_{k+1}(s) = \Tau v_k(s) =  \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v_k(s')] $$

我们引入压缩算子的概念：如果$O$是一个算子，满足$||OV - OV'||_q \le || V - V' ||_q$，则我们称$O$是一个压缩算子。其中$||x||_q$表示$x$的$L_q$范数，无穷范数$||x||_\infty = \max_i|x_i|$。

我们证明贝尔曼最优算子$\tau$是一个$\gamma$-压缩算子。

$$
\begin{align*}
||\Tau v - \Tau v'||_\infty &= \max_{s \in \mathcal{S} }| \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v(s')] - \max_a \sum_{s',r} p(s',r | s,a)[r + \gamma v'(s')]| \\
&\le \max_{s,a}| \sum_{s',r}p(s',r | s,a)[r + \gamma v(s') - r - \gamma v'(s')] \\
&= \gamma \max_{s,a} |\sum_{s',r}p(s',r | s,a)(v(s')-v'(s'))| \\
&\le \gamma \max_{s,a} \sum_{s',r}p(s',r | s,a)\max_{s'}|(v(s')-v'(s')| \\
&= \gamma||v - v'||_\infty
\end{align*}
$$

当$v'$为$v_\ast$时，因此有：

$$||v_{k+1} - v_\ast||_\infty = ||\Tau v_k(s) - \Tau v_\ast(s) ||_\infty \le \gamma ||v_k - v_\ast||_\infty \le \cdots \le \gamma_{k+1}||v_0 - v_\ast||_\infty$$

因此当$\gamma < 1$时，$\lim_{k\rightarrow \infty}v_k = v_\ast$。

整体算法过程如下：

$$
\begin{align*}
& Loop: \\
& \quad\bigtriangleup \gets 0 \\
& \quad Loop\ for\ each\ s \in \mathcal{S}: \\
& \qquad v \gets V(s) \\
& \qquad V(s) \gets \max_a\sum_{s',r} p(s',r|s,a)[r + \gamma v_V(s')] \\
& \qquad \bigtriangleup \gets \max(\bigtriangleup, | v - V(s) |) \\
& until\ \bigtriangleup < \theta \\
& Output\ a\ deterministic\ policy,\ \pi \approx \pi_\ast,\ such\ that \\
& \quad \pi(s) = \argmax_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_V(s')] 
\end{align*}
$$

## 异步动态规划

动态规划方法的一个主要缺点是其迭代涉及到所有状态集合。如果状态集合很大，那么即使是一次迭代都可能非常昂贵。

异步动态规划算法允许我们无序的迭代状态价值函数。这次方式允许我们极大的提升迭代效率，同时还能让智能体在实时应用时一边响应一边更新。这种方式会使得跟当前决策相关的状态价值函数更新更频繁。

## 泛化策略迭代

策略迭代的过程可以被泛化成评估和提升的交互，两个过程实际上可以相对独立的进行，他们是一种竞合关系，最终都达到最优价值函数和最优策略。这种思路在很多强化学习算法上都有体现。

## 动态规划的效率

动态规划的效率相比其他任务其实是比较高的。特别是它能解决状态数量很高时的情况，这是其他如线性规划算法缺乏的。并且动态规划算法在实际中往往比理论最差值更快的收敛。

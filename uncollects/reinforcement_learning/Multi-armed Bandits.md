---
tags:
  - rl
---

# Multi-armed Bandits

## A k-armed Bandit Problem

Consider the following learning problem. You are faced repeatedly with a choice among $k$ different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or time steps.

In our $k$-armed bandit problem, each of the $k$ actions has an expected or mean reward given that that action is selected; let us call this the **value** of that action. We denote the action selected on time step $t$ as $A_t$, and the corresponding reward as $R_t$. The value then of an arbitrary action $a$, denoted $q_*(a)$, is the expected reward given that $a$ is selected:

$$ q_{*}(a) \doteq \mathbb{E}[R_{t} | A_{t} = a] $$

If you knew the value of each action, then it would be trivial to solve the $k$-armed bandit problem: you would always select the action with the highest value. We assume that you do not know the action values with certainty, although you may have estimates. We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$. We would like $Q_t(a)$ to be close to $q_*(a)$.

If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest. We call these the greedy actions. When you select one of these actions, we say that you are **exploiting** your current knowledge of the values of the actions. If instead you select one of the nongreedy actions, then we say you are **exploring**, because this enables you to improve your estimate of the nongreedy action’s value. Exploitation is the right thing to do to maximize the expected reward on the one step, but exploration may produce the greater total reward in the long run.

The need to balance exploration and exploitation is a distinctive challenge that arises in reinforcement learning.

## 平衡探索和利用的四种方法

下面是四种简单但有效的平衡探索和利用的方法。

### $\varepsilon$ 贪婪方法

定义$q_{*}(a)$为在选择动作$a$的期望收益，即价值。按定义这是一个固定值。也是我们在$t$对动作$A_t$选择$a$的收益期望。

$$ q_{*}(a) \doteq \mathbb{E}[R_t|A_t=a] $$

定义$Q_t(a)$为之前选择动作$a$的平均收益，根据大数定理，会逐渐逼近真实的$q_{*}(a)$。

$$ Q_t(a) \doteq \frac{在t之前采取a动作的收益总和}{在t之前采取a动作的次数}
    = \frac{\sum_{i=1}^{t-1}R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}} $$

所谓的$\varepsilon$方法就是每次以$\varepsilon$的概率随机从所有老虎机中选择一个，而以$1 - \varepsilon$的概率选择当前$Q_t$最大的那个。前者是探索，而后者就是利用。

事实上，上面的$Q_t(a)$的计算可以用增量方法简化计算。

$$
\begin{aligned}
    Q_{t+1} &= \frac{1}{t}\sum_{i=1}^{n}R_i \\
            &= \frac{1}{t}(R_t + \sum_{i=1}^{t-1}R_i) \\
            &= \frac{1}{t}(R_t + (t-1)\frac{1}{t-1} \sum_{i=1}^{t-1}R_i) \\
            &= \frac{1}{t}(R_t + (t-1)Q_t) \\
            &= \frac{1}{t}(R_t + tQ_t-Q_t) \\
            &= Q_t + \frac{1}{t}(R_t - Q_t)
    \end{aligned}
$$

### 乐观初始值

乐观初始值基于一个简单的原理，在早期阶段我们应该更鼓励探索而非利用。因此为每个$Q_1(a)$设置一个高于正常范围的估计值，都会导致估计值下降，从而以更大的可能去探索未被探索的动作。

注意这种方法只在平稳问题（即每台老虎机的概率分布固定）中有效，如果时非平稳问题（概率分布会随时间变化）就会失效。

### 置信度上界

在$\varepsilon$ 贪婪方法的探索过程中，是一个完全随机的过程。另一种思想是同时考虑当前的估算值和被选择的次数，去拟合一个置信度，每次选择当前看起来潜力最大的动作。这就是置信度上界（UCB）算法。其公式为：

$$ A_t \doteq \argmax_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right] $$

其中$N_t(a)$表示动作已经被选择的次数，为0时置信度就是无限大。

这个公式实际来源于霍夫丁不等式。此公式的证明需要另开一篇。这里讲如何根据霍夫丁不等式推导得到。

根据霍夫丁不等式，当取值范围为$[0,1]$时，有：

$$P(|\phi - \hat{\phi}| > \gamma) \leq 2\exp(-2\gamma^2m)$$

反之，当不等式右边固定时，可以反求得左边的执行区间$\gamma$。

另右式为$p=\frac{2}{t}$，$m$即为$N_t(a)$,则置信区间

$$ \gamma = \sqrt{\frac{\ln t}{2N_t(a)}}$$

将常数提取到外面，即得到上式。

## 总结

其实人生也是如此，投资亦是如此。

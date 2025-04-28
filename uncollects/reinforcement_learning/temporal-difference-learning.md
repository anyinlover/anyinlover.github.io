
时序差分学习是蒙特卡洛方法和动态规划方法的结合。它既像蒙特卡洛方法那样从经验中学习，又像动态规划方法一样从其他状态中更新评估。时序差分学习是强化学习中最重要的概念。

和往常一样，我们首先聚焦在评估环节，对于控制问题其仍然遵循GPI框架。

## TD预测

对于每次访问的针对非稳态的蒙特卡洛方法，有如下状态价值函数的更新公式：

$$ V(S_t) \gets V(S_t) + \alpha [G_t - V(S_t)] $$

其中$\alpha$是固定步长参数，这种方法被称为常量$\alpha$ MC。对于蒙特卡洛方法而言，必须在本幕结束后才能得到$G_t$。对于TD方法而言，则只需要等到下一步就可以获得更新，这就是最简单的$TD(0)的预测方法：

$$ V(S_t) \gets V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$

$TD(0)$算法是更复杂的$TD(\lambda)$或者n-step TD算法的特殊情况。

可以看到TD方法结合了蒙特卡洛采样方法和动态规划的自举法。后面将会看到，这种方式将发挥出巨大的威力。

注意我们会把括号内的这部分看作是一种误差，衡量的是原有估计值$S_t$和更新的估计值$R_{t+1} + \gamma V(S_{t+1})$之间的差值。这个值被定义为TD误差：

$$ \delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t) $$

如果在幕中价值函数是稳定不变的，如在蒙特卡洛方法下，那么蒙特卡洛误差可以被写作是TD误差的和：

$$
\begin{align*}
G_t - V(S_t) &= R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1}) \\
&= \delta_t + \gamma (G_{t+1} - V(S_{t+1})) \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2 (G_{t+2} - V(S_{t+2})) \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2\delta_{t+2} + \cdots + \gamma^{T-t-1} \delta_{T-1} \\
&= \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k
\end{align*}
$$

如果价值函数会变化，如$TD(0)$，那这个等号无法成立。但如果步长参数足够小，那么等式仍然近似成立。

## TD预测方法的优点

TD预测方法与DP方法相比，有一个优点是不需要有一个环境模型。

和MC方法相比，它是一种自然的渐进学习，而不需要像MC那样等到每一幕结束后再计算回报。这是一个很重要的优势，有些情况下每一幕的步数很多，甚至对连续性任务而言没有分幕。

对于$TD(0)$，任意固定策略$\pi$在小步长下被证明可以收敛到$v_\pi$。这个证明对于所有表格型任务和部分线性函数近似任务有效。

## TD(0)的最优性

当经验样本有限时，一种常见的增量学习方法是是重复的学习这些经验直到收敛。对每一个时刻的某个状态而言，所有的增量可以被累加计算。这种方式称为批量更新。

这种方式下$TD(0)$收敛的值与步长参数无关。对于常量$\alpha$ MC而言也会收敛到某个值，但这两个值会不同。这里面体现了MC和TD的本质区别。对于MC而言，最终收敛值是样本集的均值，而对于TD而言，其虽然没有在训练数据上学习的更好，但他通过自举捕捉到了马尔可夫过程模型的内在性质，也就是把先验知识带进了估计之中。对于批量$TD(0)$而言，其学习的其实是马尔可夫过程模型的最大似然估计。这种估计被称为确定性等价估计。

注意对于非批量TD算法，其不满足确定性等价估计。等仍然比较接近，因此其和批量TD算法一样，收敛性一般也高于常量$\alpha$ MC。

## Sarsa同轨TD控制

理解了$TD(0)$，再来理解Sarsa是比较容易的。Sarsa属于同轨算法的一种。在Sarsa中，$TD(0)$更新的是动作价值函数。

$$ Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t)]$$

下面是Sarsa算法的过程：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Initialize } S\\
& \quad \text{Choose } A \text{ From } S \text{ using policy derived from } Q \text{ (e.g., } \varepsilon \text{-greedy)}\\\
& \quad \text{Loop for each step of episode:} \\
& \qquad \text{Take action }A, \text{ obeserve }R, S' \\
& \qquad \text{Choose } A' \text{ from } S' \text{ using policy derived from } Q text{ (e.g., } \varepsilon \text{-greedy)} \\
& \qquad Q(S, A) \gets Q(S, A) + \alpha[R+ \gamma Q(S', A') - Q(S,A)] \\
& \qquad S \gets S'; A \gets A' \\
& \quad \text{until S is terminal}
\end{align*}
$$

Sarsa的收敛性由策略对$Q$的依赖决定。只要策略能给访问到无数个状态-动作对，并且策略逐渐能收敛到贪心策略（如$\varepsilon$-贪心策略中令$\varepsilon=1/t$）。

## Q-learning异轨TD控制

Q-learning是一种流行的离轨策略算法，其更新公式为：

$$ Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_aQ(S_{t+1}, a) - Q(S_t,A_t)]$$

下面是Q-learning算法的过程：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Initialize } S\\
& \quad \text{Loop for each step of episode:} \\
& \qquad \text{Choose } A \text{ From } S \text{ using policy derived from } Q \text{ (e.g., } \varepsilon \text{-greedy)}\\\
& \qquad \text{Take action }A, \text{ obeserve }R, S' \\
& \qquad Q(S, A) \gets Q(S, A) + \alpha[R+ \gamma \max_a Q(S', a) - Q(S,A)] \\
& \qquad S \gets S' \\
& \quad \text{until S is terminal}
\end{align*}
$$

Q-learning与Sarsa相比最大的差异就是Q-learning使用的是$(s,a,r,s')$四元组，其更新的是$Q(s,a)$，而$r$和$s'$都是从环境采样得到。一般是由另一个行为策略中采集而来。由于离轨策略能给重复使用历史的训练样本，这种方式常常更受欢迎。

Q-learning算法可以被证明可以严格收敛。

## 期望sarsa

期望sarsa是另一种类似Q-learning的算法，其不使用贪心算法更新动作价值函数，而是使用期望：

$$
\begin{align*}
Q(S_t, A_t) &\gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1}) | S_{t+1}]- Q(S_t,A_t)] \\
&\gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \sum_a\pi(a|S_{t+1})Q(S_{t+1},a)- Q(S_t,A_t)]
\end{align*}
$$

期望sarsa既可以作为在轨策略，也可以作为离轨策略。

## 最大化偏差与双学习

在sarsa和Q-learning中都存在着偏好最大化值的问题，这个问题被称为最大化偏差。一个被称为双学习的算法可以用来解决这个问题。

如果我们有两个独立的估计，一个估计用来评估价值函数，另一个估计用来得到最大动作。这时候的估计就是无偏的。这两个估计可以独立的交叉的进行评估。如下是双Q-learning下的更新公式：

$$ Q_1(S_t, A_t) \gets Q_1(S_t, A_t) + \alpha[R_{t+1} + \gamma Q_2(S_{t+1},\argmax_aQ_1(S_{t+1}, a)) - Q_1(S_t,A_t)]$$

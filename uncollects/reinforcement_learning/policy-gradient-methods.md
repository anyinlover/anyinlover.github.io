# 策略梯度方法

之前所有的方法都是基于动作价值函数的，但本质上我们想学的是一个最优策略。因此我们也可以将策略参数化，从而来学习策略的概率分布$\pi(a|s, \theta) = \Pr\{A_t=a | S_t=s, \theta_t=\theta\}$。我们将这类从目标函数$J(\theta)$中学习参数$\theta$的方法统一都称为策略梯度方法。由于是为了最大化效果，我们的梯度更新公式一般是：

$$ \theta_{t+1} = \theta_t + \alpha \widehat{\nabla J(\theta_t)}$$

这类通过梯度学习近似的策略函数被称为策略梯度法，而对其中同时学习策略和价值函数的方法被称为行动器-评判器方法。

## 策略近似及其优点

在策略梯度方法中，为了保持探索，一般要求策略都不是确定的。即$\pi(a|s, \theta) \in (0,1)$。当动作空间是离散的且不是特别大。我们可以对每个状态-动作对估计一个参数化的数值偏好 $h(s,a,\theta)$，进而得到参数化的概率分布：

$$ \pi(a|s, \theta) \doteq \frac{e^{h(s,a,\theta)}}{\sum_be^{h(s,b,\theta)}}$$

这种基于柔性最大化分布的好处是近似策略可以接近一个确定策略，但永远不会成为特定策略。同时，它能够以任意概率选择动作。

策略参数化的一个最简单的优势在于往往策略参数化比动作参数化更简单。同时，策略参数化还允许我们引入对策略的先验知识。

## 策略梯度定理

策略参数化由于是连续的，其函数比较平滑，因此其收敛性会比较好。

这里我们首先考虑分幕式的情况，对于分幕式来说，优化目标如下：

$$ J(\theta) \doteq v_{\pi_\theta}(s_0)$$

即找到一个尽可能好的策略，使得其在初始状态的真实价值最高。为了简化起见，这里先忽略折扣$\gamma$。

我们有如下定理保证在分幕式下这种策略梯度方法能够获得效果收益。

$$ \nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi (s,a) \nabla \pi(a|s, \theta) $$

证明如下：

$$
\begin{align*}

\nabla v_\pi(s) &= \nabla \left[\sum_a \pi(a|s) q_\pi(s, a) \right] \\
&= \sum_a[\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \nabla q_\pi (s,a)] \\
&= \sum_a[\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \nabla \sum_{s',r} p(s',r|s,a)(r + v_\pi(s'))] \\
&= \sum_a[\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \sum_{s'} p(s'|s,a)\nabla v_\pi(s')] \\
&= \sum_a[\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \sum_{s'} p(s'|s,a) \sum_{a'}[\nabla \pi(a'|s') q_\pi(s',a') + \pi(a'|s') \sum_{s''} p(s''|s',a')\nabla v_\pi(s'')]] \\
&= \sum_{x \in \mathcal{S}} \sum_{k=0}^\infty \Pr(s \rightarrow x,k,\pi) \sum_a \nabla \pi(a|x)q_\pi(x,a) \\

\nabla J(\theta) &= \nabla v_\pi(s_0) \\
&= \sum_s \sum_{k=0}^\infty \Pr(s_0 \rightarrow s,k,\pi) \sum_a \nabla \pi(a|s)q_\pi(x,a) \\
&= \sum_s \eta(s) \sum_a \nabla \pi(a|s)q_\pi(x,a) \\
&= \sum_{s'} \eta(s') \sum_s \frac{\eta(s)}{\sum_{s'}\eta(s')}\sum_a \nabla \pi(a|s)q_\pi(x,a) \\
&= \sum_{s'} \eta(s') \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(x,a) \\
&\propto \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(x,a)

\end{align*}
$$

这里的$\mu$是策略$\pi$下的同轨策略分布。对于分幕式任务而言，正比的系数即$\sum_{s'} \eta(s')$，等于幕的平均长度。而对于连续性任务而言，这个值等于1。

## REINFORCE蒙特卡洛策略梯度

有了策略梯度定理，我们希望我们的采样样本的梯度近似于策略梯度定理右式，注意到右式可以看作目标策略$\pi$下每个状态出现的频率加权，因此可以看作为采样的期望

$$
\begin{align*}
\nabla J(\theta) &\propto \sum_s \mu(s) \sum_a q_\pi (s,a) \nabla \pi(a|s, \theta) \\
&= \mathbb{E}_\pi \left[ \sum_a q_\pi(S_t,a) \nabla \pi(a|S_t, \theta) \right] \\
&= \mathbb{E}_\pi \left[ \sum_a \pi(a|S_t, \theta)q_\pi(S_t,a) \frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)} \right] \\
&= \mathbb{E}_\pi \left[ q_\pi(S_t,A_t) \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right] \\
&= \mathbb{E}_\pi \left[ G_t \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right] \\
\end{align*}
$$

最终我们可以通过每步采样来得到更新的梯度：

$$\theta_{t+1} \doteq \theta_t + \alpha G_t \frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}$$

这个公式有直观的理解，每一次增量更新都和回报$G_t$和一个向量成正比，整个向量是选取动作的概率的梯度除以概率本身。

下面就是REINFORCE算法的整体流程：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Generate an episode following }\pi:\ S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T, \text{ following } \pi(\cdot|\cdot,\theta) \\
& \quad \text{Loop for each step of the episode } t = 0,1,\cdots,T-1: \\
& \qquad G \gets \sum_{k=t+1}^T \gamma^{k-t-1}R_k \\
& \qquad \theta \gets \theta + \alpha \gamma^t G \nabla \ln \pi(A_t | S_t, \theta)
\end{align*}
$$

REINFORCE作为一种随机梯度方法，有很好的理论收敛保证。但作为蒙特卡洛方法，其可能有较高的方差，因此导致学习较慢。

## 带基线的REINFORCE

策略梯度定理可以进行推广，加入任何一个于动作价值函数对比的基线$b(s)$而仍然成立：

$$ \nabla J(\theta) \propto \sum_s \mu(s) \sum_a (q_\pi (s,a) - b(s))\nabla \pi(a|s, \theta) $$

因为减除的那项整体贡献恒等于0

$$ \sum_a b(s) \nabla \pi(a|s, \theta) = b(s) \sum_a \nabla \pi(a|s, \theta) = b(s) \nabla 1 = 0 $$

这样可以导出带基线的REINFORCE版本：

$$\theta_{t+1} \doteq \theta_t + \alpha (G_t - b(S_t))\frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}$$

选择一个好的基线有助于降低方差。状态价值函数$\hat{v}(S_t, w)$是一个比较自然的基线。

下面是带基线的REINFORCE算法：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Generate an episode following }\pi:\ S_0,A_0,R_1,S_1,A_1,R_2,\cdots,S_{T-1},A_{T-1},R_T, \text{ following } \pi(\cdot|\cdot,\theta) \\
& \quad \text{Loop for each step of the episode } t = 0,1,\cdots,T-1: \\
& \qquad G \gets \sum_{k=t+1}^T \gamma^{k-t-1}R_k \\
& \qquad \delta \gets G - \hat{v}(S_t, w) \\
& \qquad w \gets w+\alpha^w \delta \nabla \hat{v}(S_t,w) \\
& \qquad \theta \gets \theta + \alpha^\theta \gamma^t G \nabla \ln \pi(A_t | S_t, \theta)
\end{align*}
$$

其中$\alpha^w$是一个相对容易设置的步长，如$\alpha^w = 0.1/ \mathbb{E}[||\nabla \hat{v}(S_t, w)||_\mu^2]$，而$\alpha^\theta$则跟具体的任务相关。

## 行动器评判器方法

在带基线的REINFORCE算法中，虽然学习了状态价值函数，但实际上并没有作为评判器使用，类似于蒙特卡洛方法和TD方法，我们实际上也可以把状态价值函数用作自举中，这种方式可以降低方差并加快学习。

对于单步行动器-评判器来说，单步回报可以替代REINFORCE算法：

$$
\begin{align*}
\theta_{t+1} & \doteq \theta_t + \alpha (G_{t:t+1} - \hat{v}(S_t,w))\frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)} \\
&= \theta_t + \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1},w) - \hat{v}(S_t,w))\frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)} \\
&= \theta_t + \alpha \delta_t \frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}
\end{align*}
$$

其整体算法如下：

$$
\begin{align*}
& \text{Loop forever (for each episode)}: \\
& \quad \text{Initialize }S \text{ (first state of episode)} \\
& \quad I \gets 1 \\
& \quad \text{Loop while } S \text{ is not terminal (for each time step):} \\
& \qquad A \sim \pi(\cdot | S, \theta) \\
& \qquad \text{Take action } A, \text {observe } S', R \\
& \qquad \delta \gets R + \gamma \hat{v}(S', w)- \hat{v}(S, w) \\
& \qquad w \gets w+\alpha^w \delta \nabla \hat{v}(S,w) \\
& \qquad \theta \gets \theta + \alpha^\theta I \delta \nabla \ln \pi(A | S, \theta) \\
& \qquad I \gets \gamma I \\
& \qquad S \gets S'
\end{align*}
$$

## 持续性问题的策略梯度

## 针对连续动作的策略参数化方法

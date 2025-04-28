# 随机下降提升方法

本文主要分析了Friedman经典的《Greedy Function Approximation: A Gradient Boosting Machine》论文，也是xgboost、catboost等一众流行算法的理论基础。

Gradient Boosting 的主要思想核心即通过Boosting的方式在函数空间上用梯度下降求解最小值，逐步逼近目标结果。

## 理论推导

### 函数估计

监督学习的本质就是根据训练数据集$\{y_i, \bold{x}_i\}^N_1$，找到一个最优的映射函数$F^*(\bold{x})$，使得在联合分布$(y,\bold{x})$上损失函数$L(y, F(\bold{x}))$的期望值最小。

$$ F^* = \arg \min_F E_{y,\bold{x}}L(y, F(\bold{x})) = \arg \min_F E_\bold{x} [ E_y(L(y, F(\bold{x}))) | \bold{x} ]$$

在本文中主要探讨提升方法下的函数形式：

$$ F(\bold{x}; \{ \beta_m, \bold{a}_m\}^M_1) = \sum^M_{m=1}\beta_m h(x;\bold{a}_m)$$

#### 数值优化

一般的，选择一个参数模型$F(\bold{x};\bold{P})$​可以将函数优化问题转化为参数优化问题，其中$\Phi(\bold{P}) = E_{y,\bold{x}}L(y, F(\bold{x}; \bold{P}))$:

$$ P^* = \arg \min_\bold{P} \Phi(\bold{P})$$

得到$\bold{P}^*$之后，也就得到了$F^*(\bold{x}) = F(\bold{x}; \bold{P}^*)$。

在提升方法下，$\bold{P}^*$需要通过优化方法逐步计算得到：

$$ \bold{P}^* = \sum^M_{m=0}\bold{p}_m $$

#### 梯度下降

最小梯度是最简单和常用的数值优化方法。计算梯度如下：

$$ \bold{g}_m = \{g_{jm}\} = \{[\frac{\partial\Phi(\bold{P})}{\partial P_j}]_{\bold{P}=\bold{P}_{m-1}}\}$$

再通过线性搜索的方法找到$\rho_m$，从而$\bold{p}_m = -\rho_m \bold{g}_m$​：

$$ \rho_m = \arg \min_\rho \Phi(\bold{P}_{m-1} - \rho \bold{g}_m)$$

### 函数空间的优化

如果把$F(\bold{x})$​看作一个整体，则优化问题可以转换到函数空间上求解。即找到最合适的那个$F(\bold{x})$​，使得目标函数最小。

$$\Phi(F) = E_{y,\bold{x}}L(y, F(\bold{x})) = E_\bold{x}[E_y(L(y, F(\bold{x}))) | \bold{x}]$$

即 $ \Phi(F(\bold{x})) = E_y[L(y, F(\bold{x})) | \bold{x}]$​，求解过程和上面求解$\bold{P}^*$类似：

$$ F^*(\bold{x}) = \sum_{m=0}^M f_m(\bold{x}) $$

其中：

$$ f_m(\bold{x}) = -\rho_m g_m(\bold{x}) $$

$$g_m(\bold{x}) = [\frac{\partial\Phi(F(\bold{x}))}{\partial F(\bold{x})}]_{F(\bold{x})=F_{m-1}(\bold{x})} = [\frac{\partial E_y[L(y, F(\bold{x})) | \bold{x}]}{\partial F(\bold{x})}]_{F(\bold{x})=F_{m-1}(\bold{x})}$$

$$ \rho_m = \arg \min_\rho E_{y,\bold{x}}L(y, F_{m-1}(\bold{x}) - \rho g_m(\bold{x})) $$

根据[莱布尼茨积分法则](https://en.wikipedia.org/wiki/Leibniz_integral_rule)，积分范围是常数时积分和微分可以互换，进一步可以得到：

$$ g_m(\bold{x}) = E_y[\frac{\partial L(y, F(\bold{x}))}{\partial F(\bold{x})} | \bold{x}]_{F(\bold{x})=F_{m-1}(\bold{x})}$$

### 数值计算

对于有限样本数据集$\{y_i, \bold{x}_i\}^N_1$而言，求解$\bold{P}^*$的方式变成了基于数据估算期望损失：

$$
\{\beta_m, \bold{a}_m\}_1^M = \arg \min_{\{\beta'_m, \bold{a}'_m\}_1^M}\sum_{i=1}^N L(y_i, \sum_{m=1}^M \beta'_m h(x_i;\bold{a}'_m))
$$

一般通过贪心法求解：

$$(\beta_m, \bold{a}_m) = \arg \min_{\beta, \bold{a}}\sum_{i=1}^N L(y_i, F_{m-1}(x_i) + \beta h(x_i;\bold{a}))$$

$$ F_m(\bold{x}) = F_{m-1}(\bold{x}) + \beta_m h(\bold{x}; \bold{a}_m) $$

函数$h(\bold{x};\bold{a})$被称为基函数或者弱学习器。上面的$\beta_m h(\bold{x}; \bold{a}_m)$​可以被看做基于基函数约束的最大梯度下降。

另一方面，无约束的负梯度可以计算得到：

$$-g_m(\bold{x}_i) = -[\frac{\partial L(y_i, F(\bold{x}_i))}{\partial F(\bold{x}_i)}]_{F(\bold{x})=F_{m-1}(\bold{x})}$$

一个直观的想法就是让带约束的梯度下降方向尽可能的与无约束的负梯度方向平行，用最小二乘衡量平行度：

$$ \bold{a}_m = \arg \min_{\bold{a}, \beta} \sum_{i=1}^N [-g_m(\bold{x}_i) - \beta h(\bold{x}_i;\bold{a})]^2$$

继续用上面线性搜索的方法求解$\rho_m$：

$$ \rho_m = \arg \min_\rho \sum_{i=1}^N L(y_i, F_{m-1}(\bold{x}_i) + \rho h(\bold{x}_i; \bold{a}_m))$$

最后更新$F_m(\bold{x})$：

$$ F_m(\bold{x}) = F_{m-1}(\bold{x}) + \rho_m h(\bold{x}; \bold{a}_m) $$

这里的关键点就是通过梯度近似，使得弱分类器$h(\bold{x}; \bold{a})$拟合负梯度$\{\tilde{y_i} = -g_m(\bold{x}_i)\}^N_{i=1}$，从而将问题简化。

### 算法总结

Gradient Boosting的整体算法如下：
$$
\begin{align*}
& F_0(\bold{x}) = \arg \min_\rho \sum_{i=1}^N L(y_i, \rho) \\
& For\ m = 1\ to\ M\ do: \\
& \qquad \tilde{y_i} = -[\frac{\partial L(y_i, F(\bold{x}_i))}{\partial F(\bold{x}_i)}]_{F(\bold{x})=F_{m-1}(\bold{x})}, i = 1 \dots N \\
& \qquad \bold{a}_m = \arg \min_{\bold{a}, \beta} \sum_{i=1}^N [\tilde{y_i} - \beta h(\bold{x}_i;\bold{a})]^2 \\
& \qquad \rho_m = \arg \min_\rho \sum_{i=1}^N L(y_i, F_{m-1}(\bold{x}_i) + \rho h(\bold{x}_i; \bold{a}_m)) \\
& \qquad F_m(\bold{x}) = F_{m-1}(\bold{x}) + \rho_m h(\bold{x}; \bold{a}_m) \\
& endFor
\end{align*}
$$

## 应用

下面尝试将算法应用到不同的损失函数和模型上。

### 最小二乘回归

当损失函数为最小二乘时，此时的梯度$\tilde{y_i}$即残差，由于损失函数与梯度近似时的损失函数一致，此时的$\rho_m = \beta_m$，算法整体会非常简化：
$$
\begin{align*}
& F_0(\bold{x}) = \bar{y} \\
& For\ m = 1\ to\ M\ do: \\
& \qquad \tilde{y_i} = y_i - F_{m-1}(\bold{x}_i), i = 1 \dots N \\
& \qquad (\rho_m, \bold{a}_m) = \arg \min_{\bold{a}, \rho} \sum_{i=1}^N [\tilde{y_i} - \rho h(\bold{x}_i;\bold{a})]^2 \\
& \qquad F_m(\bold{x}) = F_{m-1}(\bold{x}) + \rho_m h(\bold{x}; \bold{a}_m) \\
& endFor
\end{align*}
$$

### 回归树

当基函数是一棵树的时候，问题会变得更有意思。此时的基函数可以被定义为：

$$ h(\bold{x}; \{b_j, R_j\}^J_1) = \sum_{j=1}^J b_j 1(\bold{x} \in R_j) $$

在这种情况下，更新公式变成：

$$ F_m(\bold{x}) = F_{m-1}(\bold{x}) + \rho_m \sum_{j=1}^J b_{jm} 1(\bold{x} \in R_{jm})$$

这里$\{R_{jm}\}_1^J$​是基函数树在拟合$\{\tilde{y_i}\}_1^N$时构造的，$b_{jm}$是对应的最小二乘系数：

$$ b_jm = ave_{\bold{x}_i \in R_{jm}} \tilde{y_i}$$

令$\gamma_{jm} = \rho_m b_{jm}$，进一步得到：

$$F_m(\bold{x}) = F_{m-1}(\bold{x}) + \sum_{j=1}^J \gamma_{jm} 1(\bold{x} \in R_{jm})$$

通过将$\rho_m$內移，可以更精细的进行最优化求解。

$$ \{\gamma_{jm}\}_1^J = \arg \min_{\{\gamma_j\}_1^J} \sum_{i=1}^N L(y_i, F_{m-1}(\bold{x}_i) + \sum_{j=1}^J \gamma_j 1(\bold{x} \in R_{jm})) $$

由于$R_{jm}$相互独立，上述求解可以独立进行：

$$ \gamma_{jm} = \arg \min_\gamma \sum_{\bold{x}_i \in R_{jm}} L(y_i, F_{m-1}(\bold{x}_i)+\gamma)$$

当损失函数为LAD回归时，此时可以得到：

$$ \gamma_{jm} = median_{x_i \in R_{jm}} \{y_i - F_{m-1}(x_i)\}$$

最后的算法会非常稳定，因为残差取值范围只有两个，$\tilde{y_i} \in \{-1,1\}$​

$$
\begin{align*}
& F_0(\bold{x}) = median\{y_i\}^N_1 \\
& \qquad For\ m = 1\ to\ M\ do: \\
& \qquad \tilde{y_i} = sign(y_i - F_{m-1}(\bold{x}_i)), i = 1,N \\
& \qquad \{R_{jm}\}_1^J = J-terminal\ node\ tree(\{\tilde{y_i}, \bold{x}_i\}_1^N) \\
& \qquad \gamma_{jm} = median_{\bold{x}_i \in R_{jm}} \{y_i - F_{m-1}(\bold{x}_i)\}, j = 1, J \\
& \qquad F_m(\bold{x}) = F_{m-1}(\bold{x}) + \sum_{j=1}^J \gamma_{jm} 1(\bold{x} \in R_{jm}) \\
& endFor
\end{align*}
$$

### 二分类

在二分类问题中，损失函数被定义为：

$$ L(y, F) = \log (1 + \exp(-2yF)), y \in \{-1,1\} $$

其中$F(\bold{x}) = \frac{1}{2} \log\frac{Pr(y=1|\bold{x})}{Pr(y=-1|\bold{x})}$​​，此时可以推导出：

$$
\tilde{y_i} = -\left[\frac{\partial L(y_i, F(\bold{x}_i))}{\partial F(\bold{x}_i)}\right]_{F(\bold{x})=F_{m-1}(\bold{x})} = 2y_i/(1+\exp(2y_iF_{m-1}(\bold{x}_i)))
$$

如果以回归树作为基函数，同样可以得到：

$$ \gamma_{jm} = \arg \min_\gamma \sum_{\bold{x}_i \in R_{jm}} \log (1 + \exp(-2y_i(F_{m-1}(\bold{x}_i) + \gamma))) $$

使用牛顿法进行近似求解：

$$
y'_i = - \tilde{y_i} = - 2y_i/(1+\exp(2y_iF_{m-1}(\bold{x}_i))) \\
y''_i = \left[\frac{\partial L(y_i, F(\bold{x}_i))}{\partial^2 F(\bold{x}_i)}\right]_{F(\bold{x})=F_{m-1}(\bold{x})} = -\frac{4\exp(2y_iF_{m-1}(\bold{x}_i))}{(1+\exp(2y_iF_{m-1}(\bold{x}_i)))^2} = -\frac{2}{1+\exp(2y_iF_{m-1}(\bold{x}_i))} * (2 - \frac{2}{1+\exp(2y_iF_{m-1}(\bold{x}_i))}) = |\tilde{y_i}|(2 - |\tilde{y_i}|) \\
\gamma_{jm} = -\frac{\sum_{\bold{x}_i \in R_{jm}}y'_i}{\sum_{\bold{x}_i \in R_{jm}}y''_i} = \frac{\sum_{\bold{x}_i \in R_{jm}}\tilde{y_i}}{\sum_{\bold{x}_i \in R_{jm}}|\tilde{y_i}|(2 - |\tilde{y_i}|)}
$$

因此二分类的整体算法如下：

$$
\begin{align*}
& F_0(\bold{x}) = \frac{1}{2} \log \frac{1+\bar y}{1 - \bar y} \\
& \qquad For\ m = 1\ to\ M\ do: \\
& \qquad \tilde{y_i} = 2y_i/(1 + \exp(2y_iF_{m-1}(\bold{x}_i))), i = 1 \dots N \\
& \qquad \{R_{jm}\}_1^J = J-terminal\ node\ tree(\{\tilde{y_i}, \bold{x}_i\}_1^N) \\
& \qquad \gamma_{jm} = \frac{\sum_{\bold{x}_i \in R_{jm}}\tilde{y_i}}{\sum_{\bold{x}_i \in R_{jm}}|\tilde{y_i}|(2 - |\tilde{y_i}|)} \\
& \qquad F_m(\bold{x}) = F_{m-1}(\bold{x}) + \sum_{j=1}^J \gamma_{jm} 1(\bold{x} \in R_{jm}) \\
& endFor
\end{align*}
$$

最后得到二分类的概率：

$$
p_+(\bold{x}) = \hat{Pr}(y = 1 | \bold{x})  = 1/(1 + e^{-2F_M(\bold{x})}) \\
p_-(\bold{x}) = \hat{Pr}(y = -1 | \bold{x})  = 1/(1 + e^{2F_M(\bold{x})})
$$

#### 计算剪枝

注意到对于损失函数而言，可以拆成两部分：

$$ \phi(\rho, \bold{a}) = \sum_{i=1}^N \log(1+\exp(-2y_iF_{m-1})) \exp(-2y_i\rho h(\bold{x}_i;\bold{a}))) $$

当$y_iF_{m-1}$​过大时，此项整体值接近于0，可以忽略计算。从函数空间角度看，这意味着那些梯度下降的厉害的影响更大，可以通过二阶导衡量，即我们上面已经计算得到的：

$$ \omega_i = |\tilde y_i|(2 - |\tilde y_i|)  $$

通过对所有$\omega_i$​​​升序排序，删除前$l(\alpha)$​​个，可以有效减少计算量，下降幅度能达到10到20倍。下面时$l(\alpha)$​​​的计算公式，其中一般情况下$\alpha \in [0.05,0.2]$​。

$$ \sum_{i=1}^{l(\alpha)} \omega_{(i)} = \alpha \sum_{i=1}^N \omega_i $$

## 正则化

为了防止过拟合，需要一些正则化手段控制。最直观的就是参数M。迭代次数越多，越容易过拟合。

另外也可以使用缩减技术。在这种方式下，$F_m(\bold{x})$的更新方式变成如下：

$$ F_m(\bold{x}) = F_{m-1}(\bold{x}) + v\rho_mh(\bold{x};\bold{a}_m) ,  0 < v \leq 1$$

正则化参数的选择需要通过模型选择方法进行调优。

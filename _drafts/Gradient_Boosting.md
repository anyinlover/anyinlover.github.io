# 随机下降提升方法

本文主要分析了Friedman经典的《Greedy Function Approximation: A Gradient Boosting Machine》论文，也是xgboost、catboost等一众流行算法的理论基础。

Gradient Boosting 的主要思想核心即通过Boosting的方式在函数空间上用梯度下降求解最小值，逐步逼近目标结果。

## 理论推导

### 函数估计

监督学习的本质就是根据训练数据集$\{y_i, x_i\}^N_1$，找到一个最优的映射函数$F^*(x)$，使得在联合分布$(y,x)$上损失函数$L(y, F(x))$的期望值最小。

$$ F^* = \arg \min_F E_{y,x}L(y, F(x)) = \arg \min_F E_x [ E_y(L(y, F(x))) | x ]$$

在本文中主要探讨提升方法下的函数形式：

$$ F(x; \{ \beta_m, a_m\}^M_1) = \sum^M_{m=1}\beta_m h(x;a_m)$$

#### 数值优化

一般的，选择一个参数模型$F(x;P)$​可以将函数优化问题转化为参数优化问题，其中$\Phi(P) = E_{y,x}L(y, F(x; P))$:

$$ P^* = \arg \min_P \Phi(P)$$

得到$P^*$之后，也就得到了$F^*(x) = F(x; P^*)$。

在提升方法下，$P^*$需要通过优化方法逐步计算得到：

$$ P^* = \sum^M_{m=0}p_m $$

#### 梯度下降

最小梯度是最简单和常用的数值优化方法。计算梯度如下：

$$ g_m = \{g_{jm}\} = \{[\frac{\partial\Phi(P)}{\partial P_j}]_{P=P_{m-1}}\}$$

再通过线性搜索的方法找到$\rho_m$，从而$p_m = -\rho_m g_m$​：

$$ \rho_m = \arg \min_\rho \Phi(P_{m-1} - \rho g_m)$$

### 函数空间的优化

如果把$F(x)$​看作一个整体，则优化问题可以转换到函数空间上求解。即找到最合适的那个$F(x)$​，使得目标函数最小。

$$\Phi(F) = E_{y,x}L(y, F(x)) = E_x[E_y(L(y, F(x))) | x]$$

即 $ \Phi(F(x)) = E_y[L(y, F(x)) | x]$​，求解过程和上面求解$P^*$类似：

$$ F^*(x) = \sum_{m=0}^M f_m(x) $$

其中：

$$ f_m(x) = -\rho_m g_m(x) $$

$$g_m(x) = [\frac{\partial\Phi(F(x))}{\partial F(x)}]_{F(x)=F_{m-1}(x)} = [\frac{\partial E_y[L(y, F(x)) | x]}{\partial F(x)}]_{F(x)=F_{m-1}(x)}$$​

$$ \rho_m = \arg \min_\rho E_{y,x}L(y, F_{m-1}(x) - \rho g_m(x)) $$​

根据[莱布尼茨积分法则](https://en.wikipedia.org/wiki/Leibniz_integral_rule)，积分范围是常数时积分和微分可以互换，进一步可以得到：

$$ g_m(x) = E_y[\frac{\partial L(y, F(x))}{\partial F(x)} | x]_{F(x)=F_{m-1}(x)}$$

### 数值计算

对于有限样本数据集$\{y_i, x_i\}^N_1而言，求解$$P^*$的方式变成了基于数据估算期望损失：

$$\{\beta_m, a_m\}_1^M = \arg \min_{\{\beta'_m, a'_m\}_1^M}\sum_{i=1}^N L(y_i, \sum_{m=1}^M \beta'_m h(x_i;a'_m))$$​

一般通过贪心法求解：

$$(\beta_m, a_m) = \arg \min_{\beta, a}\sum_{i=1}^N L(y_i, F_{m-1}(x_i) + \beta h(x_i;a))$$​​

$$ F_m(x) = F_{m-1}(x) + \beta_m h(x; a_m) $$

函数$h(x;a)$被称为基函数或者弱学习器。上面的$\beta_m h(x; a_m)$​可以被看做基于基函数约束的最大梯度下降。

另一方面，无约束的负梯度可以计算得到：

$$-g_m(x_i) = -[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}]_{F(x)=F_{m-1}(x)}$$​

一个直观的想法就是让带约束的梯度下降方向尽可能的与无约束的负梯度方向平行，用最小二乘衡量平行度：

$$ a_m = \arg \min_{a, \beta} \sum_{i=1}^N [-g_m(x_i) - \beta h(x_i;a)]^2$$​

继续用上面线性搜索的方法求解$\rho_m$：

$$ \rho_m = \arg \min_\rho \sum_{i=1}^N L(y_i, F_{m-1}(x_i) + \rho h(x_i; a_m))$$

最后更新$F_m(x)$：

$$ F_m(x) = F_{m-1}(x) + \rho_m h(x; a_m) $$​

这里的关键点就是通过梯度近似将问题简化。

### 算法总结

Gradient Boosting的整体算法如下：
$$
F_0(x) = \arg \min_\rho \sum_{i=1}^N L(y_i, \rho) \\
For\ m = 1\ to\ M\ do: \\
\tilde{y_i} = -[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}]_{F(x)=F_{m-1}(x)}, i = 1,N \\
a_m = \arg \min_{a, \beta} \sum_{i=1}^N [\tilde{y_i} - \beta h(x_i;a)]^2 \\
\rho_m = \arg \min_\rho \sum_{i=1}^N L(y_i, F_{m-1}(x_i) + \rho h(x_i; a_m)) \\
F_m(x) = F_{m-1}(x) + \rho_m h(x; a_m) \\
endFor
$$


##  应用

下面尝试将算法应用到不同的损失函数和模型上。

### 最小二乘回归

当损失函数为最小二乘时，此时的梯度$\tilde{y_i}$即残差，由于损失函数与梯度近似时的损失函数一致，此时的$\rho_m = \beta_m$，算法整体会非常简化：
$$
F_0(x) = \bar{y} \\
For\ m = 1\ to\ M\ do: \\
\tilde{y_i} = y_i - F_{m-1}(x_i), i = 1,N \\
(\rho_m, a_m) = \arg \min_{a, \rho} \sum_{i=1}^N [\tilde{y_i} - \rho h(x_i;a)]^2 \\
F_m(x) = F_{m-1}(x) + \rho_m h(x; a_m) \\
endFor
$$

### 回归树

当基函数是一棵树的时候，问题会变得更有意思。此时的基函数可以被定义为：

$$ h(x; \{b_j, R_j\}^J_1) = \sum_{j=1}^J b_j 1(x \in R_j) $$

在这种情况下，更新公式变成：

$$ F_m(x) = F_{m-1}(x) + \rho_m \sum_{j=1}^J b_{jm} 1(x \in R_{jm})$$​

这里$\{R_{jm}\}_1^J$​是基函数树在拟合$\{\tilde{y_i}\}_1^N$时构造的，$b_{jm}$是对应的最小二乘系数：

$$ b_jm = ave_{x_i \in R_{jm}} \tilde{y_i}$$

令$\gamma_{jm} = \rho_m b_{jm}$，进一步得到：

$$F_m(x) = F_{m-1}(x) + \sum_{j=1}^J \gamma_{jm} 1(x \in R_{jm})$$

通过将$\rho_m$內移，可以更精细的进行最优化求解。

$$ \{\gamma_{jm}\}_1^J = \arg \min_{\{\gamma_j\}_1^J} \sum_{i=1}^N L(y_i, F_{m-1}(x_i) + \sum_{j=1}^J \gamma_j 1(x \in R_{jm})) $$

由于$R_{jm}$相互独立，上述求解可以独立进行：

$$ \gamma_{jm} = \arg \min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i)+\gamma)$$

当损失函数为LAD回归时，此时可以得到：

$$ \gamma_{jm} = median_{x_i \in R_{jm}} \{y_i - F_{m-1}(x_i)\}$$

最后的算法会非常稳定，因为残差取值范围只有两个，$\tilde{y_i} \in \{-1,1\}$​
$$
F_0(x) = median\{y_i\}^N_1 \\
For\ m = 1\ to\ M\ do: \\
\tilde{y_i} = sign(y_i - F_{m-1}(x_i)), i = 1,N \\
\{R_{jm}\}_1^J = J-terminal\ node\ tree(\{\tilde{y_i}, x_i\}_1^N) \\
\gamma_{jm} = median_{x_i \in R_{jm}} \{y_i - F_{m-1}(x_i)\}, j = 1, J \\
F_m(x) = F_{m-1}(x) + \sum_{j=1}^J \gamma_{jm} 1(x \in R_{jm}) \\
endFor
$$

###  二分类

在二分类问题中，损失函数被定义为：

$$ L(y, F) = \log (1 + \exp(-2yF)), y \in \{-1,1\} $$

其中$F(x) = \frac{1}{2} \log\frac{Pr(y=1|x)}{Pr(y=-1|x)}$​​此时可以推导出：

$$ \tilde{y_i} = -[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}]_{F(x)=F_{m-1}(x)} = 2y_i/(1+\exp(2y_iF_{m-1}(x_i)))$$

如果以回归树作为基函数，同样可以得到：

$$ \gamma_{jm} = \arg \min_\gamma \sum_{x_i \in R_{jm}} \log (1 + \exp(-2y_i(F_{m-1}(x_i) + \gamma)) $$​

使用牛顿法进行近似求解：
$$
y'_i = - \tilde{y_i} = - 2y_i/(1+\exp(2y_iF_{m-1}(x_i))) \\
y''_i = [\frac{\partial L(y_i, F(x_i))}{\partial^2 F(x_i)}]_{F(x)=F_{m-1}(x)} = -\frac{4\exp(2y_iF_{m-1}(x_i))}{(1+\exp(2y_iF_{m-1}(x_i)))^2} = -\frac{2}{1+\exp(2y_iF_{m-1}(x_i))} * (2 - \frac{2}{1+\exp(2y_iF_{m-1}(x_i))}) = |\tilde{y_i}|(2 - |\tilde{y_i}|) \\
\gamma_{jm} = -\frac{\sum_{x_i \in R_{jm}}y'_i}{\sum_{x_i \in R_{jm}}y''_i} = \frac{\sum_{x_i \in R_{jm}}\tilde{y_i}}{\sum_{x_i \in R_{jm}}|\tilde{y_i}|(2 - |\tilde{y_i}|)}
$$
因此二分类的整体算法如下：
$$
F_0(x) = \frac{1}{2} \log \frac{1+\bar y}{1 - \bar y} \\
For\ m = 1\ to\ M\ do: \\
\tilde{y_i} = 2y_i/(1 + \exp(2y_iF_{m-1}(x_i))), i = 1,N \\
\{R_{jm}\}_1^J = J-terminal\ node\ tree(\{\tilde{y_i}, x_i\}_1^N) \\
\gamma_{jm} = \frac{\sum_{x_i \in R_{jm}}\tilde{y_i}}{\sum_{x_i \in R_{jm}}|\tilde{y_i}|(2 - |\tilde{y_i}|)} \\
F_m(x) = F_{m-1}(x) + \sum_{j=1}^J \gamma_{jm} 1(x \in R_{jm}) \\
endFor
$$
最后得到二分类的概率：
$$
p_+(x) = \hat{Pr}(y = 1 | x)  = 1/(1 + e^{-2F_M(x)}) \\
p_-(x) = \hat{Pr}(y = -1 | x)  = 1/(1 + e^{2F_M(x)})
$$

#### 剪枝

注意到对于损失函数而言，可以拆成两部分：

$$ \phi(\rho, a) = \sum_{i=1}^N \log(1+\exp(-2y_iF_{m-1})) \exp(-2y_i\rho h(x_i;a))) $$

当$y_iF_{m-1}$​过大时，此项整体值接近于0，可以忽略计算。从函数空间角度看，这意味着那些梯度下降的厉害的影响更大，可以通过二阶导衡量，即我们上面已经计算得到的：

$$ \omega_i = |\tilde y_i|(2 - |\tilde y_i|)  $$

通过对所有$\omega_i$​​​升序排序，删除前$l(\alpha)$​​个，可以有效减少计算量，下降幅度能达到10到20倍。下面时$l(\alpha)$​​​的计算公式，其中一般情况下$\alpha \in [0.05,0.2]$​。

$$ \sum_{i=1}^{l(\alpha)} \omega_{(i)} = \alpha \sum_{i=1}^N \omega_i $$

## 正则化

为了防止过拟合，需要一些正则化手段控制。最直观的就是参数M。迭代次数越多，越容易过拟合。

另外也可以使用缩减技术。在这种方式下，$F_m(x)$的更新方式变成如下：

$$ F_m(x) = F_{m-1}(x) + v\rho_mh(x;a_m) ,  0 < v \leq 1$$

正则化参数的选择需要通过模型选择方法进行调优。

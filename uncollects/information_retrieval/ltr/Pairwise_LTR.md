# 常见pairwise的LTR算法

pairwise的LTR算法三兄弟，包括RankNet, LambdaRank和LambdaMART。其思想一脉相承，逐步演进。

## RankNet

pairwise的排序基本思想就是将排序问题转换为同一个query下doc之间两两比较好坏的分类问题。其数学定义如下：

$$ P_{ij} \equiv P(U_i \rhd U_j) \equiv \frac{1}{1+e^{-\sigma(s_i - s_j)}}$$

其中 $s_i = f(x_i)$ 和 $s_j = f(x_j)$表示函数$f(x)$将特征空间$x$映射成某个数值。

可以看到pairwise排序的定义和[逻辑回归](../machine_learning/逻辑回归.md)的定义非常接近，都用了一个sigmoid函数将结果映射到(0, 1]上。其损失函数也类似，使用了一个交叉熵损失函数，其中$\bar{P_{ij}}$表示已标注结果。

$$C = -\bar{P_{ij}}\log P_{ij} - (1 - \bar{P_{ij}}) \log (1 - P_{ij})$$

令$\bar P_{ij} = \frac{1}{2}(1 + S_{ij})$，其中$S_{ij} \in \{0, \pm 1\}$，上面的损失函数可以转换为：

$$
C = -\frac{1}{2}(1+S_{ij}) \log \frac{1}{1+e^{-\sigma(s_i - s_j)}} - (1 - \frac{1}{2}(1+S_{ij})) \log (1 - \frac{1}{1+e^{-\sigma(s_i - s_j)}}) = \frac{1}{2}(1 - S_{ij}) \sigma(s_i - s_j) + \log (1+e^{-\sigma(s_i - s_j)})
$$
这个损失函数有几个特点。首先这是一个对称的结果，当$S_{ij} = 1$​​时，$C = \log(1+e^{-\sigma(s_i - s_j)}) $​​，当$S_{ij} = -1$​​时，$C = \log(1+e^{-\sigma(s_j - s_i)}) $​​。​​其次，当$s_i = s_j $时，仍然有$C = \log2$​。最后，当得分错误时，损失接近线性，而得分正确时，损失接近0。​

梯度可以计算得到：

$$
\frac{\partial C}{\partial s_i} = \sigma(\frac{1}{2}(1 - S_{ij}) - \frac{1}{1+e^{\sigma(s_i - s_j)}}) = - \frac{\partial C}{\partial s_j} 
$$

通过随机梯度下降可以得到：

$$
w_k \to w_k - \eta(\frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k})
$$
根据上面的梯度结果，可以进一步做如下简化：

$$
\frac{\partial C}{\partial w_k} = \frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} +\frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k} = \sigma(\frac{1}{2}(1 - S_{ij}) - \frac{1}{1+e^{\sigma(s_i - s_j)}})(\frac{\partial s_i}{\partial w_k} - \frac{\partial s_j}{\partial w_k}) = \lambda_{ij}(\frac{\partial s_i}{\partial w_k} - \frac{\partial s_j}{\partial w_k})
$$

在这里对$\lambda_{ij}$做了如下定义：

$$
\lambda_{ij} \equiv \frac{\partial C(s_i - s_j)}{\partial s_i} = \sigma(\frac{1}{2}(1 - S_{ij}) - \frac{1}{1+e^{\sigma(s_i - s_j)}})
$$

我们定义$I$为文档对的集合$\{i,j\}$，其中文档$U_i$应该与文档$U_j$区分开。为避免重复，$I$只包含$U_i \rhd U_j$的文档，也就是$S_{ij} = 1$，这样，$w_k$的更新公式变成：

$$ \delta_{w_k} = -\eta \sum_{\{i,j\} \in I} (\lambda_{ij}\frac{\partial s_i}{\partial w_k} - \lambda_{ij}\frac{\partial s_j}{\partial w_k}) \equiv -\eta \sum_i \lambda_i \frac{\partial s_i}{\partial w_k}  $$

其中$\lambda_i = \sum_{j:\{i,j\} \in I} \lambda_{ij} - \sum_{j:\{j,i\} \in I} \lambda_{ij}$​

这种通过累积$\lambda$之后再做迭代的方式大大的提升了计算效率。

同时$\lambda_{ij}$的公式也简化为：

$$
\lambda_{ij} = \frac{\sigma}{1+e^{\sigma(s_i - s_j)}}
$$

## LambdaRank

LambdaRank是在RankNet的基础上改进的。RankNet奠定了pairwise优化方式的计算基础。但RankNet的损失函数是交叉熵损失函数，其优化目标是减少误分类率。然而在排序领域有其他更合理的评价方式，比如NDCG，ERR等，这些评价方式一般更关注TOP结果。然而这些评价函数都是不可导的，无法直接作为损失函数优化。

$\lambda$可以被视作推力，衡量了文档位置交换的幅度。LambdaRank在此基础上加入了文档位置交换的评价指标影响，这里以NDCG为例，对于交换之后NDCG变化更大的文档对应该给更高的值。因此这里定义如下，其中$|\Delta_{NDCG}|$是两文档位置互换的NDCG差：

$$ \lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta_{NDCG}| $$

此时优化问题变为优化最大值，更新公式变为：

$$ w_k \to w_k + \eta \frac{\partial C}{\partial w_k} $$

LambdaRank的思想核心是把NDCG差加入$\lambda$​中，在梯度上考虑了优化目标变化，从而绕过了优化目标不可导的问题。

## LambdaMART

LambdaMART是LambdaRank结合MART之后的算法。

根据定义，其中Z代表NDCG等搜索排序上的评价方式：

 $$\lambda_{ij} = \frac{-\sigma |\Delta Z_{ij}|}{1 + e^{\sigma(s_i - s_j)}} $$

可以得到下式，其中$\rho_{ij} \equiv \frac{1}{1 + e^{\sigma(s_i - s_j)}} = \frac{-\lambda_{ij}}{\sigma |\Delta Z_{ij}|}$：

$$ \frac{\partial C}{\partial s_i} = \sum_{\{i,j\}\leftrightarrows I}\frac{-\sigma |\Delta Z_{ij}|}{1 + e^{\sigma(s_i - s_j)}} \equiv \sum_{\{i,j\}\leftrightarrows I}-\sigma |\Delta Z_{ij}| \rho_{ij} $$

那么可以得到：

$$ \frac{\partial^2 C}{\partial s_i^2} = \sum_{\{i,j\}\in I} \sigma^2|\Delta Z_{ij}| \rho_{ij}(1-\rho_{ij})$$

根据MART构造方式，可以通过牛顿法得到，注意此时是求最大值，符号取正：

$$ \gamma_{km} = \frac{\sum_{x_i\in R_{km}\frac{\partial C}{\partial s_i}}}{\sum_{x_i \in R_{km}} \frac{\partial^2 C}{\partial s_i^2}} = \frac{-\sum_{x_i\in R_{km}}\sum_{\{i,j\}\leftrightarrows I}|\Delta Z_{ij}| \rho_{ij}}{\sum_{x_i\in R_{km}}\sum_{\{i,j\}\in I} |\Delta Z_{ij}|\sigma \rho_{ij}(1-\rho_{ij})}$$

最后，LambdaMART的整体算法如下所示：

树的棵树$n$，训练样本$m$，每棵树的叶子数$L$，学习率$\eta$
$$
\begin{align}
& for\ i = 0\ to\ m\ do \\
& \ \ F_0(x_i) = BaseModel(x_i)\ //如果基础模型为空，令\ F_0(x_i) = 0 \\
& end\ for \\
& for\ k=1\ to\ N\ do \\
& \ \ for\ i=0\ to\ m\ do \\
& \ \ \ \ y_i = \lambda_i // \lambda实际上就是函数空间上的梯度\\
& \ \ \ \ w_i = \frac{\partial y_i}{\partial F_{k-1}(x_i)} \\
& \ \ end\ for \\
& \ \ \{R_{lk}\}^L_{l=1}\ // 在\{x_i,y_i\}^m_{i=1} 创建L个叶子节点\\
& \ \ \gamma_{lk} = \frac{\sum_{x_i \in R_{lk}}y_i}{\sum_{x_i \in R_{lk}}w_i}\ //牛顿法赋值 \\
& \ \ F_k(x_i) = F_{k-1}(x_i) + \eta\sum_l\gamma_{lk}1(x_i \in R_{lk})\ //每一步学习率为 \eta \\
& end\ for
\end{align}
$$

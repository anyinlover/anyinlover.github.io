# CatBoost YetiRank 学习

YetiRank和LambdaRank等类似，都属于Pairwise算法。区别在于其基础树模型用了一种对称树结构，另外考虑了标签混淆矩阵的作用。

## YetiRank与LambdaRank的不同系数

CatBoost可以看做是从RankNet基础上改进得到的一种算法。对于RankNet而言，对于一对doc而言，其损失函数可以表示为：

$$ C = -\bar{P_{ij}}\log P_{ij} - (1 - \bar{P_{ij}}) \log (1 - P_{ij})$$

对于全量文档，只考虑$\bar{P_{ij}} =1$ ，令$\sigma=1$，全量损失函数可以表示为：

$$L = \sum_{(i,j)}-\log P_{ij} = -\sum_{(i, j)} \log \frac{e^{s_i}}{e^{s_i} + e^{s^j}}$$

根据梯度计算可以得到：

$$\frac{\partial C}{\partial s_i} = - \frac{e^{s_i}}{e^{s_i} + e^{s^j}} = - \frac{\partial C}{\partial s_j}$$

对于LambdaRank而言，在此基础上乘了一个$|\Delta_{NDCG}|$，而对于YetiRank而言，此系数使用了其他计算方法。即有：

$$\frac{\partial C}{\partial s_i} = - w_{ij}\frac{e^{s_i}}{e^{s_i} + e^{s^j}} $$

其$w_{ij}$计算有两部分组成，分别表示文档对能排序到top的权重和混淆矩阵：

$$ w_{ij} = N_{ij} c(l_i, l_j)$$

$N_{ij}$的计算出于以下的直觉考虑，对于排序问题而言，只有排序在top的结果影响最大，因此有下面的模拟实验，进行100次这样的实验，对$s_i$进行转换，其中$r_i$满足$[0, 1)$的均匀分布：

$$ \hat s_i = s_i + \log \frac{r_i}{1-r_i} $$

每一次对按得分重排后的相邻doc对累积得分$1/R$，这里$R$代表排序更高的那个doc位置。最后得到的$N_{ij}$为100次随机实验累积的结果。因此可以说，这里的$N_{ij}$思想与NDCG等评价指标类似。

$c(l_i, l_j)$的主要思想是将标注者错标的概率考虑进来，比如原来这个文档只能得1分，但被误标成2分。后续会有详细介绍。这里有如下定义：

$$ c(l_i, l_j) = \sum_u \sum_v 1_{u>v} p(u|l_i) p(v | l_j)$$

## YetiRank的优化迭代计算

$$dL = \sum_{(i,j)} w_{ij}((\delta s_i - \delta s_j) \frac{e^{s_i}}{e^{s_i} + e^{s^j}}) $$ 

固定$|\sqrt{w_{ij}}(\delta s_i - \delta s_j)|$，此优化问题可以转变为：

$$ \arg min_{\delta s}  = \sum_{(i,j)} w_{ij}((\delta s_i - \delta s_j) + \frac{e^{s_i}}{e^{s_i} + e^{s^j}})^2 $$

对于基函数为对称树的YetiRank而言，此优化问题变成了：

$$ \arg min_{\delta s}  = \sum_{(i,j)} w_{ij}((c_{leaf(d_i)} - c_{leaf(d_i)}+ \frac{e^{s_i}}{e^{s_i} + e^{s^j}})^2 $$

这个问题通过举证表示即求$(Ac-b)^Tw(Ac-b)$的最小值，有解析解$c = (A^TwA)^{-1}A^Twb$。

可以看到这种计算方式有比较大的计算量，因为是两两计算的，这也是catboost包里面的Yetirankpairwise方式，令一种yetirank方式则和LambdaMart比较接近。
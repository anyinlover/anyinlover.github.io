# 决策树

> 分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点(node)和有向边(directed edge )组成。结点有两种类型：内部结点(internal node)和叶结点(leaf node)。内部结点表示一个特征或属性，叶结点表示一个类。

决策树是一类非线性的机器学习方法，既可用于分类，也可用于回归。有两种角度可以来理解决策树。最直观的就是决策树是一组if-then规则的组合，不停的递归分类直到分类完成。另一个角度是根据条件概率分布对特征空间分割成系列长方形。

决策树学习算法一般包含特征选择、决策树生成和决策树剪枝的过程。决策树生成考虑局部最优，剪枝考虑全局最优。

决策树常见的算法有ID3，C4.5，CART。

## ID3系列

### ID3

ID3算法以熵为损失函数，每次选择信息增益最大的特征作为分类结点。下面是相关术语的定义。

设$X$是一个取有限个值的离散随机变量，概率分布为：

$$ P(X = x_i) = p_i,  i = 1,2,...,n $$

随机变量$X$的熵$H(X)$定义为：

$$ H(X) = - \sum_{i=1}^n p_{i} \log p_{i} $$

随机变量的不确定性越大，信息熵就越大。

条件熵$H(Y|X)$表示在已知随机变量$X$条件下随机变量$Y$的不确定性，定义为$X$给定条件下$Y$的条件概率分布的熵对$X$的数学期望：

$$ H(Y|X) = \sum_{i=1}^n p_i H(Y|X = x_i)$$

当上述概率从数据估计得到上，两者分别又称为经验熵和经验条件熵，信息增益就是后两者之差，表示得知特征$X$的信息而使得类$Y$的信息的不确定性减少的程度：

$$ g(D,A) = H(D) - H(D|A) $$

ID3算法没有剪枝，下面是python实现。

```python
def entropy(datasets):
    category_freq = datasets.iloc[:, -1].value_counts() / len(datasets)
    return -sum(category_freq * np.log2(category_freq))

def condition_entropy(datasets, pos):
    name = datasets.columns[pos]
    return sum(datasets.groupby(name).apply(entropy) * datasets.groupby(name).size() / len(loans))

def info_gain(datasets, pos):
    return entropy(datasets) - condition_entropy(datasets, pos)

def max_info_gain(datasets):
    info_gains = [(info_gain(datasets, pos), datasets.columns[pos]) for pos in range(len(datasets.columns)-1)]
    return max(info_gains)

def nh(datasets):
    return datasets.shape[0] * entropy(datasets)

class Node:
    def __init__(self, label=None, child={}, nh=0, leaf_num=1, leaf_label=None):
        self.label = label
        self.child = child
        self.nh = nh
        self.leaf_num = leaf_num
        self.leaf_label = leaf_label
        
    
    def __repr__(self):
        return self.label

    def __str__(self):
        if self.child:
            return json.dumps({self.label: {key: json.loads(str(value)) for key, value in self.child.items()}}, ensure_ascii=False)
        else:
            return json.dumps(self.label, ensure_ascii=False)

class DTree:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.tree = Node()

def id3_train(datasets, epsilon=0):
    if len(datasets.iloc[:,-1].value_counts()) == 1:
        return Node(label = datasets.iloc[0, -1], nh = nh(datasets))

    if datasets.shape[1] == 1:
        return Node(label = datasets.iloc[:, -1].mode()[0], nh = nh(datasets))

    max_ig = max_info_gain(datasets)

    if max_ig[0] < epsilon:
        return Node(label = datasets.iloc[:, -1].mode()[0], nh = nh(datasets))
    else:
        current = Node(label = max_ig[1], nh = nh(datasets))
        current.child = datasets.groupby(max_ig[1]).apply(id3_train).to_dict()
        current.leaf_label = datasets.iloc[:, -1].mode()[0]
        return current
```

### C4.5

C4.5算法与ID3类似， 只是选择特征的标准从信息增益改为了信息增益比。用于纠正ID3存在偏向选择取值较多的特征问题。

特征$A$对训练数据集$D$的信息增益比$g_R(D,A)$定义为其信息增益$g(D,A)$与训练数据集$D$关于特征$A$的值的熵$H_A(D)$之比：

$$ g_R(D,A) = \frac{g(D,A)}{H_A(D)}$$

下面是python实现：

```python
def info_gain_ratio(datasets, pos):
    return info_gain(datasets, pos) / entropy(datasets.iloc[:, pos:])

def max_info_gain_ratio(datasets):
    info_gain_ratios = [(info_gain_ratio(datasets, pos), datasets.columns[pos]) for pos in range(len(datasets.columns)-1)]
    return max(info_gain_ratios)

def c4_5_train(datasets, epsilon=0):
    if len(datasets.iloc[:,-1].value_counts()) == 1:
        return Node(label = datasets.iloc[0, -1])

    if len(datasets.columns) == 1:
        return Node(label = datasets.iloc[:, -1].sort_values(ascending=False).index[0])

    max_igr = max_info_gain_ratio(datasets)

    if max_igr[0] < epsilon:
        return Node(label = datasets.iloc[:, -1].sort_values(ascending=False).index[0])
    else:
        current = Node(label = max_igr[1])
        child = datasets.groupby(max_igr[1]).apply(id3_train)
        current.child = child.to_dict()
        return current
```

### 树的剪枝

决策树的剪枝通过极小化决策树整体的代价函数来实现，代价函数定义为：

$$ C_\alpha(T) = \sum_{t=1}^{|t|}N_tH_t(T) + \alpha|T| $$

这意味着树的复杂度和拟合度之间的平衡。$\alpha$越小，树越复杂，否则相反。当$\alpha=0$时，结果就是一个满树$T_0$。

树的剪枝分为预剪枝和后剪枝，预剪枝是在树的生成过程中就比较划分前后的代价函数决定是否继续划分，这种算法计算成本小，但问题是容易过度剪枝，忽略了后续节点存在显著划分的可能性。更推荐的方法是下面描述的后剪枝。

树的后剪枝是从叶子节点自下而上进行，比较叶结点剪除后整体决策树的$C_\alpha(T)$，如果剪除后代价函数变小了，就将叶节点剪除，如此反复，直到到达根节点为止。注意代价函数差的计算可以在局部进行。

下面是ID3的后剪枝算法python实现。

```python
def prune(node):
    if node.child:
        nh_sum = 0
        leaf_sum = 0
        for child in node.child.values():
            child = prune(child)
            nh_sum += child.nh
            leaf_sum += child.leaf_num
        if node.nh <= nh_sum + alpha * (leaf_sum - 1):
            node.label = node.leaf_label
            node.child = {}
        else:
            node.nh = nh_sum
            node.leaf_num = leaf_sum
    
    return node
```

## CART

CART是另一类决策树算法，构建在二叉树结构上。这也意味着每个特征只能一分为二。因此与ID3算法不同的是，在特征选择的时候，不仅要计算最优的特征，还要在特征内部计算最优的切分点。

对于回归树使用平方误差最小化准则，对于分类树用基尼指数最小化准则进行特征选择。

### 回归树

现在我们来讨论如何构造一棵回归树。对于 $N$ 次观察，我们的数据由 $p$ 个输入和一个输出构成。即$(x_i,y_i), \text{ for } i=1,2,\cdots,N, \text{ with } x_i=(x_{i1},x_{x2},\cdots,x_{ip})$。算法需要自动确定划分的特征和划分点，以及树的模型。首先假设我们划分成 M 个区域$R_1,R_2,\cdots,R_M$，将每个区域映射成一个常量$c_m$：

$$f(x) = \sum_{m=1}^M c_m I(x \in R_m)$$

如果我们采用最小二乘法$\sum(y_i - f(x_i))^2$，很容易得到最好的$\hat{c}_m$就是区域$R_m$中$y_i$的平均值：

$$\hat{c}_m = ave(y_i \mid x_i \in R_m)$$

我们使用贪婪算法来找出最优的划分特征 $j$ 和划分点 $s$，定义如下一对划分面：

$$R_1(j,s)=\{X \mid X_j \leq s\} \text{ and } R_2(j,s)=\{X \mid X_j > s\}$$

通过求解下式来找到划分特征 $j$ 和划分点 $s$：

$$min_{j,s}[\min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2]$$

公式看着吓人，求解方法其实就是遍历法，遍历所有输入变量，找到最优的那个划分点。

后续的迭代生成和ID3一样，划分直到满足停止条件。常见的停止条件有：

- 最小叶节点大小
- 最大树深度
- 最大叶节点个数

下面是最小二乘回归树生成算法的python实现：

```python
def least_square(datasets):
    return np.var(datasets.iloc[:, -1]) * datasets.shape[0]
  
def best_split_point(datasets, pos):
    least_square_sum = np.inf
    for sp in datasets.iloc[:,pos].unique():
        square_sum = least_square(datasets[datasets.iloc[:,pos] <= sp]) + least_square(datasets[datasets.iloc[:,pos] > sp])
        if square_sum < least_square_sum:
            least_square_sum = square_sum
            least_sp = sp
    return least_square_sum, least_sp

def best_split_index(datasets):
    least_square_sums = [(*best_split_point(datasets, pos), datasets.columns[pos]) for pos in range(datasets.shape[1]-1)]
    return min(least_square_sums)

class Node:
    def __init__(self, label=None, sp=None, child={}, nh=0, nh_sum=0, alpha=0, leaf_num=1, leaf_label=None):
        self.label = label
        self.child = child
        self.sp = sp
        self.nh = nh
        self.nh_sum = nh_sum
        self.alpha = alpha
        self.leaf_num = leaf_num
        self.leaf_label = leaf_label
    
    def __repr__(self):
        return self.label

    def __str__(self):
        if self.child:
            return json.dumps({self.label: {key: json.loads(str(value)) for key, value in self.child.items()}}, ensure_ascii=False)
        else:
            return json.dumps(self.label, ensure_ascii=False)

def cart_regress_train(datasets):
    if len(datasets.iloc[:,-1].value_counts()) == 1:
        return Node(label = datasets.iloc[0, -1])
    
    if len(datasets.columns) == 1:
        return Node(label = datasets.iloc[:, -1].mean())

    least_square_sum, least_sp, least_index = best_split_index(datasets)
    leaf_least_square = least_square(datasets)
    if leaf_least_square - least_square_sum < epsilon:
        return Node(label = datasets.iloc[:, -1].mean())
    else:
        current = Node(label=least_index, sp=least_sp)
        less_data = datasets[datasets[least_index] <= least_sp].drop(least_index, axis=1)
        more_data = datasets[datasets[least_index] > least_sp].drop(least_index, axis=1)
        current.child = {"less": cart_regress_train(less_data), "more": cart_regress_train(more_data)}
        return current
```

### 分类树

对于CART分类树而言，则是使用基尼指数选择。

假设有$K$个类，样本点属于第$k$类的概率为$p_k$，基尼系数定义为：

$$Gini(p) = \sum_{k=1}^K p_{k} (1 - p_{k}) = 1 - \sum_{k=1}^K p_k^2$$

基尼指数和ID3的熵值除以2接近。分类树寻找最优划分特征和划分点的方法与回归树完全一致。下面的python实现：

```python
def gini(datasets):
    category_freq = datasets.iloc[:, -1].value_counts() / len(datasets)
    return 1 - sum(np.square(category_freq))

def condition_gini(datasets, pos, sp):
    name = datasets.columns[pos]
    r_data = datasets[datasets[name] == sp]
    w_data = datasets[datasets[name] != sp]
    return (gini(r_data) * r_data.shape[0] + gini(w_data) * w_data.shape[0]) / datasets.shape[0]

def gini_split_point(datasets, pos):
    name = datasets.columns[pos]
    condition_ginis = [(condition_gini(datasets, pos, sp), sp) for sp in datasets[name].unique()]
    return min(condition_ginis)

def gini_split_index(datasets):
    min_ginis = [(*gini_split_point(datasets, pos), datasets.columns[pos]) for pos in range(datasets.shape[1]-1)]
    return min(min_ginis)

def cart_classifier_train(datasets):
    if datasets.shape[1] <= min_s:
        return Node(label = datasets.iloc[:, -1].mode()[0], nh=nh(datasets))

    if len(datasets.iloc[:,-1].value_counts()) == 1:
        return Node(label = datasets.iloc[0, -1], nh=nh(datasets))

    if len(datasets.columns) == 1:
        return Node(label = datasets.iloc[:, -1].mode()[0], nh=nh(datasets))

    min_gini, least_sp, least_index = gini_split_index(datasets)

    if min_gini < epsilon:
        return Node(label = datasets.iloc[:, -1].mode()[0])
    else:
        current = Node(label = least_index, sp = least_sp, nh=nh(datasets))
        y_data = datasets[datasets[least_index] == least_sp].drop(least_index, axis=1)
        n_data = datasets[datasets[least_index] != least_sp].drop(least_index, axis=1)
        current.child = {"y": cart_classifier_train(y_data), "n": cart_classifier_train(n_data)}
        return current
```

### CART的剪枝

首先定义树的代价函数，和ID3系列类似，这里的损失函数定义为：

$$ C_\alpha(T) = C(T) + \alpha|T| $$

其中$C(T)$为子树对训练数据的预测误差，比如基尼指数或平方误差。

对于任意内部节点$t$而言，都存在一个$\alpha$的临界值决定是否剪枝：

$$ \alpha = \frac{C(t) - C(T_t)}{|T_t|-1}$$

对于任一个$\alpha$都有一个特定的最小子树$T_\alpha$使得$C_\alpha(T)$最小。我们使用最弱连接剪枝算法：逐步增加$\alpha$，每次选择最小的$\alpha$节点，直至只留下根节点。可以证明我们要的最小子树必然在这一系列树中。

在生成的一系列子树序列$T_0, T_1,...,T_n$中通过独立验证集交叉验证，选择基尼指数或者平方误差最小的那个子树，即得到最优决策树。

很遗憾，这部分我还木有写代码~~

## 决策树的优缺点

优点：

- 容易理解
- 数据量要求少
- 学习时间短是$O(nlogn)$
- 可处理类别数据
- 异常数据抗干扰性好

缺点：

- 不稳定性，一个小的改变可能造成一个非常不同的切分
- 容易过拟合
- 缺失平滑性

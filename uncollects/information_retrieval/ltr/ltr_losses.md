# LTR损失函数总结

排序学习中很重要的一点就是损失函数的设计。对此，当前见到的综述的最好的是[tensorflow-ranking](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/python/losses.py)，在这里做一下总结。

排序学习按学习模式的不同，损失函数也可以分为三类。Pointwise，Pairwise和Listwise。

## Pointwise 损失函数

Pointwise是最简单的一类，将每个q-d对作为一个学习样本，计算完成后可独立计算损失，也是pytorch这种机器学习库原生支持最好的一类损失函数。

### MultiClass Cross Entropy 损失函数

深度语义排序场景下，最常见的就是把排序问题规约为多分类问题，一般默认就是使用 MultiClass Cross Entropy损失函数。

$$ L({y}, {s}) = - \sum_i y_i \log(\frac{\exp(s_i)}{\sum_j \exp(s_j)}) $$

在pytorch中即CrossEntropyLoss。

实际的训练数据大部分是二分类的，也可以应用在多级标签上。

### Sigmoid Cross Entropy 损失函数

这是把排序问题规约为二分类问题，与MultiClass Cross Entropy不同，它只会输出单个logit，通过sigmod函数转换为概率再计算损失。

$$ L({y}, {s}) = -\sum_i y_i \log(sigmoid(s_i)) + (1-y_i) log(1 - sigmoid(s_i)) $$

在pytorch中即BCEWithLogitsLoss。

实际使用较少，只能应用在二分类标签上。

### Mean Square 损失函数

这是把排序问题规约为回归问题，最经典的L2损失函数。

$$ L({y}, {s}) = \sum_i(y_i - s_i)^2 $$

Cossock, D., Zhang, T.: Subset ranking using regression. In: Proceedings of the 19th Annual Conference on Learning Theory (COLT 2006), pp. 605–619 (2006)

在pytorchvs即MSELoss。

实际使用较少。

## Pairwise 损失函数

pairwise是LTR中主流的学习方法，将问题转换为分类问题，通过比较进行学习，能够更好的捕捉到排序信息。当然pytorch也没有原生支持。

### Pairwise Hinge 损失函数

Hinge损失函数是计算量最小的一个损失函数。

$$ L({y}, {s}) = \sum_i \sum_j I[y_i > y_j] \max(0, 1-(s_i - s_j)) $$

### Pairwise Logistc 损失函数

Pairwise Logistic 损失函数是pairwise中最经典的损失函数。

$$ L({y}, {s}) = \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j))) $$

OpenAI训练Reward模型用的也是这个损失函数。

### Pairwise Soft Zero One 损失函数

Soft Zero One 损失函数利用了sigmoid函数来衡量损失：

$$ L({y}, {s}) = \sum_i \sum_j I[y_i > y_j] \log(1 - sigmoid(s_i - s_j)) $$

### Pairwise MSE 损失函数

在 Pairwise下应用 MSE损失函数。

$$ L({y}, {s}) = \sum_{i \neq j} ((s_i - s_j) - (y_i - y_j))^2 $$

## Listwise 损失函数

### Circle 损失函数

### Softmax 损失函数

### Poly One Softmax 损失函数

### Unique Softmax 损失函数

### Mixture EM 损失函数

### ListMLE 损失函数

### ApproxNDCG 损失函数

### ApproxMRR 损失函数

### Neural Sort Cross Entropy 损失函数

### PiRank NDCG 损失函数

### Coupled RankDistil 损失函数


# Prompting Engineer 总结

## Prompting作为一种新的范式

如何设计一个好的提示？

以下五点是设计提示时应该充分考虑的。

1. 预训练模型的选择
2. 提示模板工程
3. 提示答案工程
4. 多提示学习
5. 基于提示的训练策略

## Pre-trained LM choice

## Prompt Template engineering

两类变种：填空型和前缀型。

人工设计特征模板存在着两个问题：

1. 耗时耗力
2. 非最优解

因此，发展出自动设计特征模板的方法，试图来解决这些问题。

这里方法的分类又有两个维度：

1. 提示是离散的（文本）还是连续的（向量）
2. 提示函数是静态的还是动态的

### Discrete Prompts

离散提示更常被称为硬提示。

1. 提示挖掘：通过大的语料库对训练样本中的输入x和输出y进行挖掘，找到频繁的中间词作为模板。
2. 提示意译：
3. 梯度搜索：
4. 提示生成：
5. 提示打分：对一组提示通过LM打分来判断哪个提示最合适


## Prompt Answer Engineering

## Multi-Prompt Learning

## Prompt-based Training Strategies


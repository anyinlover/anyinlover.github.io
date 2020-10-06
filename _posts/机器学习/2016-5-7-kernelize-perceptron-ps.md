---
layout: single
title: "感知机核函数化习题"
subtitle: "斯坦福大学机器学习习题集二之二"
date: 2016-5-7
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - Ng机器学习系列
---

## 高维系数向量

使用高维映射后，更新$$\theta$$的方法如下：

$$ \theta := \theta + \alpha(y^{(i)} - h_\theta(\phi(x^{(i)})))\phi(x^{(i)})$$

初始化$$\theta^{(0)} = \overrightarrow{0}$$，$$\theta$$可被看做是$$\phi(x^{(i)})$$的线性组合，即$$\exists \beta_l, \theta^{(i)} = \sum_{l=1}^i \beta_l \phi(x^{(l)})$$，因此$$\theta^{(i)}$$可以用线性组合的系数$$\beta_l$$表示。初始的$$\theta^{(0)}$$即是系数$$\beta_l$$的空列表。

## 预测新输入

$$g({\theta^{(i)}}^T \phi(x^{(i+1)})) = g(\sum_{l=1}^i \beta_l \cdot \phi(x^{(l)})^T\phi(x^{i+1})) = g(\sum_{l=1}^i \beta_l K(x^{(l)}, x^{(i+1)}))$$

因此只需要在每次迭代时计算$$\beta_i = \alpha(y^{(i)} - g({\theta^{(i-1)}}^T \phi(x^{(i)})))$$。而$${\theta^{(i-1)}}^T \phi(x^{(i)})$$同样可以用上面的方法更新。

## 更新新训练集

因为这里是感知机，因此除非样本$$\phi(x^{(i)})$$错误分类，$$y^{(i)} - h_\theta(\phi(x^{(i)}))$$一般是0，否则就是$$\pm 1, y,h \in \{ 0,1\}$$。或者是$$\pm 2, y,h \in \{-1,1\}$$。因此可以用$$\sum_{\{i:y^{(i)} \neq h_{\theta^{(i)}}(\phi(x^{(i)}))\}} \alpha(2y^{(i)}-1)\phi(x^{(i)})$$可以表示向量$$\theta$$，即$$\theta^{(i)}=\sum_{i \in Misclassified} \beta_i \phi(x^{(i)})$$，即只有分类错误的样本才会增添系数。对于新加的样本同样如此。

---
layout: single
title: "Getting to Know Your Data"
subtitle: "摘自《数据挖掘概念与技术》第二章"
date: 2015-12-05 16:04
author: "Anyinlover"
mathjax: true
category: 机器学习
tags:
  - 机器学习
  - 数据挖掘
---

## Data Objects and Attribute Types

- Nominal Attributes
- Binary Attributes
- Ordinal Attributes
- Numeric Attributes
  - Interval-Scaled Attributes
  - Ratio-Scaled Attributes

## Basic Statistical Descriptions of Data

### Measuring the Central Tendency

#### Mean

$$
\bar{x} = \frac{\displaystyle\sum_{i=1}^{N} x_i}{N} =\frac{x_1+x_2+\dots+x_N}{N}
$$

#### Median

$$
median = L_1 + \left(\frac{N/2-(\sum freq)_l}{freq_{median}}\right)width
$$

$$L_1$$ is the lower boundary of the median interval.

$$N$$ is the number of values in the entire data set.

$$(\sum freq)_l$$ is the sum of the frequencies of all of the intervals that lower than the median interval.

$$freq_{median}$$ is the frequency of the median interval.
$$width$$ is the width of the median interval.

#### Mode

$$
mean - mode \approx 3 \times (mean - median)
$$

Applied to unimodal numeric data that are moderately skewed (asymmetrical).

#### Midrange

$$
midrange = \frac{\max()+\min()}{2}
$$

### Measuring the Dispersion of Data

#### Range, Quartiles, and Interquartile Range

##### Range

The difference between the largest and smallest values.

##### Quantiles

The points taken at regular intervals of a data distribution.

##### quartiles

The 4-quantiles

##### Interquartile range

The distance between the first and third quartiles.

$$
IQR = Q_3 - Q_1
$$

#### Five-Number Summary, Boxplots, and Outliers

##### Five-Number Summary

$$
Minimum, Q_1, Median, Q_3, Maximum
$$

##### Boxplots

A way of visulizing a distribution using Five-Number Summary.

##### Outliers

The whiskers are extended to $$1.5 \times IQR$$ of the quartiles. Others are Outliers and plotted individually.

#### Variance and Standard Deviation

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})^2 =
\left(\frac{1}{N}\sum_{i=1}^{N} x_i^2\right) - \bar{x}^2
$$

- $$\sigma$$ meassures spread about the mean and should be considered onlu when the mean is chosen as the measure of center.
- Using Chebyshev's inequality show that at least $$(1-\frac{1}{k^2}) \times 100\% $$ of the obesrvations are no more than _k_ standard deviations from the mean.

### Graphic Display of Basic Statistical Description of Data

- Quantile plot
- Quantile-Quantile plot
- Histograms
- Scatter plot

## Data Visualization

- Pixel-Oriented Visualization Techniques
- Geometric Projection Visualization Techniques
- Icon-Based Visualization Techniques
- Hierarchical Visualization Techniques
- Tag cloud and tree-map

## Measuring Data Similarity and Dissimilarity

### Data Matrix versus Dissimilarity Matrix

Data Matrix

$$
\begin{bmatrix}
  x_{11} & \dots & x_{1f} & \dots & x_{1p} \\
  \dots  & \dots & \dots  & \dots & \dots \\
  x_{i1} & \dots & x_{if} & \dots & x_{ip} \\
  \dots  & \dots & \dots  & \dots & \dots \\
  x_{n1} & \dots & x_{nf} & \dots & x_{np}
\end{bmatrix}
$$

Dissimilarity Matrix

$$
\begin{bmatrix}
  0 \\
  d(2,1) & 0 \\
  d(3,1) & d(3,2) & 0 \\
  \vdots & \vdots & \vdots \\
  d(n,1) & d(n,2) & \dots & \dots & 0
\end{bmatrix}
$$

$$
sim(i,j) = 1 - d(i,j)
$$

### Proximity Measures for Nominal Attributes

$$
d(i,j) = \frac{p-m}{p}
$$

_m_ is the number of matches
_p_ is the total number of attributes

### Proximity Measures for Binary Attributes

- q: (i,j) = (1,1)
- r: (i,j) = (1,0)
- s: (i,j) = (0,1)
- t: (i,j) = (0,0)

Symmetric Binary Dissimilarity

$$
d(i,j) = \frac{r+s}{q+r+s+t}
$$

Asymmetric Binary Dissimilarity

$$
d(i,j) = \frac{r+s}{q+r+s}
$$

### Dissimilarity of Numeric Data

#### Euclidean distance

$$
d(i,j) = \sqrt{(x_{i1} - x_{j1})^2 + (x_{i2}-x_{j2})^2
+ \dots + (x_{ip}-x_{jp})^2}
$$

Weighted Euclidean distance

$$
d(i,j) = \sqrt{\omega_1(x_{i1} - x_{j1})^2 + \omega_2(x_{i2}-x_{j2})^2
+ \dots + \omega_m(x_{ip}-x_{jp})^2}
$$

#### Manhattan distance

$$
d(i,j) = |x_{i1} - x_{j1}| + |x_{i2} - x_{j2}| + \dots + |x_{ip} - x_{jp}|
$$

- Non-negativity: $$ d(i,j) \geq 0 $$
- Identity of indiscernibles: $$ d(i,i) = 0 $$
- Symmetry: d(i,j) = d(j,i)
- Triangle inequality: $$d(i,j) \leq d(i,k) + d(k,j)$$

#### Minkowski distance

$$
d(i,j) = \sqrt[h]{|x_{i1} - x_{j1}|^h + |x_{i2}-x_{j2}|^h
+ \dots + |x_{ip}-x_{jp}|^h}
$$

#### Supremum distance

($$L_{max}$$, $$L_{\infty}$$ norm, Chebyshev distance)

$$
d(i,j)  = \lim_{h \to \infty} \left( \sum_{f=1}^{p} |x_{if} - x_{jf}|^h \right)^{\frac{1}{h}} = \max_{f}^{p}|x_{if} - x_{jf}|
$$

### Proximity Measures for Ordinal Attributes

$$
z_{if} = \frac{r_{if} - 1}{M_f - 1}
$$

$$
r_{if} \in \{1, \dots, M_f\}
$$

### Dissimilarity for Attributes of Mixed Types

$$
d(i,j) = \frac{\sum_{f=1}^{p} \delta_{ij}^{(f)} d_{ij}^{(f)}}
{\sum_{f=1}^{p} \delta_{ij}^{(f)}}
$$

- _f_ is numeric:

$$
d_{ij}^{(f)} = \frac{|x_{if}-x_{jf}|}{\max_{h} x_{hf} - \min_{h} x_{hf}}
$$

- _f_ is nominal or binary: $$d_{ij}^{(f)} = 0$$ if $$x_{if} = x_{jf}$$; otherwise, $$d_{ij}^{(f)} = 1$$
- _f_ is ordinal: $$
z_{if} = \frac{r_{if} - 1}{M_f - 1}$$

### Cosine Similarity

$$
sim(x,y) = \frac{x^t \cdot y}{\|x\|\|y\|}
$$

A simple variation frequently used in information retrieval and biology taxnomy.

$$
sim(x,y) = \frac{x \cdot y}{x \cdot x + y \cdot y - x \cdot y}
$$

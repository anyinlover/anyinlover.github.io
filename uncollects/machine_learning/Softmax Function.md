# Softmax Function

The **softmax function** is a mathematical function that converts a vector of raw scores (also called logits) into probabilities. It is often used in machine learning, especially in classification tasks, where we want to interpret the output of a model as probabilities for each class.

The softmax function maps a vector $z$ of real numbers to a probability distribution. Given an input vector $z = [z_1, z_2, ..., z_n]$, the softmax function outputs a vector $\sigma(z) = [\sigma(z_1), \sigma(z_2), ..., \sigma(z_n)]$, where each component is computed as:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

![softmax](../../assets/image/softmax.png)

Key Properties:

- **Output Range:** The output of the softmax function is always between 0 and 1 for each element.
- **Sum of Probabilities:** The sum of the output probabilities equals 1, i.e., $\sum_{i=1}^{n} \sigma(z_i) = 1$.
- **Effect of Large Inputs:** If any element of the input vector is significantly larger than the others, the corresponding probability will be close to 1, while the others will approach 0.

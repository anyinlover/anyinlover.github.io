# Sigmoid Function

A **sigmoid function** is a type of mathematical function that produces an S-shaped curve. It's commonly used in machine learning and statistics, particularly in logistic regression and neural networks as an activation function. The function maps any real-valued input to a value between 0 and 1, which makes it useful for representing probabilities or binary outcomes.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

![sigmoid](../../assets/image/sigmoid.png)

Properties of the Sigmoid Function:

- **Range:** The output is always between 0 and 1.
- **Shape:** The function has a characteristic "S" shape (sigmoid curve).
- **Asymptotes:** As $x$ approaches $\infty$, the output approaches 1. As $x$ approaches $-\infty$, the output approaches 0.
- **Symmetry:** The sigmoid function is symmetric about $x = 0$.

It can also be used as one of [[Activation Functions]].

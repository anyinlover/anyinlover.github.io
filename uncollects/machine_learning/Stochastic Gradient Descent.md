# Stochastic Gradient Descent

The essence of gradient descent is to iteratively update the parameters by taking small steps in the steepest direction (gradient) until reaching the minimum value. It is important to note that gradient descent can only find local optima.

Starting with an initial guess for $\theta$ (which is often set to 0), the following iteration is performed, where $\alpha$ is a fixed value known as the learning rate:

$$
\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
$$

The previous equation holds for a single training example. In practice, we have multiple training examples. There are two variations for handling multiple examples:

1. Batch gradient descent: In each iteration, we compute the error for all training examples and update $\theta$ accordingly
2. Stochastic gradient descent: In each iteration, we compute the error and update $\theta$ for a single training example

Stochastic gradient descent is typically faster since it updates the parameters after a single instance, while batch gradient descent requires computing errors for all training examples, which can be computationally expensive. However, stochastic gradient descent may not reach the global minimum and instead hovers around it. Therefore, for large datasets, stochastic gradient descent is preferred, while batch gradient descent is suitable for smaller datasets.

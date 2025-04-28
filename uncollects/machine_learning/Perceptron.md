# Perceptron

The perceptron is a long-standing model that was initially used to model the principles of neurons. Later, it was applied in machine learning and, despite its simplicity, proved to be very effective. It still serves as a cornerstone for neural network algorithms.

The main difference between the perceptron and logistic regression lies in the replacement of the logistic function with the step function:

$$
g(z)=
\begin{cases}
1 \quad\text{if }z\geq0 \\
0 \quad\text{if }z<0
\end{cases}
$$

The iteration process still employs gradient descent as mentioned earlier.

At first glance, the perceptron and logistic regression seem very similar, but in reality, there are significant differences. The output of the perceptron model is still discrete, whereas logistic regression produces continuous outputs. Specifically, the perceptron cannot be interpreted using probabilities and its decisions are black and white, which means we cannot use the maximum likelihood function to measure its fit.

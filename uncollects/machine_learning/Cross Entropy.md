# Cross Entropy

Cross entropy is a measure of the difference between two probability distributions.

$$
\mathbb{H}_{\text{ce}}(p, q) \triangleq -\sum_{y} p(y) \log q(y)
$$

In the context of classification, cross entropy is used as a loss function to quantify how well the predicted probability distribution aligns with the true distribution (the actual labels).

In multi-class classification, the cross-entropy loss can be generalized as:

$$
L = - \sum_{i} y_i \log(p_i)
$$

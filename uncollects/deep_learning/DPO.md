# DPO

**DPO** stands for **Direct Preference Optimization**. It's a technique used in the field of machine learning, particularly in training language models, to improve their alignment with human preferences. The main idea behind DPO is to directly optimize models to prefer outputs that are more aligned with human judgments, without relying on reward models (which are used in reinforcement learning-based approaches like **RLHF**, or Reinforcement Learning from Human Feedback).

The **objective function** in **Direct Preference Optimization (DPO)** is designed to optimize a language model by increasing the likelihood of the preferred output (as judged by human evaluators) relative to non-preferred outputs. The objective function can be viewed as a form of binary [[Classification]] — where the model is trained to prefer one output over the other. A common way to define this is by using the **log-likelihood ratio** between the preferred and non-preferred outputs.

In a simple form, for a pair $(y_1, y_2)$, the objective can be written as:

$$
L(\theta) = \log \frac{P(y_1 \mid x; \theta)}{P(y_2 \mid x; \theta)}
$$

Where:
- $P(y_1 \mid x; \theta)$ is the probability of generating the preferred output $y_1$ for input $x$ (model output given input).
- $P(y_2 \mid x; \theta)$ is the probability of generating the non-preferred output $y_2$ for the same input.
- $\theta$ represents the model parameters that we're optimizing.

In practice, this is implemented as a binary [[Cross Entropy]] loss function that compares the probabilities of the two outputs. This loss is typically averaged over a batch of preference pairs.

$$
\mathcal{L} = - \mathbb{E}_{x, (y_1, y_2)} \left[ \log \sigma(P(y_1 \mid x)) - \log \sigma(P(y_2 \mid x)) \right]
$$

Where:
- $\sigma(\cdot)$ is the [[Sigmoid Function]], ensuring that the outputs are between 0 and 1 (indicating probabilities).
- $x$ is the input, and $(y_1, y_2)$ are the preference pairs.

# Internal Covariate Shift

Think of a deep neural network as a series of interconnected layers. Each layer takes the output of the previous layer as its input.
During training, the network's parameters (weights and biases) are constantly being updated. This is how the network learns to map inputs to desired outputs.
As the parameters of a layer change, the distribution of its outputs (the activations) also changes. This means that the input distribution for the subsequent layers is also changing.
This change in input distribution for each layer during training is called "internal covariate shift."

Why is Internal Covariate Shift a Problem?

Each layer in the network is trying to learn a mapping from its inputs to its outputs. If the input distribution keeps changing, the layer has to constantly readjust its learning.
This constant readjustment slows down the training process. It's like trying to hit a moving target – it's much harder than hitting a stationary one.
Internal covariate shift can also lead to instability in training. The network might oscillate or get stuck in suboptimal solutions.

1. [Gemini](https://gemini.google.com)

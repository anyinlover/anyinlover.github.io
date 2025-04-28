# LayerNorm

LayNorm is short for Layer Normalization. It is a technique used in deep learning to normalize the activations of neurons within a layer for each training case. This helps to stabilize the learning process and can improve the performance of neural networks.

1. Stabilizing activations: reduce [[Internal Covariate Shift]] problem.
2. Improving gradient flow: reduce [[Vanishing Gradient]] problem.

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma}
$$

It is preferred over [[BatchNorm]] because of Sequence length variability: In natural language processing tasks, input sequences can have varying lengths.

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        return self.weight * normalized_x + self.bias
```

1. [Gemini](https://gemini.google.com)

# RMSNorm

RMSNorm (Root Mean Square Normalization) is a variant of normalization used in neural networks, particularly in transformer models like LLaMA.

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2}}
$$

In practice, RMSNorm is often preferred to [[LayerNorm]] in models like LLaMA because it reduces computational overhead.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized_x = x / rms
        return self.weight * normalized_x
```

1. [ChatGPT](https://chatgpt.com)

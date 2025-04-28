# Dropout

Dropout is a regularization technique used in deep learning to prevent overfitting.

Randomly selected neurons are "dropped out" or ignored during each training iteration.

1. Reduces interdependence between neurons.
2. Simulates training multiple networks.
3. Reduces overfitting.

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float() / (1 - self.p)
            return x * mask
        return x
```

1. [Gemini](https://gemini.google.com)

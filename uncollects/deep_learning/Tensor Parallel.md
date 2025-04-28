# Tensor Parallel

![mlp-tp](../../assets/image/mlp_tp.png)
![attention-tp](../../assets/image/attention_tp.png)

```python
class f(torch.autograd.Function):
    def forward(ctx, x):
        return x
    def backward(ctx, gradient):
        all_reduce(gradient)
        return gradient

class g(torch.autograd.Function):
    def forward(ctx, x):
        all_reduce(x)
        return x
    def backward(ctx, gradient):
        return gradient
```

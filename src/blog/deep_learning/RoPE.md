---
title: Rotary Positional Embedding
pubDate: 2025-04-10 08:56:00
tags:
  - embedding
  - dl
---

# RoPE

## Introduction to RoPE

Rope (Rotary Positional Embedding) is a type of relative positional encoding. Currently, most mainstream large models use Rope or its variants. The original paper on Rope can be found in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).

Since self-attention computation is position-independent, positional encoding has been added since the invention of the transformer to capture dependencies between different positions. The transformer uses absolute positional encoding.

$$
\begin{aligned}
p_{k, 2t} &= \sin(k / 10000^{2t / d}) \\
p_{k, 2t+1} &= \cos(k / 10000^{2t / d})
\end{aligned}
$$

Because absolute positional encoding is directly added to the token embedding, it cannot directly model the relative positions between tokens. The inference performance on sequences exceeding the training data length drops sharply. Relative positional encoding is used to correct this problem, and Rope has become the mainstream approach.

The core idea of Rope is to find a positional encoding function such that the following equation holds:

$$
\langle f(\mathbf{q}_m, m), f(\mathbf{k}_n, n) \rangle = g(\mathbf{q}_m, \mathbf{k}_n, m - n)
$$

That is, when calculating the dot product of $q$ and $k$ during attention, the result is independent of the absolute positions $m$ and $n$ of the tokens in $q$ and $k$, and only depends on the relative position $m - n$.

When the embedding dimension $d$ is only 2, the following formulas precisely satisfy the above property:

$$
\begin{aligned}
f(\mathbf{q}_m, m) &= (\mathbf{q}_m)e^{im\theta} \\
f(\mathbf{k}_n, n) &= (\mathbf{k}_n)e^{in\theta} \\
g(\mathbf{q}_m, \mathbf{k}_n, m - n) &= \text{Re}[(\mathbf{q}_m)(\mathbf{k}_n)^* e^{i(m-n)\theta}]
\end{aligned}
$$

The proof of the above equation involves the application of complex exponential functions, mainly relying on the following three properties:

$$
\begin{aligned}
\langle z_1, z_2 \rangle &= \text{Re}[z_1 z_2^*] \\
(z_1z_2)^* &= z_1^* z_2^* \\
(e^{i\phi})^* &= e^{-i\phi}
\end{aligned}
$$

Using the above three formulas, the Rope formula above can be easily derived.

According to Euler's formula:

$$
e^{i\phi} = \cos(\phi) + i \sin(\phi)
$$

Expanding this, we can obtain $f$, which is consistent for both $q$ and $k$:

$$
f(\boldsymbol{q}_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix}
$$

Actually, we can understand the above Rope formula from another more intuitive perspective. The geometric meaning of the dot product of two-dimensional vectors is the product of their lengths multiplied by the cosine of the angle between them. The above Rope positional encoding function is equivalent to rotating the vector while keeping its length unchanged. Therefore, calculating the dot product of two rotated vectors only involves the relative rotation angle and is independent of the absolute angles.

Once the $d=2$ scenario is understood, it becomes relatively easy to understand the scenario where $d$ is any even number. The embedding dimension is divided into pairs, and different $\theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, ..., d/2]$ are applied according to the pair number, resulting in the complete Rope positional encoding function.

$$
\mathbf{R}_{\boldsymbol{\Theta}, m}^{d} = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
$$

The core concept of RoPE is to rotate the embedding vectors based on the token's position. This is achieved by applying a **rotation matrix** to the token's embedding, where the rotation angle is determined by the token's position in the sequence. By rotating the embeddings instead of using fixed position encodings, the model can maintain more flexible and continuous position information.

## Introduction to YaRN

Although Rope apply relative position embedding, it is still limit in generalizing past the context windows seen during training. Several extension methods were proposed. YaRN is the most popular one between 
performance and complexity. The original paper can be found in [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071).


$$
\theta_i^{new} = \left[ \gamma_i + (1 - \gamma_i) \frac{L}{L'} \right] \theta_i, \quad \gamma_i = \begin{cases} 1, & r_i > \tau \\ 0, & r_i < 1 \\ \frac{r_i - 1}{\tau - 1}, & \text{others} \end{cases}
$$

The key point is that when $\theta_i$ is small enough, it rotate slow, the max rotated angle is $r_i = \frac{\theta_i L}{2\pi}$. if $r_i < 1$, it didn't go through the whole cycle during training, so we should interpolate the $\theta_i$. If $r_i > \tau$, it can safely extrapolate. A linear translation is applied between the two conditions.

YaRN also add a scale weight to softmax, which is:

$$
\lambda = \left( 1 + 0.1 \ln \frac{L'}{L} \right)^2
$$

It is just an experience value without theory support.

## Open Source Implementation

### Transformer-Engine

```python
class FusedRoPEFunc(torch.autograd.Function):
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        interleaved: bool = False,
        cu_seqlens: Union[torch.Tensor, None] = None,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
```

### Flash-Attention

```python
class ApplyRotaryEmb(torch.autograd.Function):
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):

    def backward(ctx, do):
```
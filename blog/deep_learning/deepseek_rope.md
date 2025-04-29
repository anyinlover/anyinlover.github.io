---
title: Unveiling the RoPE Implementations of Llama and DeepSeek
date: 2025-04-29 14:20:00
tags:
  - embedding
  - dl
---

Almost all open-source models use RoPE (Rotary Position Embedding) based on the same theory from [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). However, there are two ways to implement RoPE: the GPT-J style and the GPT-NeoX style.

## GPT-J and GPT-NeoX RoPE Implementations

The GPT-J style is identical to the original RoFormer, using an interleaved method to calculate RoPE. The GPT-NeoX style uses an alternative, non-interleaved method. According to the [Eleuther AI blog](https://blog.eleuther.ai/rotary-embeddings/), they considered the original implementation inefficient and thus improved it by splitting the dimension into two halves (non-interleaved). Note that the GPT-NeoX and GPT-J styles produce different results.

The GPT-NeoX style RoPE calculation is as follows:

```python
import torch

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[:, None, None, :]
        sin = emb.sin()[:, None, None, :]
        return cos, sin

def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, cos, sin):
    return t * cos + _rotate_half(t) * sin
```

The GPT-J style has two ways to implement RoPE, with the complex number method being more intuitive.

Complex number method:

```python
import torch

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis[:, None, None, :]

def apply_rotary_pos_emb(t, freqs_cis):
    t_ = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
    rotated_t_complex = t_ * freqs_cis
    return torch.view_as_real(rotated_t_complex).flatten(3)
```

Traditional method:

```python
import torch

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        emb = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=1)
        cos = emb.cos()[:, None, None, :]
        sin = emb.sin()[:, None, None, :]
        return cos, sin

def _rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.flatten(start_dim=3)

def apply_rotary_pos_emb(t, cos, sin):
    return t * cos + _rotate_every_two(t) * sin

```

## Llama and DeepSeek RoPE Implementations

Due to the significant influence of the Hugging Face community, many people believe Llama uses the GPT-NeoX style based on its [inference code in the transformers library](https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/modeling_llama.py#L173-L188). However, this is not the case. In [Llama's original code](https://github.com/meta-llama/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py#L64-L74), it implements the GPT-J style RoPE using the complex number method. So, why the difference between the two codebases? The answer lies in [this issue](https://github.com/huggingface/transformers/issues/25199). In the [weight conversion script](https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/convert_llama_weights_to_hf.py#L113-L115), they permuted the weights of `q_proj` and `k_proj`.

```python
def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

```

It's not immediately obvious why this works. We will come back to explain everything later.

A similar situation occurred with DeepSeek-V3. In the [original code for deepseek-v3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L375-390), it uses the same complex number method as Llama to compute GPT-J style RoPE (in fact, their code is very similar). Again, in the [Hugging Face code for deepseek-v3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L339-371), it uses a style similar to GPT-NeoX, just like Llama (again, the code is very similar), but with an exception on lines 364 and 367. Yes, it's very similar to the `permute` function mentioned above.

```python
q = q.view(b, s, h, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
k = k.view(b, s, h, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
```

So, what is the difference between these two implementations? Many people have the same question, as seen in this [discussion](https://huggingface.co/deepseek-ai/DeepSeek-V3/discussions/83).

Since RoPE only acts on the last dimension, a simple example helps understand why.

In the GPT-NeoX style, if $dim=6$, we have:

$$
\begin{bmatrix}
d_0 \\
d_1 \\
d_2 \\
d_3 \\
d_4 \\
d_5
\end{bmatrix}
\cdot
\begin{bmatrix}
c_0 \\
c_1 \\
c_2 \\
c_0 \\
c_1 \\
c_2
\end{bmatrix}
+
\begin{bmatrix}
-d_3 \\
-d_4 \\
-d_5 \\
d_0 \\
d_1 \\
d_2
\end{bmatrix}
\cdot
\begin{bmatrix}
s_0 \\
s_1 \\
s_2 \\
s_0 \\
s_1 \\
s_2 \\
\end{bmatrix}
=
\begin{bmatrix}
d_0c_0 - d_3s_0 \\
d_1c_1 - d_4s_1 \\
d_2c_2 - d_5s_2 \\
d_3c_0 + d_0s_0 \\
d_4c_1 + d_1s_1 \\
d_5c_2 + d_2s_2
\end{bmatrix}
$$

In the GPT-J style (interleaved), we have:

$$
\begin{bmatrix}
d_0 \\
d_1 \\
d_2 \\
d_3 \\
d_4 \\
d_5 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
c_0 \\
c_0 \\
c_1 \\
c_1 \\
c_2 \\
c_2 \\
\end{bmatrix}
+
\begin{bmatrix}
-d_1 \\
d_0 \\
-d_3 \\
d_2 \\
-d_5 \\
d_4 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
s_0 \\
s_0 \\
s_1 \\
s_1 \\
s_2 \\
s_2 \\
\end{bmatrix}
=
\begin{bmatrix}
d_0c_0 - d_1s_0 \\
d_1c_0 + d_0s_0 \\
d_2c_1 - d_3s_1 \\
d_3c_1 + d_2s_1 \\
d_4c_2 - d_5s_2 \\
d_5c_2 + d_4s_2 \\
\end{bmatrix}
$$

Now, let's look at the DeepSeek style (as implemented in Hugging Face, applying the permutation from the `transpose` operation before applying the NeoX-style rotation):

First, permute the input vector $d$:
$$
d_{permuted} =
\begin{bmatrix}
d_0 \\
d_2 \\
d_4 \\
d_1 \\
d_3 \\
d_5 \\
\end{bmatrix}
$$

Then apply the NeoX-style rotation logic to $d_{permuted}$:
$$
\begin{bmatrix}
d_0 \\
d_2 \\
d_4 \\
d_1 \\
d_3 \\
d_5 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
c_0 \\
c_1 \\
c_2 \\
c_0 \\
c_1 \\
c_2 \\
\end{bmatrix}
+
\begin{bmatrix}
-d_1 \\
-d_3 \\
-d_5 \\
d_0 \\
d_2 \\
d_4 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
s_0 \\
s_1 \\
s_2 \\
s_0 \\
s_1 \\
s_2 \\
\end{bmatrix}
=
\begin{bmatrix}
d_0c_0 - d_1s_0 \\
d_2c_1 - d_3s_1 \\
d_4c_2 - d_5s_2 \\
d_1c_0 + d_0s_0 \\
d_3c_1 + d_2s_1 \\
d_5c_2 + d_4s_2 \\
\end{bmatrix}
$$

We find that the DeepSeek style (permute then apply NeoX RoPE) simply produces a permuted version of the GPT-J style result. Specifically, the resulting vector is $[r_0, r_2, r_4, r_1, r_3, r_5]$ where $r$ is the result vector from the GPT-J style calculation.

Recall how attention is calculated, using the dot product of `q` and `k`:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

When `q` and `k` are permuted in the *same* way, their dot product remains unchanged. Let $P$ be the permutation matrix. Then $(Pq)^T(Pk) = q^T P^T P k$. If $P$ is a permutation matrix representing the specific swap used, $P^T P = I$ (identity matrix), so $q^T P^T P k = q^T I k = q^T k$. The dot product result is the same.

So the DeepSeek style (in Hugging Face) is actually equivalent to the GPT-J style in terms of the final attention scores.

Now we can return to the Llama code in transformers. Permuting the `qw` and `kw` weights beforehand has the same effect as permuting the resulting `q` and `k` vectors after the matrix multiplication but before applying RoPE.

In the regular approach (no weight permutation), applying the linear layer:

$$
\mathbf{q}^T = \mathbf{x}^T \times \mathbf{W_q} = \mathbf{x}^T \times
\begin{bmatrix}
\mathbf{w_0} &
\mathbf{w_1} &
\mathbf{w_2} &
\mathbf{w_3} &
\mathbf{w_4} &
\mathbf{w_5}
\end{bmatrix}
=
\begin{bmatrix}
d_0 & d_1 & d_2 & d_3 & d_4 & d_5
\end{bmatrix}
$$
where $d_i = \mathbf{x} \cdot \mathbf{w_i}$.

In the permuted weight approach:

$$
\mathbf{q}_{permuted\_weights}^T = \mathbf{x}^T \times \mathbf{W_q}_{permuted} = \mathbf{x}^T \times
\begin{bmatrix}
\mathbf{w_0} &
\mathbf{w_2} &
\mathbf{w_4} &
\mathbf{w_1} &
\mathbf{w_3} &
\mathbf{w_5}
\end{bmatrix}
=
\begin{bmatrix}
d_0 & d_2 & d_4 & d_1 & d_3 & d_5
\end{bmatrix}
$$

Applying the linear layer with permuted weights yields a result vector `q` that is already permuted in the same way as the DeepSeek style's explicit permutation of `q`. Then, the Hugging Face Llama code applies the NeoX-style RoPE to this already-permuted vector, which, as shown above, is equivalent to applying the GPT-J style RoPE to the original, unpermuted `q`.

Now, we understand the complete picture:

Llama used GPT-J style RoPE during training. However, when converting its original weights to the Hugging Face format, it permuted the `qw` and `kw` weights and used a GPT-NeoX-like style for inference (for performance reasons).

DeepSeek also used GPT-J style RoPE during training, but it *forgot* to permute the `qw` and `kw` weights during the weight conversion. Therefore, it needed to add the permutation of `q` and `k` within the transformer's inference code (thus gaining no performance benefit from using the NeoX RoPE calculation itself).

One last question remains: if the GPT-NeoX style RoPE calculation is more performant, why do most open-source models still use the GPT-J style RoPE during training? The answer might be related to long context window extension. I haven't delved deeply into this issue yet.

## Adverse Effects on Other AI Frameworks

Now that we understand the RoPE situation for Llama and DeepSeek, unfortunately, the story is far from over. Many AI frameworks copied the Hugging Face code for DeepSeek, leading to unnecessary complexity in their training code.

For example, in the [Megatron-LM code](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py#L961-962):

```python
if self.multi_latent_attention and self.rotary_interleaved:
    raise ValueError("rotary_interleaved does not work with multi_latent_attention.")
```

It prohibits the use of interleaved RoPE and specifically handles DeepSeek in its [RoPE code](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rope_utils.py#L117-120), mimicking the Hugging Face code:

```python
if multi_latent_attention:
    x1 = t[..., 0::2]
    x2 = t[..., 1::2]
    t = torch.cat((x1, x2), dim=-1)
```

In [PaddlePaddle](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/deepseek_v2/modeling.py#L591-595), it also handles DeepSeek RoPE in the same way:

```python
b, s, h, d = q.shape
q = q.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

b, s, h, d = k.shape
k = k.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])
```

## Lessons Learned

So, what can we learn from this experience?

1.  Trust different sources according to the following priority: Paper > Original implementation code > Hugging Face code > Third-party frameworks.
2.  Solutions from the open-source community might not be the optimal implementation; think before you act (or copy).
3.  Pay special attention to the differences between inference code and training code when converting between them.

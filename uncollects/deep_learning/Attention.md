# Attention

Attention in deep learning is a technique that mimics human cognitive attention, allowing neural networks to focus on the most relevant parts of the input data when making predictions.

![Attention](../../assets/image/2024-12-24-10-16-13.png)

1. Selective Focus: Instead of processing the entire input data uniformly, attention mechanisms enable the model to assign different weights or importance to different parts of the input.
2. Contextual Understanding: Attention helps the model to capture long-range dependencies and contextual relationships within the data.

The most popular attention is scaled dot product attention.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

```python
def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0):
    d_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    output = torch.bmm(attention_weights, value)
    return output
```

```python
def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0):
    d_k = query.size(-1)
    scores = torch.einsum('bqd,bkd->bqk', query, key) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    output = torch.einsum('bqk,bkd->bqd', attention_weights, value)
    return output
```

1. [Gemini](https://gemini.google.com)
2. [d2l.ai](https://d2l.ai/chapter_attention-mechanisms-and-transformers/)

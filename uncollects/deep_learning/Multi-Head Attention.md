# Multi-Head Attention

Multi-head attention is a powerful extension of the attention mechanism that allows a model to attend to information from different representation subspaces at different positions.

![Multi-Head Attention](../../assets/image/Multi-Head%20Attention.png)

Instead of performing a single attention calculation, multi-head attention runs through multiple [[Attention]] mechanisms in parallel.

```python
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output
```

```python
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = rearrange(self.q_proj(query), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(key), 'b m (h d) -> b h m d', h=self.num_heads)
        v = rearrange(self.v_proj(value), 'b m (h d) -> b h m d', h=self.num_heads)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p)
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        attn_output = self.out_proj(attn_output)
        return attn_output
```

1. [Gemini](https://gemini.google.com)
2. [d2l.ai](https://d2l.ai/chapter_attention-mechanisms-and-transformers/)

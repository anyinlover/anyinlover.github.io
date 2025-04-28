# KVCache

In the context of large language models (LLMs), **KV cache** refers to a technique used to speed up the inference process by caching the **key** and **value** tensors generated during the self-attention computation. The key and value tensors are part of the attention mechanism, which allows the model to focus on relevant parts of the input sequence when making predictions.

### How KV Cache Works

- **Self-Attention:** In LLMs, the attention mechanism computes a weighted sum of the values (V) based on the similarity of keys (K) to the query (Q). The process typically requires calculating the dot product between the query and all the keys in the sequence, which can be expensive.

- **During Inference:** When generating tokens in an autoregressive model (like GPT), the model attends to all previous tokens generated so far. Instead of recalculating the keys and values for all the previous tokens during every new token generation, the KV cache stores these keys and values once they are computed.

- **KV Cache Use:** The cached keys and values can be reused for each new token generation, so the model doesn’t have to recompute them each time. This significantly speeds up the inference process, especially when working with long sequences, since the model only needs to compute the query for the current token and use the cached keys and values for the rest.

### Benefits of KV Cache
1. **Speed:** Reduces redundant computations and accelerates token generation.
2. **Memory Efficiency:** The cached values only need to be stored for the sequence generated up to that point, and can be efficiently managed.
3. **Scalability:** Especially useful in autoregressive models, where the number of tokens can grow long and recalculating the attention for every token would be inefficient.

```python
class TransformerWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size, seq_length):
        super(TransformerWithKVCache, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Layer for generating keys and values
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None):
        """
        Forward pass with KV cache.
        :param x: Input tensor of shape [batch_size, sequence_length]
        :param cache: Dictionary to hold the cached keys and values
        :return: output logits and updated cache
        """
        batch_size, seq_len = x.size()

        # Embed the input tokens
        x_emb = self.embedding(x)

        # Initialize cache if not provided
        if cache is None:
            cache = {'key': [], 'value': []}

        # Loop through each transformer layer
        for layer_idx in range(self.n_layers):
            # Project input embeddings to query, key, and value
            q = self.q_proj(x_emb)  # [batch_size, seq_len, d_model]
            k = self.k_proj(x_emb)  # [batch_size, seq_len, d_model]
            v = self.v_proj(x_emb)  # [batch_size, seq_len, d_model]

            # Cache key and value tensors (extend with the current layer's K and V)
            cache['key'].append(k)
            cache['value'].append(v)

            # Perform multi-head self-attention (using cached keys and values)
            for t in range(seq_len):
                # Use the cached K and V tensors from the previous tokens
                cached_k = cache['key'][-1][:, :t + 1, :]  # [batch_size, t+1, d_model]
                cached_v = cache['value'][-1][:, :t + 1, :]  # [batch_size, t+1, d_model]

                # Compute attention
                q_t = q[:, t, :].unsqueeze(1)  # Current token query [batch_size, 1, d_model]
                attn_scores = torch.matmul(q_t, cached_k.transpose(1, 2)) / (self.d_model ** 0.5)  # [batch_size, 1, t+1]
                attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, 1, t+1]

                # Weighted sum of values
                attn_output = torch.matmul(attn_probs, cached_v)  # [batch_size, 1, d_model]
                x_emb[:, t, :] = attn_output.squeeze(1)

            # Pass through transformer layer
            x_emb = self.encoder_layers[layer_idx](x_emb)

        # Final output layer
        logits = self.output_layer(x_emb)
        return logits, cache
```

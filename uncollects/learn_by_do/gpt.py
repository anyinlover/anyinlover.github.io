import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


class GPTBlock(nn.Module):
    def __init__(self, emb_size, n_head, dropout, bias):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GeLU(),
            nn.Linear(4 * emb_size, emb_size),
        )
        self.ln1 = nn.LayerNorm(emb_size, elementwise_affine=bias)
        self.ln2 = nn.LayerNorm(emb_size, elementwise_affine=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_attn = self.ln1(x)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn)
        x = x + self.dropout(attn_out)
        x_ffn = self.ln2(x)
        ff_out = self.ffn(x_ffn)
        x = x + self.dropout(ff_out)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, emb_size, n_layer, n_head, dropout, bias, max_position_embeddings=512):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GPTBlock(emb_size, n_head, dropout, bias) for _ in range(n_layer)])
        self.lm_head = nn.Linear(emb_size, vocab_size, bias=False)
        self.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        token_embeds = self.token_embeddings(x)
        position_embeds = self.position_embeddings(positions)
        x = self.embedding_dropout(token_embeds + position_embeds)

        for layer in self.layers:
            x = layer(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, -1, :])
            loss = None
            return logits, loss


# Example Usage

# Vocabulary size and other hyperparameters
vocab_size = 10000  # Example vocab size
emb_size = 256  # Embedding dimension size
n_layer = 6  # Number of transformer layers
n_head = 8  # Number of attention heads
seq_length = 512  # Sequence length for input

# Instantiate the model
model = GPT(vocab_size, emb_size, n_layer, n_head)

# Example input (batch of 2 sequences)
input_tensor = torch.randint(0, vocab_size, (seq_length, 2))  # (seq_len, batch_size)

# Forward pass
logits = model(input_tensor)
print(logits.shape)  # Should print (seq_len, batch_size, vocab_size)

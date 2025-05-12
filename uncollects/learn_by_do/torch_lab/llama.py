from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1024
    ffn_hidden_dim: int = 8192
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_seq_len: int = 2048

    def __init__(self, **kwargs):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads


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


def apply_rotary_emb(t: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    t_ = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
    rotated_t_complex = t_ * freqs_cis
    return torch.view_as_real(rotated_t_complex).flatten(3)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_hidden_dim: int):
        self.w1 = nn.Linear(dim, ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(ffn_hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xk = torch.repeat_interleave(xk, dim=2, repeats=self.n_rep)
        xv = torch.repeat_interleave(xv, dim=2, repeats=self.n_rep)
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        proj = self.wo(output)
        return proj


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, ffn_hidden_dim=args.ffn_hidden_dim)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(TransformerBlock(params) for _ in range(params.n_layers))
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

"""Models: a bigram baseline and a small GPT-style transformer, written by hand.

Both take token ids (batch, T) and return logits (batch, T, vocab_size):
for every position, one score per possible next token.
Run `python src/model.py` to check shapes and the loss of an untrained model.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 286
    block_size: int = 256   # context window (tokens the model can see)
    d_model: int = 128      # size of each token's vector ("identity card")
    n_layers: int = 4       # how many attention + MLP blocks are stacked
    n_heads: int = 4        # attention heads per layer (128 / 4 = 32 numbers each)
    dropout: float = 0.1    # fraction of values randomly zeroed during training


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy: how surprised the model is by the real next token (lower = better)."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


# --------------------------------------------------------------------------- #
# 1. Bigram baseline: predicts the next token from the CURRENT token only.
# --------------------------------------------------------------------------- #
class BigramModel(nn.Module):
    """A 286 x 286 table: row = current token, columns = scores for the next one.

    No memory, no context. Any real model must beat its loss.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table(idx)  # (B, T) -> (B, T, vocab)


# --------------------------------------------------------------------------- #
# 2. Transformer
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Each token looks at the previous tokens and gathers what is useful to it.

    For every token we compute 3 vectors with Linear layers:
      query (Q) = "what am I looking for?"
      key   (K) = "what do I contain?"
      value (V) = "what I give if someone picks me"
    score(i, j) = Q_i . K_j  -> how much token i should listen to token j.
    "Causal": token i may only look at j <= i (no peeking at the future).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # Q, K, V in one go
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # mix the heads back
        self.dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: 1 where looking is allowed (past and present).
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, hs = self.n_heads, C // self.n_heads
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split into H heads: (B, T, C) -> (B, H, T, hs). Each head attends separately.
        q = q.view(B, T, H, hs).transpose(1, 2)
        k = k.view(B, T, H, hs).transpose(1, 2)
        v = v.view(B, T, H, hs).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(hs)      # (B, H, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                   # each row sums to 1
        weights = self.dropout(weights)
        out = weights @ v                                     # weighted mix of values

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # glue heads back
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Per-token "thinking": widen 128 -> 512, non-linearity, back to 128."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One layer = communicate (attention) then think (MLP).

    `x + ...` is a residual connection: each part adds a correction to the token's
    vector instead of replacing it, which keeps training stable when stacking layers.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # what the token is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)  # where it is in the window
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)         # 128 -> 286 scores

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than the context window"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))      # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))                            # (B, T, vocab)


if __name__ == "__main__":
    from dataset import get_batch, load_split

    torch.manual_seed(0)
    cfg = Config()
    x, y = get_batch(load_split("train"), cfg.block_size, batch_size=8)
    print(f"Batch: x {tuple(x.shape)}, y {tuple(y.shape)}")
    print(f"Pure guessing among {cfg.vocab_size} tokens: loss = ln({cfg.vocab_size}) "
          f"= {math.log(cfg.vocab_size):.2f}")

    for model in (BigramModel(cfg), Transformer(cfg)):
        logits = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{type(model).__name__}: {n_params:,d} parameters")
        print(f"  logits {tuple(logits.shape)}  (one score per possible next token)")
        print(f"  untrained loss: {loss_fn(logits, y).item():.2f}")

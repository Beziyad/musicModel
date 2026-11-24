"""Parameter counter: where do the ~0.9 M numbers of the transformer come from?

Run `python src/param_count.py` to see:
  1. the count per part of the model, measured by PyTorch,
  2. the same numbers recomputed by hand from the formulas (they must match),
  3. how the total changes when the hyperparameters change.
"""
from collections import defaultdict

from model import Config, Transformer


def count_by_part(model: Transformer) -> dict[str, int]:
    """Group parameters by the first part of their name (tok_emb, blocks.0, head...)."""
    parts = defaultdict(int)
    for name, p in model.named_parameters():
        key = ".".join(name.split(".")[:2]) if name.startswith("blocks") else name.split(".")[0]
        parts[key] += p.numel()
    return dict(parts)


def by_hand(cfg: Config) -> dict[str, int]:
    """Same count from the formulas. A Linear(a, b) has a*b weights + b biases."""
    d, V, T = cfg.d_model, cfg.vocab_size, cfg.block_size
    layer_norm = 2 * d                                   # one scale + one shift per number
    attention = (d * 3 * d + 3 * d) + (d * d + d)        # qkv Linear + output projection
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)          # widen 4x, then back
    block = layer_norm + attention + layer_norm + mlp
    return {
        "tok_emb  (vocab x d)": V * d,
        "pos_emb  (block_size x d)": T * d,
        f"blocks   ({cfg.n_layers} x {block:,d})": cfg.n_layers * block,
        "   of which attention": cfg.n_layers * attention,
        "   of which MLP": cfg.n_layers * mlp,
        "   of which LayerNorm": cfg.n_layers * 2 * layer_norm,
        "ln_f": layer_norm,
        "head     (d x vocab + vocab)": d * V + V,
    }


if __name__ == "__main__":
    cfg = Config()
    model = Transformer(cfg)
    total = sum(p.numel() for p in model.parameters())

    print("1. Measured by PyTorch")
    for part, n in count_by_part(model).items():
        print(f"  {part:10s} {n:>9,d}  {100 * n / total:5.1f} %")
    print(f"  {'TOTAL':10s} {total:>9,d}")

    print("\n2. Recomputed by hand")
    hand = by_hand(cfg)
    for part, n in hand.items():
        print(f"  {part:30s} {n:>9,d}")
    hand_total = sum(n for k, n in hand.items() if "of which" not in k)
    print(f"  {'TOTAL':30s} {hand_total:>9,d}  -> {'match' if hand_total == total else 'MISMATCH'}")
    print(f"  File size in float32 (4 bytes each): {total * 4 / 1e6:.1f} MB")

    print("\n3. What if... (only the listed setting changes)")
    variants = [
        ("baseline", {}),
        ("d_model 64", {"d_model": 64}),
        ("d_model 256", {"d_model": 256}),
        ("n_layers 2", {"n_layers": 2}),
        ("n_layers 8", {"n_layers": 8}),
        ("n_heads 8", {"n_heads": 8}),
        ("block_size 512", {"block_size": 512}),
    ]
    train_tokens = 2_950_765
    for label, change in variants:
        n = sum(p.numel() for p in Transformer(Config(**change)).parameters())
        print(f"  {label:15s} {n:>11,d} params  ({train_tokens / n:4.1f} train tokens per param)")

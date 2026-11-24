"""Training loop: show the model millions of "guess the next token" exercises.

Usage:
  python src/train.py --model bigram --epochs 1
  python src/train.py --model transformer --epochs 10

One "epoch" here = as many random windows as there are tokens in train
(2.95 M tokens / (32 x 256) ≈ 360 steps ≈ 7 min for the transformer on this Mac).

After each epoch:
  - measure the loss on train AND validation (same fixed windows every time),
  - save a checkpoint (checkpoints/<model>_epoch<N>.pt) and keep the best one,
  - append a line to checkpoints/<model>_log.csv (for the curves).
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import torch

from dataset import get_batch, load_split
from model import BigramModel, Config, Transformer, loss_fn

CKPT_DIR = Path("checkpoints")


@torch.no_grad()
def evaluate(model, data: torch.Tensor, cfg: Config, n_batches: int, batch_size: int) -> float:
    """Average loss on `n_batches` fixed windows. No learning here (no_grad, eval mode)."""
    model.eval()                                   # dropout off
    gen = torch.Generator().manual_seed(1234)      # same windows at every evaluation
    losses = []
    for _ in range(n_batches):
        starts = torch.randint(len(data) - cfg.block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([data[s : s + cfg.block_size] for s in starts])
        y = torch.stack([data[s + 1 : s + cfg.block_size + 1] for s in starts])
        losses.append(loss_fn(model(x), y).item())
    model.train()                                  # dropout back on
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bigram", "transformer"], default="transformer")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="learning rate (default per model)")
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--no-augment", action="store_true", help="disable transposition")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = Config()
    model = BigramModel(cfg) if args.model == "bigram" else Transformer(cfg)
    lr = args.lr or (1e-2 if args.model == "bigram" else 1e-3)
    # AdamW: the algorithm that applies the corrections computed by backward().
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train, val = load_split("train"), load_split("validation")
    tokens_per_step = args.batch_size * cfg.block_size
    steps_per_epoch = math.ceil(len(train) / tokens_per_step)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{args.model}: {n_params:,d} params | lr {lr} | {steps_per_epoch} steps/epoch "
          f"| augment {'off' if args.no_augment else 'on'}")

    CKPT_DIR.mkdir(exist_ok=True)
    log_path = CKPT_DIR / f"{args.model}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "minutes"])

    def log_eval(epoch: int, step: int, t0: float) -> float:
        tr = evaluate(model, train, cfg, args.eval_batches, args.batch_size)
        va = evaluate(model, val, cfg, args.eval_batches, args.batch_size)
        minutes = (time.time() - t0) / 60
        print(f"epoch {epoch:2d} | train loss {tr:.3f} | val loss {va:.3f} | {minutes:5.1f} min")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, f"{tr:.4f}", f"{va:.4f}", f"{minutes:.2f}"])
        return va

    t0 = time.time()
    best_val = log_eval(0, 0, t0)                  # epoch 0 = before any training
    step = 0
    for epoch in range(1, args.epochs + 1):
        for i in range(steps_per_epoch):
            x, y = get_batch(train, cfg.block_size, args.batch_size, augment=not args.no_augment)
            loss = loss_fn(model(x), y)            # 1. forward: how wrong are we?
            optimizer.zero_grad()
            loss.backward()                        # 2. backward: which way to move each param
            optimizer.step()                       # 3. step: move them a little
            step += 1
            if i % 50 == 0:
                print(f"  step {step:5d} | batch loss {loss.item():.3f}", flush=True)

        val_loss = log_eval(epoch, step, t0)
        ckpt = {"model": model.state_dict(), "config": cfg.__dict__, "kind": args.model,
                "epoch": epoch, "val_loss": val_loss}
        torch.save(ckpt, CKPT_DIR / f"{args.model}_epoch{epoch:02d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, CKPT_DIR / f"{args.model}_best.pt")
            print(f"  new best val loss -> {args.model}_best.pt")

    print(f"Done. Best val loss {best_val:.3f}. Log: {log_path}")


if __name__ == "__main__":
    main()

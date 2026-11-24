"""Dataset and batching: turn the tokenized corpus into (input, target) training windows.

Run `python src/dataset.py` to build the token cache and print one example batch.

Key ideas:
  - Each split (train/validation/test) becomes ONE long list of token ids:
      BOS piece1 EOS BOS piece2 EOS ...
  - A training example is a window of `block_size` tokens (x) and the same window
    shifted by one (y): at every position the model must predict the next token.
  - A batch stacks `batch_size` random windows so they are processed together.
  - Transposition (data augmentation): shift every Pitch token by k semitones.
    Same music, other key -> more distinct examples for free.
"""
import random
from pathlib import Path

import torch

from tokenizer import RAW, build_tokenizer, corpus_rows, encode_file

CACHE = Path("data/processed")

TOKENIZER = build_tokenizer()
BOS = TOKENIZER.vocab["BOS_None"]
EOS = TOKENIZER.vocab["EOS_None"]
ID_TO_TOKEN = {i: tok for tok, i in TOKENIZER.vocab.items()}  # 21 -> "Pitch_37"
_pitch_ids = [i for tok, i in TOKENIZER.vocab.items() if tok.startswith("Pitch_")]
PITCH_MIN, PITCH_MAX = min(_pitch_ids), max(_pitch_ids)  # contiguous range (verified)


def load_split(split: str) -> torch.Tensor:
    """All pieces of one split, concatenated into a single 1-D tensor of token ids.

    Tokenizing takes ~10 s, so the result is cached in data/processed/<split>.pt.
    """
    cache_file = CACHE / f"{split}.pt"
    if cache_file.exists():
        return torch.load(cache_file)
    ids = []
    for r in corpus_rows():
        if r["split"] == split:
            ids += [BOS] + encode_file(TOKENIZER, RAW / r["midi_filename"]) + [EOS]
    data = torch.tensor(ids, dtype=torch.long)
    torch.save(data, cache_file)
    return data


def transpose(x: torch.Tensor, semitones: int) -> torch.Tensor:
    """Shift every Pitch token by `semitones`. Other tokens (time, velocity...) unchanged.

    If a note would fall off the keyboard, the window is returned untouched.
    """
    is_pitch = (x >= PITCH_MIN) & (x <= PITCH_MAX)
    shifted = torch.where(is_pitch, x + semitones, x)
    if ((shifted[is_pitch] < PITCH_MIN) | (shifted[is_pitch] > PITCH_MAX)).any():
        return x
    return shifted


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              augment: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick `batch_size` random windows. Returns x and y, both (batch_size, block_size)."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    xs, ys = [], []
    for s in starts.tolist():
        chunk = data[s : s + block_size + 1]       # block_size + 1 tokens
        if augment:                                 # same shift for x and y
            chunk = transpose(chunk, random.randint(-5, 6))
        xs.append(chunk[:-1])                       # input : tokens 0 .. n-1
        ys.append(chunk[1:])                        # target: tokens 1 .. n
    return torch.stack(xs), torch.stack(ys)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    splits = {s: load_split(s) for s in ("train", "validation", "test")}
    for s, d in splits.items():
        print(f"{s:11s} {len(d):>9,d} tokens")

    block_size, batch_size = 256, 8
    train = splits["train"]
    print(f"\nblock_size={block_size}: {len(train) // block_size:,d} non-overlapping "
          f"windows in train (but any start position is allowed)")

    # 1. The shift by one: x[i] is what the model sees, y[i] what it must guess.
    x, y = get_batch(train, block_size, batch_size)
    print(f"\nBatch shapes: x {tuple(x.shape)}, y {tuple(y.shape)}")
    show = [ID_TO_TOKEN[i] for i in x[0, :6].tolist()], [ID_TO_TOKEN[i] for i in y[0, :6].tolist()]
    print("First window, first 6 positions:")
    for i, (a, b) in enumerate(zip(*show)):
        print(f"  sees ...{a:16s} -> must predict {b}")

    # 2. Transposition: only Pitch tokens move.
    chunk = train[1000:1012]
    up = transpose(chunk, 3)
    print("\nTransposition +3 semitones:")
    for a, b in zip([ID_TO_TOKEN[i] for i in chunk.tolist()], [ID_TO_TOKEN[i] for i in up.tolist()]):
        if a != b:
            print(f"  {a:10s} -> {b}")

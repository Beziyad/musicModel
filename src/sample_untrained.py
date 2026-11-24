"""'Before' sample: what the transformer plays with random (untrained) weights.

Run `python src/sample_untrained.py` -> outputs/before_untrained.mid
Will be replaced by src/generate.py at step 7 (which loads trained weights).
"""
from collections import Counter
from pathlib import Path

import torch
from miditok import TokSequence

from dataset import BOS, ID_TO_TOKEN, TOKENIZER
from model import Config, Transformer

N_TOKENS = 1000  # ~25 s of music at ~41 tokens/s
SEED = 0         # same seed -> same result every run

torch.manual_seed(SEED)
cfg = Config()
model = Transformer(cfg).eval()      # random weights, nothing learned

# Generation loop: look at the last block_size tokens, pick the next one, append, repeat.
idx = torch.tensor([[BOS]])
with torch.no_grad():
    for _ in range(N_TOKENS):
        logits = model(idx[:, -cfg.block_size:])[:, -1]          # scores for the next token
        probs = torch.softmax(logits, dim=-1)                     # scores -> probabilities
        next_id = torch.multinomial(probs, num_samples=1)         # random draw
        idx = torch.cat([idx, next_id], dim=1)

ids = idx[0].tolist()
print("first 20 tokens:", [ID_TO_TOKEN[i] for i in ids[:20]])
print("token types:", Counter(ID_TO_TOKEN[i].split("_")[0] for i in ids).most_common())

out = Path("outputs/before_untrained.mid")
out.parent.mkdir(exist_ok=True)
score = TOKENIZER.decode([TokSequence(ids=ids)])
score.dump_midi(out)
notes = len(score.tracks[0].notes) if score.tracks else 0
print(f"{notes} notes written to {out}")

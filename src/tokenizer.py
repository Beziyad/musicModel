"""MIDI tokenizer (miditok REMI): turn a MIDI file into a list of integers and back.

Run `python src/tokenizer.py` to:
  1. show the vocabulary size,
  2. encode one validation piece and print its first tokens,
  3. decode it back to MIDI (outputs/roundtrip.mid) to check nothing is lost,
  4. count the tokens of the whole Chopin corpus, per split.
"""
import csv
from collections import defaultdict
from pathlib import Path

from miditok import REMI, TokenizerConfig

RAW = Path("data/raw/maestro-v3.0.0")
CORPUS_CSV = Path("data/processed/chopin.csv")


def build_tokenizer() -> REMI:
    """REMI = one token per musical event: Bar, Position, Pitch, Velocity, Duration."""
    config = TokenizerConfig(
        pitch_range=(21, 109),     # the 88 piano keys (A0..C8)
        num_velocities=32,         # loudness 0..127 squeezed into 32 levels
        use_sustain_pedals=True,   # Chopin without pedal sounds dry
        use_time_signatures=False, # MAESTRO is a fake 4/4 everywhere: no information
        use_tempos=False,          # MAESTRO is a fixed 120 bpm everywhere: no information
    )
    return REMI(config)


def encode_file(tokenizer: REMI, midi_path: Path) -> list[int]:
    """MIDI file -> list of token ids. MAESTRO files have a single piano track."""
    return tokenizer.encode(midi_path)[0].ids


def corpus_rows() -> list[dict]:
    with open(CORPUS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    tok = build_tokenizer()
    print(f"Vocabulary size: {tok.vocab_size} distinct tokens")

    # 1. One example, token by token.
    rows = corpus_rows()
    example = next(r for r in rows if r["split"] == "validation")
    path = RAW / example["midi_filename"]
    seq = tok.encode(path)[0]
    print(f"\nExample: {example['title']}  ({float(example['duration_s']):.0f} s)")
    print(f"  {len(seq.ids)} tokens")
    print("  first tokens:", seq.tokens[:20])
    print("  as ids      :", seq.ids[:20])

    # 2. Round trip: tokens -> MIDI, to listen and compare with the original.
    Path("outputs").mkdir(exist_ok=True)
    tok.decode([seq]).dump_midi("outputs/roundtrip.mid")
    print("  decoded back to outputs/roundtrip.mid")

    # 3. Token count for the whole corpus.
    stats = defaultdict(lambda: [0, 0.0])
    for r in rows:
        stats[r["split"]][0] += len(encode_file(tok, RAW / r["midi_filename"]))
        stats[r["split"]][1] += float(r["duration_s"])
    print("\nTokens per split:")
    for split, (n, dur) in stats.items():
        print(f"  {split:11s} {n:>9,d} tokens  ({n / dur:4.1f} tokens/s)")

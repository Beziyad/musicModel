"""Filter the MAESTRO catalogue down to Chopin pieces and print per-split stats.

Input : data/raw/maestro-v3.0.0/maestro-v3.0.0.csv (one row per recording)
Output: data/processed/chopin.csv (only Chopin rows, 4 columns)
"""
import csv
from collections import defaultdict
from pathlib import Path

RAW = Path("data/raw/maestro-v3.0.0")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Read the catalogue and keep Chopin rows whose MIDI file is really on disk.
rows = []
with open(RAW / "maestro-v3.0.0.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if "Chopin" in r["canonical_composer"]:
            if (RAW / r["midi_filename"]).exists():
                rows.append(r)

# 2. Count files and total duration per split (train / validation / test).
stats = defaultdict(lambda: [0, 0.0])
for r in rows:
    stats[r["split"]][0] += 1
    stats[r["split"]][1] += float(r["duration"])

print(f"Chopin files: {len(rows)}")
for split, (n, dur) in stats.items():
    print(f"  {split:11s} {n:4d} files  {dur/3600:5.2f} h")

# 3. Write the reduced catalogue used by the next steps.
with open(OUT / "chopin.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["midi_filename", "split", "title", "duration_s"])
    for r in rows:
        w.writerow([r["midi_filename"], r["split"], r["canonical_title"], r["duration"]])

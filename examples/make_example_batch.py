"""Builds a synthetic delivered batch with a known answer, for trying the QA summary.
Ten raters (r01, r02 are deep experts), one extra rater (r11) who lets a language model write 60% of the ratings,
two item types (B holds most hard items), 10% gold items, 30 canary items, and a 98-item acceptance review."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent / "batch"
OUT.mkdir(exist_ok=True)
N, RATERS = 2000, [f"r{i:02d}" for i in range(1, 12)]
deep = {"r01", "r02"}
typ = np.where(rng.random(N) < 0.25, "B", "A")
hard = rng.random(N) < np.where(typ == "B", 0.65, 0.05)
truth = rng.integers(0, 2, N)
model_ok = np.where(hard, rng.random(N) < 0.30, rng.random(N) < 0.97)
rows = []
for i in range(N):
    for r in rng.choice(RATERS, 3, replace=False):
        if r == "r11" and rng.random() < 0.6:
            ok = model_ok[i]
        else:
            p = (0.85 if r in deep else 0.30) if hard[i] else 0.96
            ok = rng.random() < p
        rows.append((f"i{i:04d}", r, int(truth[i] if ok else 1 - truth[i]), typ[i]))
for c in range(30):                          # canaries: the model is wrong, people are right 90% of the time
    y = int(rng.integers(0, 2))
    for r in RATERS:
        ok = False if (r == "r11" and rng.random() < 0.6) else rng.random() < 0.90
        rows.append((f"c{c:02d}", r, y if ok else 1 - y, "A"))
    pd.DataFrame([(f"c{c:02d}", 1 - y)], columns=["item", "model_label"]).to_csv(
        OUT / "canary.csv", mode="w" if c == 0 else "a", header=c == 0, index=False)
ratings = pd.DataFrame(rows, columns=["item", "rater", "label", "type"])
ratings.to_csv(OUT / "ratings.csv", index=False)
gold_idx = rng.choice(N, 200, replace=False)
pd.DataFrame({"item": [f"i{i:04d}" for i in gold_idx], "label": truth[gold_idx]}).to_csv(OUT / "gold.csv", index=False)
final = ratings[ratings.item.str.startswith("i")].groupby("item")["label"].agg(lambda s: int(s.mean() > 0.5))
review = rng.choice(N, 98, replace=False)
pd.DataFrame({"item": [f"i{i:04d}" for i in review],
              "defect": [int(final[f"i{i:04d}"] != truth[i]) for i in review]}).to_csv(OUT / "review.csv", index=False)
print("written to", OUT)

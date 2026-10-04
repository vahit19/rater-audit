"""MT-Bench human judgments (Zheng et al., 2023; CC-BY-4.0): 3,355 expert pairwise votes and 2,400 GPT-4 votes
on answers from six models to 80 questions. Votes are normalised so that model A is always the alphabetically
first model of the pair; every kind of tie becomes 'tie'. In the GPT-4 split, 'tie (inconsistent)' marks a
verdict that changed when the two answers were shown in the opposite order (FastChat llm_judge/common.py)."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def load(split: str = "human") -> pd.DataFrame:
    d = pd.read_parquet(DATA / f"mtbench_{split}.parquet",
                        columns=["question_id", "model_a", "model_b", "winner", "judge", "turn"])
    swap = d["model_a"] > d["model_b"]
    a = np.where(swap, d["model_b"], d["model_a"])
    b = np.where(swap, d["model_a"], d["model_b"])
    w = d["winner"].where(~d["winner"].str.startswith("tie"), "tie")
    w = np.where(swap & (w == "model_a"), "model_b", np.where(swap & (w == "model_b"), "model_a", w))
    out = pd.DataFrame({"question_id": d["question_id"], "turn": d["turn"], "judge": d["judge"],
                        "A": a, "B": b, "vote": pd.Series(w).str.replace("model_", "", regex=False),
                        "raw_winner": d["winner"]})
    out["unit"] = out["question_id"].astype(str) + "|" + out["A"] + "|" + out["B"] + "|" + out["turn"].astype(str)
    return out


def judge_pairs(votes: pd.DataFrame) -> pd.DataFrame:
    """Every pair of distinct judges who voted on the same unit."""
    rows = []
    for u, d in votes.groupby("unit"):
        v = d.drop_duplicates("judge")[["judge", "vote"]].to_numpy()
        for (j1, v1), (j2, v2) in combinations(v, 2):
            rows.append((u, j1, j2, v1, v2))
    return pd.DataFrame(rows, columns=["unit", "j1", "j2", "v1", "v2"])


def agreement(v1: pd.Series, v2: pd.Series, drop_ties: bool) -> tuple[float, int]:
    keep = ~((v1 == "tie") | (v2 == "tie")) if drop_ties else pd.Series(True, index=v1.index)
    return float((v1[keep] == v2[keep]).mean()), int(keep.sum())


def win_rates(votes: pd.DataFrame) -> pd.Series:
    """Share of points won by each model (win 1, tie 0.5, loss 0) over all votes that involve it."""
    pts = []
    for m_col, other, win in (("A", "B", "a"), ("B", "A", "b")):
        s = np.where(votes["vote"] == win, 1.0, np.where(votes["vote"] == "tie", 0.5, 0.0))
        pts.append(pd.DataFrame({"model": votes[m_col], "p": s}))
    p = pd.concat(pts)
    return p.groupby("model")["p"].mean().sort_values(ascending=False)


def bootstrap_win_rates(votes: pd.DataFrame, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Win rates recomputed on question-level bootstrap resamples (questions are the sampling unit)."""
    rng = np.random.default_rng(seed)
    groups = {q: d for q, d in votes.groupby("question_id")}
    qs = np.array(list(groups))
    rows = []
    for _ in range(reps):
        pick = rng.choice(qs, len(qs), replace=True)
        rows.append(win_rates(pd.concat([groups[q] for q in pick])))
    return pd.DataFrame(rows)


def tiers(boot: pd.DataFrame, order: list, level: float = 0.95) -> list[list]:
    """Group models in rank order; start a new tier when the higher model beats the next one in >= level of resamples."""
    out = [[order[0]]]
    for hi, lo in zip(order, order[1:]):
        if float(np.mean(boot[hi] > boot[lo])) >= level:
            out.append([lo])
        else:
            out[-1].append(lo)
    return out

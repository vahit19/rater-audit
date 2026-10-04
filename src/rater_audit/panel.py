"""Audits on a long table of ratings: columns item, rater, label (labels are hashable, e.g. 0/1 or 'a'/'b'/'tie')."""
import numpy as np
import pandas as pd

from .stats import wilson_ci, krippendorff_alpha_nominal


def majority(ratings: pd.DataFrame) -> pd.Series:
    """Majority label per item; items with a tied vote get NaN."""
    def vote(s):
        c = s.value_counts()
        return c.index[0] if len(c) == 1 or c.iloc[0] > c.iloc[1] else np.nan
    return ratings.groupby("item")["label"].agg(vote)


def alpha(ratings: pd.DataFrame) -> float:
    """Krippendorff's alpha (nominal) over the whole table."""
    codes = {v: i for i, v in enumerate(sorted(ratings["label"].astype(str).unique()))}
    wide = ratings.assign(code=ratings["label"].astype(str).map(codes)).pivot_table(
        index="item", columns="rater", values="code", aggfunc="first")
    return krippendorff_alpha_nominal(wide.to_numpy(dtype=float))


def rater_accuracy(ratings: pd.DataFrame, gold: pd.Series) -> pd.DataFrame:
    """Accuracy of each rater on items with a known answer, with a Wilson interval."""
    d = ratings[ratings["item"].isin(gold.index)].copy()
    d["correct"] = d["label"].to_numpy() == gold.loc[d["item"]].to_numpy()
    out = d.groupby("rater")["correct"].agg(n="size", k="sum").reset_index()
    out["accuracy"] = out["k"] / out["n"]
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(out["k"], out["n"])]
    out["lo"], out["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return out


def agreement_with_panel(ratings: pd.DataFrame) -> pd.DataFrame:
    """For each rater: how often their label matches the majority of the OTHER raters on the same item."""
    rows = []
    for item, d in ratings.groupby("item"):
        if len(d) < 3:
            continue
        for i, r in d.iterrows():
            others = d.drop(index=i)["label"].value_counts()
            if len(others) > 1 and others.iloc[0] == others.iloc[1]:
                continue
            rows.append((r["rater"], r["label"] == others.index[0]))
    a = pd.DataFrame(rows, columns=["rater", "agree"])
    out = a.groupby("rater")["agree"].agg(n="size", k="sum").reset_index()
    out["agreement"] = out["k"] / out["n"]
    return out


def two_sigma_screen(agreement: pd.DataFrame, column: str = "agreement") -> list:
    """Raters a standard screen would drop: agreement more than two standard deviations below the panel mean."""
    m, s = agreement[column].mean(), agreement[column].std(ddof=1)
    return agreement.loc[agreement[column] < m - 2 * s, "rater"].tolist()


def bottom_screen(agreement: pd.DataFrame, k: int, column: str = "agreement") -> list:
    """Raters a 'drop the k least-agreeing' screen would remove."""
    return agreement.nsmallest(k, column)["rater"].tolist()


def canary_check(ratings: pd.DataFrame, canary_answers: pd.Series, human_rate: float,
                 margin: float = 0.0) -> pd.DataFrame:
    """Canary items: items where a language model reliably gives a known wrong answer.
    A rater whose answers match the model's on these items far more often than people do is flagged.
    Flag when the lower end of the rater's match-rate interval is above human_rate + margin."""
    d = ratings[ratings["item"].isin(canary_answers.index)].copy()
    d["match"] = d["label"].to_numpy() == canary_answers.loc[d["item"]].to_numpy()
    out = d.groupby("rater")["match"].agg(n="size", k="sum").reset_index()
    out["match_rate"] = out["k"] / out["n"]
    out["lo"] = [wilson_ci(int(k), int(n))[0] for k, n in zip(out["k"], out["n"])]
    out["flagged"] = out["lo"] > human_rate + margin
    return out

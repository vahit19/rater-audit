"""Shared figure style; figures are saved as PNG for the README."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
RES = ROOT / "results"

BLUE, RED, GRAY, GREEN, ORANGE = "#2f6db5", "#c8423a", "#8a96a3", "#2e8b57", "#d98c1f"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _clean(x):
    """Strict JSON: NaN becomes null, numpy scalars become plain numbers."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and x != x:
        return None
    return x


def write_json(name: str, obj) -> None:
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(_clean(obj), indent=2, allow_nan=False), encoding="utf-8")

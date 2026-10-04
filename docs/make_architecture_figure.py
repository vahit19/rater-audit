"""Draws figures/fig11_system.png: how rater-audit sits next to a rating platform in production."""
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.plotting import plt, save, BLUE, GRAY, GREEN, RED  # noqa: E402

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(0.1, 6.2)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, title, lines, color, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, fontweight="bold", color=tc)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.55 - i * 0.27, ln, ha="center", va="top", fontsize=8.3, color=tc)


def arrow(x1, y1, x2, y2, text=None, dy=0.12, color="#1f2933", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, lw=1.2, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", fontsize=7.8, color=color)


box(0.1, 4.3, 2.2, 1.6, "Client contract", ["defect limit, e.g. 5%", "task types, rubric", "acceptance terms"], GRAY)
box(0.1, 1.4, 2.2, 2.4, "Rating platform", ["(existing system)", "task queue, experts", "gold and canary items", "mixed into queues", "routing rules"], GRAY)
box(3.0, 0.7, 5.6, 5.2, "", [], "#e8eef6", tc="#1f2933")
ax.text(5.8, 5.72, "rater-audit service (runs per batch, inside the client's environment)", ha="center", fontsize=9.5,
        fontweight="bold", color=BLUE)
steps = [("plan", "size gold, canaries, review"), ("validate", "data contract check"), ("measure", "accuracy on gold, by type"),
         ("raters", "gold accuracy, canaries"), ("route", "weak types to qualified"), ("accept", "sized review, decision"),
         ("report", "QA summary, results.json, gate")]
for i, (t, s) in enumerate(steps):
    y = 5.15 - i * 0.62
    box(3.3, y - 0.42, 5.0, 0.5, "", [], BLUE)
    ax.text(3.5, y - 0.17, f"{i + 1}  {t}", ha="left", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(5.1, y - 0.17, s, ha="left", va="center", fontsize=8.5, color="white")
    if i:
        arrow(5.8, y + 0.2, 5.8, y + 0.08)
box(9.3, 4.35, 2.6, 1.5, "To the client", ["QA summary per batch", "defect rate with a range", "accept / reject"], GREEN)
box(9.3, 2.45, 2.6, 1.5, "To delivery ops", ["route type B to reviewers", "keep / retrain raters", "review canary flags"], RED)
box(9.3, 0.75, 2.6, 1.3, "To dashboards", ["results.json per batch", "trends, alerts", "exit code: CI gate"], GRAY)
arrow(2.3, 5.1, 3.3, 5.0, "limits", dy=0.1)
arrow(2.3, 2.9, 3.3, 3.9, "batch export", dy=0.15)
arrow(8.3, 1.1, 9.3, 5.0, None)
arrow(8.3, 1.1, 9.3, 3.2, None)
arrow(8.3, 1.1, 9.3, 1.4, None)
ax.plot([9.3, 8.95, 8.95, 1.2], [2.6, 2.6, 0.35, 0.35], color=RED, lw=1.2)
arrow(1.2, 0.35, 1.2, 1.4, None, color=RED)
ax.text(5.8, 0.42, "routing and rater actions go back into the platform; lessons go into the next batch's guidelines and gold set",
        ha="center", fontsize=8, color=RED)
save(fig, "fig11_system")
print("ok")

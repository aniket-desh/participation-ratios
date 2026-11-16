"""Batch 3b figure: every graded comparison's Delta, in units of its threshold, under three gradings.

Usage: python plot.py   (from the project root; reads Batch 3's grades.json and this folder's two grades files)

x = Delta / threshold, error bar = +-2 SD / threshold. In these units the verdict rule reads off the plot:
a size pass needs |x| > 1 and the bar to exclude 0; a null pass needs |x| < 1 and the bar's half-length < 1.
(The step-0 and prediction-quality conditions are not drawn; they are in the grades files.)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
G = {"Batch 3, seeds 0-2": json.loads((HERE.parent / "transformer-ablations" / "grades.json").read_text()),
     "seeds 3-5 alone": json.loads((HERE / "grades_seeds3-5.json").read_text()),
     "seeds 0-5 pooled": json.loads((HERE / "grades_seeds0-5.json").read_text())}
COL = dict(zip(G, ["0.55", "C1", "C0"]))
NAMES = {"A1 nomlp vs base @resid_post": "A1 attn-only vs base", "A2 preln vs base @resid_post": "A2 pre-LN vs base",
         "A2 preln ln2 vs resid_mid": "A2 within pre-LN: ln2 vs resid_mid",
         "A3 base resid_post vs resid_mid": "A3 within base: resid_post vs resid_mid"}
first = next(iter(G.values()))["grades"]
rows = [(c, m, i) for c in first for m in ["belief", "pr"] for i in range(2)]

fig, ax = plt.subplots(figsize=(10, 9))
for k, (lab, g) in enumerate(G.items()):
    for j, (c, m, i) in enumerate(rows):
        p = g["grades"][c][m]["per_condition"][i]
        x, e = p["delta"] / p["thr"], 2 * p["sd"] / p["thr"]
        y = j + (k - 1) * 0.25
        if np.isfinite(x):
            ax.errorbar(x, y, xerr=e, fmt="o", c=COL[lab], ms=4, capsize=2, label=lab if j == 0 else None)
        elif k == 1:
            ax.text(0, j, "undefined: no attention-only seed passes the prediction bar", fontsize=7, va="center",
                    ha="center", c="0.3")
labels = [f"{NAMES[c]}  [{'belief' if m == 'belief' else 'PR'}]  q2={first[c][m]['per_condition'][i]['q2']}"
          for c, m, i in rows]
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
for v in (-1, 1):
    ax.axvline(v, c="k", ls="--", lw=0.8)
ax.axvline(0, c="k", lw=0.5)
for j in range(1, len(rows) // 4):
    ax.axhline(4 * j - 0.5, c="0.7", lw=0.6)
ax.set_xscale("symlog", linthresh=2)  # linear within +-2 thresholds, where the verdicts are decided; log beyond
ax.set_xticks([-10, -5, -2, -1, 0, 1, 2, 5, 10]); ax.set_xticklabels(["-10", "-5", "-2", "-1", "0", "1", "2", "5", "10"])
ax.set_xlabel("Δ / threshold  (symlog; bars ±2·SD / threshold; dashed = threshold)")
ax.set_title("Batch 3's comparisons at step 20,000, three seed sets\n"
             "(3, 3 and 6 seeds per cell; thresholds 0.1 belief, 5% PR)", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
(HERE / "figures").mkdir(exist_ok=True)
fig.savefig(HERE / "figures" / "deltas.png", dpi=150)
print("wrote", HERE / "figures" / "deltas.png")

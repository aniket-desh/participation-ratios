"""Batch 4 figure: belief-component PR vs reliability, predicted curve vs per-seed observations; plus full PR(C_h)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text()); g = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
from prt.generators import two_factor_hmm, belief_coordinates   # dense generator-only curve (T = 16, MC)
from prt.spectra import weighted_cov
qs = np.linspace(0.15, 0.9, 16); curve = []
for q in qs:
    h = two_factor_hmm(0.8, 0.5, 0.8, q, perm=[2, 0, 3, 1])
    m = belief_coordinates(h, h.filter(h.sample(100_000, 16, np.random.default_rng(0))))["m"]
    _, C = weighted_cov(m); curve.append(np.trace(C) ** 2 / np.sum(C * C))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(qs, curve, "k-", lw=2, label="locked prediction: isometric in m (generator only)")
for r in g["runs"].values():
    held = r["q2"] in (0.35, 0.65)
    ax[0].scatter(r["q2"], r["pr_belief_m"], c="C3" if held else "C0", marker="o" if r["passing"] else "x", s=25, zorder=3)
    ax[1].scatter(r["q2"], r["pr_Ch"], c="C3" if held else "C0", marker="o" if r["passing"] else "x", s=25)
ax[0].scatter([], [], c="C3", label="held-out reliability (per seed)"); ax[0].scatter([], [], c="C0", label="discovery reliability, new seeds")
ax[0].set_xlabel("q2"); ax[0].set_ylabel("PR of token-partialled belief component"); ax[0].legend(fontsize=7)
ax[0].set_title("belief-component PR at step 20,000 (5 seeds per q2; x = not passing)")
ax[1].set_xlabel("q2"); ax[1].set_ylabel("PR(C_h), full final-position residual"); ax[1].set_title("full covariance PR (not predicted)")
fig.tight_layout(); fig.savefig(out / "pr_curve.png", dpi=130); print("figure ->", out)

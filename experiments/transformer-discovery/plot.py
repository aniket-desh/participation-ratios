"""Figures for Batch 2 from results.json (plotting is separate from computation)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
main = [r for r in recs if r["run"].startswith("main_") and r["analysis"] == "analysis"]
Q2 = sorted({r["q2"] for r in main})
col = dict(zip(Q2, ["C0", "C1", "C2"]))
SITE = "L0.resid_post"

# 1. Trajectories: PR(C_h), PR(C_belief[m]) and final-position excess KL vs step.
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for run in sorted({r["run"] for r in main}):
    rs = sorted((r for r in main if r["run"] == run), key=lambda r: r["step"])
    st = np.array([max(r["step"], 0.5) for r in rs]); c = col[rs[0]["q2"]]
    ax[0].plot(st, [r["sites"][SITE]["C_h"]["pr"] for r in rs], c=c, alpha=.7)
    ax[1].plot(st, [r["sites"][SITE]["m"]["pr_C_belief"] for r in rs], c=c, alpha=.7)
    ax[2].plot(st, [r["kl_final"] / r["kl_uniform_final"] for r in rs], c=c, alpha=.7)
for q in Q2:
    g = next(r for r in main if r["q2"] == q)["generator"]
    ax[1].axhline(g["pr_factored"], c=col[q], ls=":", lw=1)
    ax[0].plot([], [], c=col[q], label=f"q2={q}")
ax[0].legend(); ax[0].set_title(f"PR(C_h), {SITE}"); ax[1].set_title("PR(C_belief), b=m (dotted: isometric factored)")
ax[2].set_title("final-pos KL(Bayes||model) / KL(Bayes||uniform)"); ax[2].set_yscale("log")
for a in ax: a.set_xscale("log"); a.set_xlabel("step (0 plotted at 0.5)")
fig.tight_layout(); fig.savefig(out / "trajectories.png", dpi=130)

# 2. Final-checkpoint variance explained by each declared coordinate set, per site.
last = max(r["step"] for r in main)
fin = [r for r in main if r["step"] == last]
sites = list(fin[0]["sites"]); sets = ["m", "u", "m_aug", "last_token", "m_token", "u_token"]
fig, ax = plt.subplots(1, len(sites), figsize=(4 * len(sites), 4), sharey=True)
for i, s in enumerate(sites):
    for j, cs in enumerate(sets):
        for q in Q2:
            v = [r["sites"][s][cs]["r2_heldout"] for r in fin if r["q2"] == q and cs in r["sites"][s]]
            ax[i].scatter([j + 0.15 * (Q2.index(q) - 1)] * len(v), v, c=col[q], s=12)
    ax[i].set_xticks(range(len(sets))); ax[i].set_xticklabels(sets, rotation=45); ax[i].set_title(s)
ax[0].set_ylabel("held-out R^2 of forward regression b -> h")
fig.tight_layout(); fig.savefig(out / "variance_explained.png", dpi=130)

# 3. Learned metric: PR(C_belief) vs generator-only factored PR, and normalized G eigenvalues.
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for r in fin:
    g = r["generator"]
    ax[0].scatter(g["pr_factored"], r["sites"][SITE]["m"]["pr_C_belief"], c=col[r["q2"]])
    ax[1].scatter([r["q2"]] * 2, r["sites"][SITE]["m"]["G_norm_eigs"], c=col[r["q2"]])
lim = [1, 2.05]; ax[0].plot(lim, lim, "k:"); ax[0].set_xlabel("isometric factored PR (generator only)")
ax[0].set_ylabel("PR(C_belief), b=m, trained"); ax[1].axhline(.5, c="k", ls=":")
ax[1].set_xlabel("q2"); ax[1].set_ylabel("eig(G)/tr G, b=m")
fig.tight_layout(); fig.savefig(out / "metric.png", dpi=130)
print("figures ->", out)

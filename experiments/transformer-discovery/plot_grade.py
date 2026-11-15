"""Figure of the graded quantity (token-partialled belief block at L0.resid_post) from results.json and grade.json.

Usage: python plot_grade.py results.json grade.json figures/
Panel 1: top normalized eigenvalue of G_b vs training step, b = m (solid) and b = u (dashed), with the H1 band [0.4, 0.6].
Panel 2: PR(E_b Sigma_b E_b^T) at step 20,000 against the generator-only isometric PR of the same coordinates.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

recs = json.loads(Path(sys.argv[1]).read_text()); G = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3]); SITE = "L0.resid_post"
col = {0.25: "C0", 0.5: "C1", 0.8: "C2"}
main = sorted(k for k in G["runs"] if k.startswith("main_"))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for run in main:
    rs = sorted((r for r in recs if r["run"] == run and r["analysis"] == "analysis"), key=lambda r: r["step"])
    st = [max(r["step"], 0.5) for r in rs]; c = col[rs[0]["q2"]]
    ax[0].plot(st, [r["sites"][SITE]["m_token"]["partialled"]["G_norm_eigs"][0] for r in rs], c=c, alpha=.8)
    ax[0].plot(st, [r["sites"][SITE]["u_token"]["partialled"]["G_norm_eigs"][0] for r in rs], c=c, alpha=.5, ls="--")
ax[0].axhspan(0.4, 0.6, color="k", alpha=.07, label="H1 band [0.4, 0.6]")
for q, c in col.items():
    ax[0].plot([], [], c=c, label=f"q2={q}")
ax[0].plot([], [], c="k", label="b = m"); ax[0].plot([], [], c="k", ls="--", label="b = u")
ax[0].set_xscale("log"); ax[0].set_xlabel("training step (0 plotted at 0.5)")
ax[0].set_ylabel("top eig(G_b) / tr G_b"); ax[0].set_ylim(0.45, 1.0); ax[0].legend(fontsize=8, ncol=2)
ax[0].set_title("Token-partialled belief metric, 9 main runs (3 seeds x 3 q2)")

for run in main:
    v = G["runs"][run]
    ax[1].scatter(v["pr_m_iso"], v["m"]["pr"], c=col[v["q2"]], marker="o")
    ax[1].scatter(v["pr_u_iso"], v["u"]["pr"], c=col[v["q2"]], marker="x")
lim = [1.1, 2.05]; ax[1].plot(lim, lim, "k:", lw=1)
ax[1].fill_between(lim, [x * 0.9 for x in lim], [x * 1.1 for x in lim], color="k", alpha=.07, label="within 10%")
ax[1].scatter([], [], c="k", marker="o", label="b = m"); ax[1].scatter([], [], c="k", marker="x", label="b = u")
ax[1].set_xlabel("generator-only isometric PR of b (eval contexts, n = 65,536)")
ax[1].set_ylabel("PR(E_b Sigma_b E_b^T), trained, step 20,000"); ax[1].legend(fontsize=8)
ax[1].set_title("Partialled belief PR vs isometric prediction")
fig.tight_layout(); fig.savefig(out / "partialled_metric.png", dpi=130)
print("->", out / "partialled_metric.png")

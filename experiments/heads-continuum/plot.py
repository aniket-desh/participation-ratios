"""Figures for the Package 5 study, from results/*.json (no computation beyond summaries)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent
R = {p.stem: json.loads(p.read_text()) for p in (D / "results").glob("*.json")}
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# (a) continuum PR trajectory with finite-H runs
f = R["finite"]; tau = np.array(f["tau"])
a = ax[0, 0]
for H, c in zip(["16", "64", "256", "1024"], ["C3", "C1", "C2", "C0"]):
    for i, r in enumerate(f["runs"][H]):
        a.plot(tau, r["pr"], c=c, alpha=.25, lw=.8, label=f"H={H} ({len(f['runs'][H])} seeds)" if i == 0 else None)
a.plot(tau, f["pr_ref"], "k", lw=2.5, label="continuum (24x24 cubature)")
a.set_xlabel("tau"); a.set_ylabel("PR of S = P^1/2 K P^1/2"); a.set_title("(a) PR trajectory: continuum vs finite heads"); a.legend(fontsize=7)
a.set_ylim(1.98, 2.10)

# (b) H^{-1/2} scaling of RMS errors
a = ax[0, 1]
Hs = np.array([16, 64, 256, 1024])
for idx, ls in [(0, "-"), (10, "--"), (20, ":")]:
    fro = [np.sqrt(np.mean([r["fro_err"][idx] ** 2 for r in f["runs"][str(H)]])) for H in Hs]
    dpr = [np.sqrt(np.mean([(r["pr"][idx] - f["pr_ref"][idx]) ** 2 for r in f["runs"][str(H)]])) for H in Hs]
    a.loglog(Hs, fro, "C0" + ls, marker="o", label=f"RMS ||S_H - S||_F, tau={tau[idx]:.0f}")
    a.loglog(Hs, dpr, "C1" + ls, marker="s", label=f"RMS |PR_H - PR|, tau={tau[idx]:.0f}")
a.loglog(Hs, 0.9 * Hs ** -0.5, "k", lw=.8, label="H^-1/2")
a.set_xlabel("heads H"); a.set_title("(b) finite-head error vs H"); a.legend(fontsize=6)

# (c) cubature / step-size convergence
a = ax[0, 2]
q = R["quad"]; ref = np.array(q["n24_dt0.2"]["pr"])
for k, v in q.items():
    if k != "n24_dt0.2":
        a.semilogy(v["tau"], np.abs(np.array(v["pr"]) - ref) + 1e-16, label=f"{k} vs n24_dt0.2")
a.set_xlabel("tau"); a.set_title("(c) solver convergence: |PR difference|"); a.legend(fontsize=7)

# (d) weak-signal scaling of onset coefficients
a = ax[1, 0]
w = R["weak"]; e = np.array([r["eps"] for r in w])
for k, m in [("A", "o"), ("B", "s"), ("Kdd", "^")]:
    a.loglog(e, [r[k] for r in w], marker=m, label=f"max |{k}|")
a.loglog(e, [abs(r["prdd0"]) for r in w], marker="d", label="|PR''(0)|")
a.loglog(e, 1.5 * e**2, "k:", lw=.8, label="eps^2"); a.loglog(e, 0.17 * e**4, "k--", lw=.8, label="eps^4")
a.set_xlabel("eps (q_j = eps * qbar_j)"); a.set_title("(d) weak-signal scaling of the onset"); a.legend(fontsize=7)

# (e) generator sweep
a = ax[1, 1]
for q2, c in zip(["0.25", "0.5", "0.8"], ["C0", "C1", "C2"]):
    t = R["sweep"][q2]["traj"]
    a.plot(t["tau"], t["pr"], c, label=f"q2={q2}  PR''(0)={R['sweep'][q2]['onset']['prdd0']:.4f}")
a.set_xlabel("tau"); a.set_title("(e) continuum PR vs generator ((rho,q1)=(0.8,0.5), rho2=0.8, T=3)"); a.legend(fontsize=7)

# (f) controls
a = ax[1, 2]
if "controls" in R:
    for k, c in [("full", "k"), ("frozen_attention", "C1"), ("readout_only", "C2")]:
        a.plot(R["controls"][k]["tau"], R["controls"][k]["pr"], c, label=k)
    a.set_xlabel("tau"); a.set_title("(f) controls at the primary generator"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(D / "continuum.png", dpi=120)
print("wrote", D / "continuum.png")

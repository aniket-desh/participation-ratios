"""Batch 2b: trajectories and stability of the token-partialled belief metric and PR(C_h) to the final step.

    python stability.py results.json runs/transformer-longrun stability.json figures 50000
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text()); R = Path(sys.argv[2]); out = Path(sys.argv[3])
figs = Path(sys.argv[4]); figs.mkdir(parents=True, exist_ok=True); LAST = int(sys.argv[5])
SITE = "L0.resid_post"
runs = defaultdict(dict)
for r in recs:
    if r["analysis"] != "analysis":
        continue
    s = r["sites"][SITE]; g = r["generator"]
    pm, pu = s["m_token"]["partialled"], s["u_token"]["partialled"]
    runs[r["run"]][r["step"]] = dict(q2=r["q2"], seed=r["seed"], top_m=pm["G_norm_eigs"][0], top_u=pu["G_norm_eigs"][0],
                                     pr_belief_m=pm["pr_C_belief"], pred_m=g["pr_factored"], pr_Ch=s["C_h"]["pr"],
                                     gain=s["m_token"]["r2_heldout"] - s["last_token"]["r2_heldout"], kl_final=r["kl_final"])
res = {}
for name, d in runs.items():
    if LAST not in d or 20000 not in d:
        res[name] = dict(finished=False); continue
    rows = [json.loads(l) for l in open(R / name / "metrics.jsonl")]
    tm = lambda lo, hi: float(np.mean([x["kl_final"] for x in rows if lo <= x["step"] <= hi]))
    a, b = d[20000], d[LAST]
    res[name] = dict(finished=True, q2=a["q2"], seed=a["seed"], floor4=json.loads((R / name / "config.json").read_text())["kl_lastk_final"]["4"],
                     kl_tail_20k=tm(15000, 20000), kl_tail_last=tm(0.75 * LAST, LAST),
                     top_m={k: d[k]["top_m"] for k in sorted(d)}, top_u={k: d[k]["top_u"] for k in sorted(d)},
                     pr_belief_m={k: d[k]["pr_belief_m"] for k in sorted(d)}, pred_m=b["pred_m"],
                     pr_Ch={k: d[k]["pr_Ch"] for k in sorted(d)}, gain={k: d[k]["gain"] for k in sorted(d)},
                     d_top_m=b["top_m"] - a["top_m"], d_pr_Ch_rel=(b["pr_Ch"] - a["pr_Ch"]) / a["pr_Ch"],
                     in_band_last=0.4 <= b["top_m"] <= 0.6, e_pr_last=abs(b["pred_m"] - b["pr_belief_m"]) / b["pr_belief_m"],
                     m_closer_last=abs(b["top_m"] - .5) < abs(b["top_u"] - .5))
fin = [v for v in res.values() if v.get("finished")]
summary = dict(n_finished=len(fin), n_runs=len(res),
               frac_in_band_last=float(np.mean([v["in_band_last"] for v in fin])) if fin else None,
               frac_e_pr_lt_0p1_last=float(np.mean([v["e_pr_last"] < 0.1 for v in fin])) if fin else None,
               frac_m_closer_last=float(np.mean([v["m_closer_last"] for v in fin if v["q2"] != 0.5])) if fin else None,
               median_abs_d_top_m=float(np.median([abs(v["d_top_m"]) for v in fin])) if fin else None,
               median_d_pr_Ch_rel=float(np.median([v["d_pr_Ch_rel"] for v in fin])) if fin else None)
out.write_text(json.dumps(dict(summary=summary, runs=res), indent=1)); print(json.dumps(summary, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); col = {0.25: "C0", 0.5: "C1", 0.8: "C2"}
for v in fin:
    st = list(v["top_m"]); c = col[v["q2"]]
    ax[0].plot(st, list(v["top_m"].values()), c=c, alpha=.7); ax[0].plot(st, list(v["top_u"].values()), c=c, ls="--", alpha=.5)
    ax[1].plot(st, [v["pr_belief_m"][k] / v["pred_m"] for k in st], c=c, alpha=.7)
    ax[2].plot(st, list(v["pr_Ch"].values()), c=c, alpha=.7)
ax[0].axhspan(0.4, 0.6, color="grey", alpha=.15); ax[0].set_title("top eig of partialled G_b (solid m, dashed u)")
ax[1].axhline(1, c="k", ls=":"); ax[1].set_title("belief PR / m-isometric prediction"); ax[2].set_title("PR(C_h), full residual")
for a_ in ax: a_.set_xlabel("step"); a_.axvline(20000, c="k", lw=.5)
for q, c in col.items(): ax[2].plot([], [], c=c, label=f"q2={q}")
ax[2].legend(); fig.tight_layout(); fig.savefig(figs / "stability.png", dpi=130); print("figure ->", figs)

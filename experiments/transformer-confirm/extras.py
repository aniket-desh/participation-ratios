"""Batch 4, reported-not-graded quantities (EXPERIMENT.md "Reported, not graded") and one trajectory figure.

    python extras.py results.json grade.json extras.json figures/

Written by the supervising session after grading, from results.json and grade.json only; it changes no verdict.
All quantities are at site L0.resid_post, token-partialled belief block, b = m unless named otherwise.
e_PR = |pred - obs| / obs, as in grade.py.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text()); g = json.loads(Path(sys.argv[2]).read_text())
out, figdir = Path(sys.argv[3]), Path(sys.argv[4]); figdir.mkdir(parents=True, exist_ok=True)
SITE, Q2S, STEPS = "L0.resid_post", ["0.2", "0.25", "0.35", "0.5", "0.65", "0.8", "0.9"], [0, 100, 1000, 5000, 20000]
HELD = ["0.2", "0.35", "0.65", "0.9"]
runs = g["runs"]
by = defaultdict(list)
for r in runs.values():
    by[str(r["q2"])].append(r)


def part(run, step, cs="m_token"):
    x = next(x for x in recs if x["run"] == run and x["step"] == step and x["analysis"] == "analysis")
    return x["sites"][SITE][cs]["partialled"], x


def ms(v):
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v)), max=float(np.max(v)))


res = {}
# Per-reliability: e_PR of each run against the m-isometric prediction (mean, seed SD), top eigenvalue, factor-1 share.
per = {}
for q in Q2S:
    names = [n for n, r in runs.items() if str(r["q2"]) == q]
    e = [abs(runs[n]["pred"]["m_isometric"] - runs[n]["pr_belief_m"]) / runs[n]["pr_belief_m"] for n in names]
    f1 = [part(n, 20000)[0]["factor1_share"] for n in names]
    f1_0 = [part(n, 0)[0]["factor1_share"] for n in names]
    # Where the observed PR sits between the u-isometric (0) and m-isometric (1) predictions.
    pm, pu = runs[names[0]]["pred"]["m_isometric"], runs[names[0]]["pred"]["u_isometric"]
    obs = np.mean([runs[n]["pr_belief_m"] for n in names])
    per[q] = dict(e_PR_m_per_run=ms(e), n_e_below_0p1=int(sum(x < 0.1 for x in e)), top_m=ms([runs[n]["top_m"] for n in names]),
                  top_m_step0=ms([runs[n]["top_m_step0"] for n in names]), n_in_band=int(sum(runs[n]["in_band"] for n in names)),
                  n_in_band_step0=int(sum(runs[n]["in_band_step0"] for n in names)),
                  factor1_share_m=ms(f1), factor1_share_m_step0=ms(f1_0),
                  pos_between_u0_m1=float((obs - pu) / (pm - pu)) if abs(pm - pu) > 1e-3 else None,
                  delta=ms([runs[n]["delta"] for n in names]),
                  kl_tail_over_floor4=ms([runs[n]["kl_tailmean"] / runs[n]["floor4"] for n in names]),
                  n_pass_single_final_eval=int(sum(runs[n]["kl_single"] < runs[n]["floor4"] and runs[n]["delta"] > 0.02 for n in names)),
                  generator_v1_v2=[part(names[0], 20000)[1]["generator"]["v1"], part(names[0], 20000)[1]["generator"]["v2"]])
res["per_reliability"] = per
allr = list(runs.values())
res["pooled"] = dict(
    in_band_frac=float(np.mean([r["in_band"] for r in allr])), in_band_frac_step0=float(np.mean([r["in_band_step0"] for r in allr])),
    e_m_below_0p1_frac_all=float(np.mean([abs(r["pred"]["m_isometric"] - r["pr_belief_m"]) / r["pr_belief_m"] < 0.1 for r in allr])),
    e_m_below_0p1_frac_heldout=float(np.mean([abs(r["pred"]["m_isometric"] - r["pr_belief_m"]) / r["pr_belief_m"] < 0.1
                                              for q in HELD for r in by[q]])),
    n_pass_single_final_eval=int(sum(r["kl_single"] < r["floor4"] and r["delta"] > 0.02 for r in allr)),
    p3_counts={q: int(sum(r["m_closer"] for r in by[q])) for q in Q2S if q != "0.5"})
# Trajectories: seed mean (and SD) of belief PR and top eigenvalue at each analysed step.
traj = {}
for q in Q2S:
    names = [n for n, r in runs.items() if str(r["q2"]) == q]
    traj[q] = {str(s): dict(pr_belief_m=ms([part(n, s)[0]["pr_C_belief"] for n in names]),
                            top_m=ms([part(n, s)[0]["G_norm_eigs"][0] for n in names]),
                            top_u=ms([part(n, s, "u_token")[0]["G_norm_eigs"][0] for n in names])) for s in STEPS}
res["trajectories"] = traj
out.write_text(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cm = plt.get_cmap("viridis")
for i, q in enumerate(Q2S):
    c = cm(i / (len(Q2S) - 1)); xs = [max(s, 10) for s in STEPS]
    t = [traj[q][str(s)]["top_m"]["mean"] for s in STEPS]; sd = [traj[q][str(s)]["top_m"]["sd"] for s in STEPS]
    ax[0].errorbar(xs, t, yerr=sd, color=c, marker="o", ms=3, capsize=2, label=f"q2 = {q}" + (" (held out)" if q in HELD else ""))
    p = [traj[q][str(s)]["pr_belief_m"]["mean"] for s in STEPS]
    ax[1].plot(xs, p, color=c, marker="o", ms=3)
    ax[1].axhline(runs[next(n for n, r in runs.items() if str(r["q2"]) == q)]["pred"]["m_isometric"], color=c, ls=":", lw=1)
ax[0].axhspan(0.4, 0.6, color="grey", alpha=0.15, label="P1 band [0.4, 0.6]")
ax[0].set_xscale("log"); ax[0].set_xlabel("training step (step 0 drawn at 10)"); ax[0].set_ylabel("top eigenvalue of normalized G_b (b = m)")
ax[0].set_title("belief-metric anisotropy, seed mean ± SD (n = 5)"); ax[0].legend(fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xlabel("training step (step 0 drawn at 10)"); ax[1].set_ylabel("PR of token-partialled belief component")
ax[1].set_title("belief PR, seed mean (dotted: m-isometric prediction)")
fig.tight_layout(); fig.savefig(figdir / "trajectories.png", dpi=130)
print(json.dumps({q: dict(e=round(v["e_PR_m_per_run"]["mean"], 3), e_sd=round(v["e_PR_m_per_run"]["sd"], 3), n_e=v["n_e_below_0p1"],
                         top=round(v["top_m"]["mean"], 3), inb=v["n_in_band"], inb0=v["n_in_band_step0"], top0=round(v["top_m_step0"]["mean"], 3),
                         f1=round(v["factor1_share_m"]["mean"], 3), f1_0=round(v["factor1_share_m_step0"]["mean"], 3),
                         pos=None if v["pos_between_u0_m1"] is None else round(v["pos_between_u0_m1"], 2),
                         klr=[round(v["kl_tail_over_floor4"]["min"], 2), round(v["kl_tail_over_floor4"]["max"], 2)],
                         single=v["n_pass_single_final_eval"], v=[round(x, 3) for x in v["generator_v1_v2"]]) for q, v in per.items()}, indent=0))
print(json.dumps(res["pooled"]))
for q in Q2S:
    print(q, " ".join(f"{s}:{traj[q][str(s)]['top_m']['mean']:.3f}/{traj[q][str(s)]['pr_belief_m']['mean']:.3f}" for s in STEPS))

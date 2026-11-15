"""Grade Batch 2b (transformer-longrun) against its definition's criteria. Written before any result was read.

    python grade.py stability.json ../transformer-discovery/results.json results.json grade.json

Inputs: stability.json (per-run trajectories at L0.resid_post, from stability.py), Batch 2's results.json (the 20k
reference for seeds 0-2), and this experiment's results.json (for per-step PR(C_h) and the success criterion).

Rules, from EXPERIMENT.md "Success and failure criteria":
  succeeded      >= 10 of 12 runs reach 50k with analysis at 20k and 50k
  eligible       finished, and last-quarter tail-mean final-position KL (steps 37.5k-50k) strictly below the k=4 floor
  stable         >= 80% of eligible in band [0.4, 0.6] at 50k (top eig of partialled G_b in m; both eigs sum to 1,
                 so this is the same as both in band), >= 80% of eligible anisotropic (q2 != 0.5) runs closer to
                 0.5 in m than in u at 50k, and median |top_m(50k) - top_m(20k)| <= 0.05
  drift          median |d top_m| > 0.05, or < 50% of eligible in band at 50k
                 direction "toward u" if in >= 80% of eligible anisotropic runs the distance gap
                 (|top_u - .5| - |top_m - .5|) is smaller at 50k than at 20k
  inconclusive   otherwise
Reported, not graded: PR(C_h) median relative change 40k -> 50k; seeds 0-2 against Batch 2 at 20k; q2 = 0.8 seeds
3-5 against Batch 2's outlier (main_q0.8_s2, top_m 0.615); belief PR / m-isometric prediction over time.
The verdict is also computed over all finished runs, ignoring the KL bar, as a sensitivity check.
"""
import json
import sys
from pathlib import Path

import numpy as np

stab = json.loads(Path(sys.argv[1]).read_text())["runs"]
b2 = json.loads(Path(sys.argv[2]).read_text())
recs = json.loads(Path(sys.argv[3]).read_text())
out = Path(sys.argv[4])
SITE, LAST, MID = "L0.resid_post", "50000", "20000"   # stability.json keys are strings after the JSON round trip

fin = {n: v for n, v in stab.items() if v.get("finished")}
dist = lambda x: abs(x - 0.5)


def grade(runs):
    if not runs:
        return dict(n=0, verdict="no runs")
    inband = [v["in_band_last"] for v in runs.values()]
    aniso = {n: v for n, v in runs.items() if v["q2"] != 0.5}
    m_closer = [dist(v["top_m"][LAST]) < dist(v["top_u"][LAST]) for v in aniso.values()]
    d = [abs(v["top_m"][LAST] - v["top_m"][MID]) for v in runs.values()]
    gap = lambda v, k: dist(v["top_u"][k]) - dist(v["top_m"][k])
    toward_u = [gap(v, LAST) < gap(v, MID) for v in aniso.values()]
    f_band, f_closer, med = float(np.mean(inband)), (float(np.mean(m_closer)) if m_closer else None), float(np.median(d))
    f_tu = float(np.mean(toward_u)) if toward_u else None
    stable = f_band >= 0.8 and (f_closer is not None and f_closer >= 0.8) and med <= 0.05
    drift = med > 0.05 or f_band < 0.5
    verdict = "stable m-isometry" if stable else "drift" if drift else "inconclusive"
    return dict(n=len(runs), n_aniso=len(aniso), frac_in_band_50k=f_band, frac_m_closer_50k=f_closer,
                median_abs_d_top_m=med, frac_gap_shrinks=f_tu, toward_u=(f_tu is not None and f_tu >= 0.8),
                verdict=verdict)


elig = {n: v for n, v in fin.items() if v["kl_tail_last"] < v["floor4"]}
flagged = {n: dict(kl_tail_last=v["kl_tail_last"], floor4=v["floor4"], d_top_m=v["d_top_m"],
                   top_m_20k=v["top_m"][MID], top_m_50k=v["top_m"][LAST]) for n, v in fin.items() if n not in elig}

# PR(C_h) settling, 40k -> 50k
pr_rel = {n: (v["pr_Ch"][LAST] - v["pr_Ch"]["40000"]) / v["pr_Ch"]["40000"] for n, v in fin.items() if "40000" in v["pr_Ch"]}
pr_rel_20_50 = {n: v["d_pr_Ch_rel"] for n, v in fin.items()}

# Reproducibility of seeds 0-2 against Batch 2 at 20k
ref = {}
for r in b2:
    if r["analysis"] == "analysis" and r["step"] == 20000 and r["run"].startswith("main_"):
        s = r["sites"][SITE]
        ref[r["run"].removeprefix("main_")] = dict(top_m=s["m_token"]["partialled"]["G_norm_eigs"][0], pr_Ch=s["C_h"]["pr"])
repro = {}
for n, v in fin.items():
    if n in ref:
        repro[n] = dict(top_m_b2=ref[n]["top_m"], top_m_here=v["top_m"][MID], d_top_m=v["top_m"][MID] - ref[n]["top_m"],
                        pr_Ch_b2=ref[n]["pr_Ch"], pr_Ch_here=v["pr_Ch"][MID],
                        d_pr_Ch_rel=(v["pr_Ch"][MID] - ref[n]["pr_Ch"]) / ref[n]["pr_Ch"])

# q2 = 0.8: all seeds' top_m at 20k and 50k, against Batch 2's outlier 0.615 (main_q0.8_s2)
q08 = {n: dict(seed=v["seed"], top_m={k: v["top_m"][k] for k in (MID, LAST)}, top_u={k: v["top_u"][k] for k in (MID, LAST)},
               eligible=n in elig) for n, v in fin.items() if v["q2"] == 0.8}

# Belief PR / m-isometric prediction over time
ratio = {n: {k: v["pr_belief_m"][k] / v["pred_m"] for k in v["pr_belief_m"]} for n, v in fin.items()}

# Success: count runs with analysis files at 20k and 50k (from results.json)
have = {}
for r in recs:
    if r["analysis"] == "analysis":
        have.setdefault(r["run"], set()).add(r["step"])
n_ok = sum(1 for s in have.values() if {20000, 50000} <= s)

res = dict(succeeded=n_ok >= 10, n_runs_with_20k_and_50k=n_ok, n_runs_seen=len(have),
           graded=grade(elig), sensitivity_all_finished=grade(fin), flagged_above_floor=flagged,
           pr_Ch_rel_40k_50k=pr_rel, median_pr_Ch_rel_40k_50k=float(np.median(list(pr_rel.values()))) if pr_rel else None,
           median_abs_pr_Ch_rel_40k_50k=float(np.median(np.abs(list(pr_rel.values())))) if pr_rel else None,
           pr_Ch_rel_20k_50k=pr_rel_20_50, repro_vs_batch2_20k=repro, q08_seeds=q08, belief_pr_over_pred=ratio,
           per_run={n: dict(q2=v["q2"], seed=v["seed"], eligible=n in elig, kl_tail_last=v["kl_tail_last"],
                            kl_tail_20k=v["kl_tail_20k"], floor4=v["floor4"], top_m=v["top_m"], top_u=v["top_u"],
                            pr_Ch=v["pr_Ch"], gain=v["gain"], d_top_m=v["d_top_m"]) for n, v in fin.items()})
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ["succeeded", "n_runs_with_20k_and_50k", "graded", "sensitivity_all_finished",
                                      "median_pr_Ch_rel_40k_50k"]}, indent=1))

"""Grade Batch 4 against the locked predictions (EXPERIMENT.md "Success and failure criteria").

    python grade.py results.json runs/transformer-confirm predictions.json grade.json 20000

A run without a step-STEPS analysis counts as not finished and not passing. Passing run: tail-mean final-position KL
(evaluations at step >= 0.75*STEPS) below the k = 4 floor, and belief-content gain Delta > 0.02.
The P2 rule is applied to the locked prediction and, identically, to three alternatives (constant PR 2 = whitened
encoding, the u-isometric curve, and each run's own step-0 belief PR), so a P2 pass counts as discriminating only
where the alternatives fail.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text())
R, preds, out, LAST = Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]), int(sys.argv[5])
SITE = preds["site"]
Q2S = [0.2, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9]
HELDOUT = ["0.2", "0.35", "0.65", "0.9"]
ANISO = ["0.2", "0.25", "0.35", "0.65", "0.8", "0.9"]


def rec(run, step):
    return next((r for r in recs if r["run"] == run and r["step"] == step and r["analysis"] == "analysis"), None)


runs, unfinished = {}, []
for name in sorted({r["run"] for r in recs}):
    f, f0 = rec(name, LAST), rec(name, 0)
    if f is None or f0 is None:
        unfinished.append(name); continue
    s, s0, g = f["sites"][SITE], f0["sites"][SITE], f["generator"]
    rows = [json.loads(l) for l in open(R / name / "metrics.jsonl")]
    tail = [r["kl_final"] for r in rows if r["step"] >= 0.75 * LAST]
    floor4 = f["kl_lastk_final"]["4"]
    dlt = (s["m_token"]["r2_heldout"] - s["last_token"]["r2_heldout"]) - (s0["m_token"]["r2_heldout"] - s0["last_token"]["r2_heldout"])
    pm, pu, pm0 = s["m_token"]["partialled"], s["u_token"]["partialled"], s0["m_token"]["partialled"]
    obs = pm["pr_C_belief"]
    runs[name] = dict(q2=f["q2"], seed=f["seed"], kl_tailmean=float(np.mean(tail)), floor4=floor4, kl_single=f["kl_final"],
                      delta=dlt, passing=bool(np.mean(tail) < floor4 and dlt > 0.02),
                      pr_belief_m=obs, pr_belief_m_step0=pm0["pr_C_belief"],
                      pred={"m_isometric": g["pr_factored"], "whitened_const2": 2.0, "u_isometric": g["pr_u_isometric"],
                            "step0_own": pm0["pr_C_belief"]},
                      top_m=pm["G_norm_eigs"][0], top_m_step0=pm0["G_norm_eigs"][0],
                      in_band=all(0.4 <= e <= 0.6 for e in pm["G_norm_eigs"]),
                      in_band_step0=all(0.4 <= e <= 0.6 for e in pm0["G_norm_eigs"]),
                      toward_iso=abs(pm["G_norm_eigs"][0] - .5) < abs(pm0["G_norm_eigs"][0] - .5),
                      top_u=pu["G_norm_eigs"][0], m_closer=abs(pm["G_norm_eigs"][0] - .5) < abs(pu["G_norm_eigs"][0] - .5),
                      pr_Ch=s["C_h"]["pr"], pr_Ch_step0=s0["C_h"]["pr"])

by = defaultdict(list)
for r in runs.values():
    by[str(r["q2"])].append(r)
passing = {q: [r for r in by.get(q, []) if r["passing"]] for q in map(str, Q2S)}
ok = {q: ps for q, ps in passing.items() if len(ps) >= 2}


def e(pred, obs):
    return abs(pred - obs) / obs


def p2_rule(key, scope):
    """Seed-mean e_PR < 0.1 at every reliability in scope, and >= 80% of pooled passing runs with e_PR < 0.1."""
    if any(q not in ok for q in scope):
        return "inconclusive", {}
    per = {}
    for q in scope:
        obs = np.mean([r["pr_belief_m"] for r in ok[q]])
        pred = np.mean([r["pred"][key] for r in ok[q]])
        per[q] = dict(pred=float(pred), obs_mean=float(obs), obs_sd=float(np.std([r["pr_belief_m"] for r in ok[q]], ddof=1)),
                      e_mean=float(e(pred, obs)))
    pooled = [e(r["pred"][key], r["pr_belief_m"]) < 0.1 for q in scope for r in ok[q]]
    if all(v["e_mean"] < 0.1 for v in per.values()) and np.mean(pooled) >= 0.8:
        return "supported", per
    if any(v["e_mean"] >= 0.1 for v in per.values()):
        return "not supported", per
    return "inconclusive", per


ALL = [str(q) for q in Q2S]
res = dict(n_runs=len(runs) + len(unfinished), n_finished=len(runs), unfinished=unfinished,
           run_failed=len(runs) < 30 or any(not by.get(q) for q in HELDOUT),
           n_passing={q: len(ps) for q, ps in passing.items()})

# P1: geometry (pooled band + per-reliability seed-mean top eigenvalue), at step 20,000 and, for reference, step 0.
def p1(field_band, field_top):
    if any(q not in ok for q in ALL):
        return "inconclusive"
    pooled = np.mean([r[field_band] for q in ALL for r in ok[q]])
    tops = {q: np.mean([r[field_top] for r in ok[q]]) for q in ALL}
    if pooled >= 0.8 and all(t <= 0.6 for t in tops.values()):
        return "supported"
    if pooled < 0.5 or any(t > 0.6 for t in tops.values()):
        return "not supported"
    return "inconclusive"


res["P1"] = p1("in_band", "top_m")
res["P1_at_step0_reference"] = p1("in_band_step0", "top_m_step0")
pool = [r for q in ALL if q in ok for r in ok[q]]
res["P1_learned_toward_isotropy"] = dict(frac=float(np.mean([r["toward_iso"] for r in pool])) if pool else None,
                                         verdict=("supported" if pool and np.mean([r["toward_iso"] for r in pool]) >= 0.8 else
                                                  "not supported" if pool and np.mean([r["toward_iso"] for r in pool]) < 0.5 else "inconclusive"))
for scope_name, scope in [("all", ALL), ("heldout", HELDOUT)]:
    for key in ["m_isometric", "whitened_const2", "u_isometric", "step0_own"]:
        v, per = p2_rule(key, scope)
        res[f"P2_{scope_name}_{key}"] = dict(verdict=v, per_reliability=per)
for scope_name in ["all", "heldout"]:
    alts = [res[f"P2_{scope_name}_{k}"]["verdict"] for k in ["whitened_const2", "u_isometric", "step0_own"]]
    res[f"P2_{scope_name}_discriminating"] = (res[f"P2_{scope_name}_m_isometric"]["verdict"] == "supported" and all(a == "not supported" for a in alts))
# P3: pooled over anisotropic reliabilities, plus no reliability below 50%.
if any(q not in ok for q in ANISO):
    res["P3"] = "inconclusive"
else:
    fr = {q: np.mean([r["m_closer"] for r in ok[q]]) for q in ANISO}
    pooled = np.mean([r["m_closer"] for q in ANISO for r in ok[q]])
    res["P3"] = "supported" if pooled >= 0.8 and min(fr.values()) >= 0.5 else ("not supported" if pooled < 0.5 or min(fr.values()) < 0.5 else "inconclusive")
    res["P3_frac_by_reliability"] = {q: float(v) for q, v in fr.items()}
if all(q in ok for q in ALL):
    pr_obs = {q: np.mean([r["pr_belief_m"] for r in ok[q]]) for q in ALL}
    pr_pred = {q: np.mean([r["pred"]["m_isometric"] for r in ok[q]]) for q in ALL}
    res["ordering"] = dict(predicted=sorted(ALL, key=pr_pred.get), observed=sorted(ALL, key=pr_obs.get))
    res["ordering"]["same"] = res["ordering"]["predicted"] == res["ordering"]["observed"]
res["pr_Ch_by_reliability"] = {q: dict(mean=float(np.mean([r["pr_Ch"] for r in ps])), sd=float(np.std([r["pr_Ch"] for r in ps], ddof=1)))
                               for q, ps in ok.items()}
res["runs"] = runs
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: (v if not isinstance(v, dict) or "per_reliability" not in v else v["verdict"]) for k, v in res.items() if k != "runs"}, indent=1))

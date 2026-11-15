"""Grade Batch 4 against the locked predictions (EXPERIMENT.md "Success and failure criteria").

    python grade.py results.json runs/transformer-confirm predictions.json grade.json 20000

Passing run: tail-mean final-position KL (evaluations at step >= 0.75*STEPS) below the k = 4 floor, and
belief-content Delta = [R2(m_token) - R2(last_token)] at STEPS minus at step 0 greater than 0.02.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text())
R, preds, out, LAST = Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]), int(sys.argv[5])
SITE = preds["site"]
Q2S = [0.25, 0.35, 0.5, 0.65, 0.8]


def rec(run, step):
    return next(r for r in recs if r["run"] == run and r["step"] == step and r["analysis"] == "analysis")


runs = {}
for name in sorted({r["run"] for r in recs}):
    f, f0 = rec(name, LAST), rec(name, 0)
    s, s0, g = f["sites"][SITE], f0["sites"][SITE], f["generator"]
    rows = [json.loads(l) for l in open(R / name / "metrics.jsonl")]
    tail = [r["kl_final"] for r in rows if r["step"] >= 0.75 * LAST]
    floor4 = f["kl_lastk_final"]["4"]
    dlt = (s["m_token"]["r2_heldout"] - s["last_token"]["r2_heldout"]) - (s0["m_token"]["r2_heldout"] - s0["last_token"]["r2_heldout"])
    pm, pu = s["m_token"]["partialled"], s["u_token"]["partialled"]
    runs[name] = dict(q2=f["q2"], seed=f["seed"], kl_tailmean=float(np.mean(tail)), floor4=floor4,
                      kl_single=f["kl_final"], delta=dlt, passing=bool(np.mean(tail) < floor4 and dlt > 0.02),
                      pred_pr=g["pr_factored"], pr_belief_m=pm["pr_C_belief"],
                      e_pr=abs(g["pr_factored"] - pm["pr_C_belief"]) / pm["pr_C_belief"],
                      top_m=pm["G_norm_eigs"][0], in_band=all(0.4 <= e <= 0.6 for e in pm["G_norm_eigs"]),
                      top_u=pu["G_norm_eigs"][0], m_closer=abs(pm["G_norm_eigs"][0] - .5) < abs(pu["G_norm_eigs"][0] - .5),
                      pr_Ch=s["C_h"]["pr"], pr_Ch_step0=s0["C_h"]["pr"])

by = defaultdict(list)
for r in runs.values():
    by[r["q2"]].append(r)
cond = {}
for q in Q2S:
    rs, ps = by.get(q, []), [r for r in by.get(q, []) if r["passing"]]
    c = dict(n=len(rs), n_passing=len(ps))
    if len(ps) >= 2:
        prs = np.array([r["pr_belief_m"] for r in ps])
        c.update(pred=ps[0]["pred_pr"], pr_mean=float(prs.mean()), pr_sd=float(prs.std(ddof=1)),
                 e_pr_mean=abs(ps[0]["pred_pr"] - prs.mean()) / prs.mean(),
                 frac_e_pr_ok=float(np.mean([r["e_pr"] < 0.1 for r in ps])),
                 frac_in_band=float(np.mean([r["in_band"] for r in ps])),
                 top_m_mean=float(np.mean([r["top_m"] for r in ps])),
                 frac_m_closer=float(np.mean([r["m_closer"] for r in ps])),
                 pr_Ch_mean=float(np.mean([r["pr_Ch"] for r in ps])), pr_Ch_sd=float(np.std([r["pr_Ch"] for r in ps], ddof=1)))
    cond[str(q)] = c

ok = {q: c for q, c in cond.items() if c["n_passing"] >= 2}
heldout = [q for q in ("0.35", "0.65") if q in ok]


def verdict(pass_fn, fail_fn, scope):
    qs = [q for q in scope if q in ok]
    if len(qs) < len(scope):
        return "inconclusive"
    if all(pass_fn(ok[q]) for q in qs):
        return "supported"
    if any(fail_fn(ok[q]) for q in qs):
        return "not supported"
    return "inconclusive"


ALL = [str(q) for q in Q2S]
P1 = verdict(lambda c: c["frac_in_band"] >= 0.8 and c["top_m_mean"] <= 0.6, lambda c: c["top_m_mean"] > 0.6 or c["frac_in_band"] < 0.5, ALL)
P2 = verdict(lambda c: c["e_pr_mean"] < 0.1 and c["frac_e_pr_ok"] >= 0.8, lambda c: c["e_pr_mean"] >= 0.1, ALL)
P2_heldout = verdict(lambda c: c["e_pr_mean"] < 0.1 and c["frac_e_pr_ok"] >= 0.8, lambda c: c["e_pr_mean"] >= 0.1, ["0.35", "0.65"])
P3 = verdict(lambda c: c["frac_m_closer"] >= 0.8, lambda c: c["frac_m_closer"] < 0.5, ["0.25", "0.35", "0.65", "0.8"])
order_pred = sorted(ok, key=lambda q: ok[q]["pred"]); order_obs = sorted(ok, key=lambda q: ok[q]["pr_mean"])
res = dict(runs=runs, conditions=cond, P1=P1, P2=P2, P2_heldout=P2_heldout, P3=P3,
           ordering=dict(predicted=order_pred, observed=order_obs, same=order_pred == order_obs))
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))

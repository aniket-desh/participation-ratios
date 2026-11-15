"""Grade Batch 2 against the criteria of EXPERIMENT.md (as amended by A1), from results.json and the runs' metrics.jsonl.

Usage: python grade.py results.json runs/transformer-discovery grade.json

Everything is at site L0.resid_post, step 20,000, n_eval 65,536 unless stated. Definitions follow EXPERIMENT.md:
  passing run   = main run with (tail-mean final-position KL < k=4 floor) [A1] and belief-content Delta > 0.02,
  Delta         = [R2(m_token) - R2(last_token)]_{20k} - [same]_{step 0}   (held-out R2 of the forward fit b -> h),
  partialled G_b from the joint fit h ~ c + E [b, onehot(x_T)],  e_PR(pred, obs) = |pred - obs| / obs.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SITE, LAST, Q2S = "L0.resid_post", 20000, (0.25, 0.5, 0.8)
recs = json.loads(Path(sys.argv[1]).read_text())
R, out = Path(sys.argv[2]), Path(sys.argv[3])


def rec(run, step, analysis="analysis"):
    return next(r for r in recs if r["run"] == run and r["step"] == step and r["analysis"] == analysis)


def tail_mean_kl(run):
    """Mean of kl_final over evaluations with step >= 15,000 (last quarter; amendment A1)."""
    rows = [json.loads(l) for l in open(R / run / "metrics.jsonl")]
    t = [r["kl_final"] for r in rows if "kl_final" in r and r["step"] >= 15000]
    return float(np.mean(t)), len(t)


def delta(run):
    d = {s: rec(run, s)["sites"][SITE] for s in (0, LAST)}
    return {s: d[s]["m_token"]["r2_heldout"] - d[s]["last_token"]["r2_heldout"] for s in d}


def e_pr(pred, obs):
    return abs(pred - obs) / obs


runs = {}
for name in sorted({r["run"] for r in recs if r["run"].startswith(("main_", "ctrl_"))}):
    f, f0 = rec(name, LAST), rec(name, 0)
    s, g = f["sites"][SITE], f["generator"]
    tm, ntail = tail_mean_kl(name)
    floor4 = f["kl_lastk_final"]["4"] if isinstance(f["kl_lastk_final"], dict) else f["kl_lastk_final"][3]
    d = delta(name)
    r = dict(q2=f["q2"], seed=f["seed"], kl_final_last=f["kl_final"], kl_final_tailmean=tm, n_tail=ntail,
             floor4=floor4, kl_uniform_final=f["kl_uniform_final"],
             bar_tail=tm < floor4, bar_single=f["kl_final"] < floor4,
             r2_mtok_minus_last={str(k): v for k, v in d.items()}, delta=d[LAST] - d[0],
             pr_Ch=s["C_h"]["pr"], pr_Ch_step0=f0["sites"][SITE]["C_h"]["pr"],
             pr_m_iso=g["pr_factored"], pr_u_iso=g["pr_u_isometric"], generator=g)
    for b, ref in (("m", "pr_m_iso"), ("u", "pr_u_iso")):
        p, p0 = s[f"{b}_token"]["partialled"], f0["sites"][SITE][f"{b}_token"]["partialled"]
        r[b] = dict(eigs=p["G_norm_eigs"], top=p["G_norm_eigs"][0], factor1_share=p["factor1_share"],
                    pr=p["pr_C_belief"], e_pr=e_pr(r[ref], p["pr_C_belief"]),
                    in_band=all(0.4 <= e <= 0.6 for e in p["G_norm_eigs"]),
                    eigs_step0=p0["G_norm_eigs"], pr_step0=p0["pr_C_belief"], e_pr_step0=e_pr(r[ref], p0["pr_C_belief"]),
                    raw_G_eigs=s[b]["G_norm_eigs"], raw_G_eigs_step0=f0["sites"][SITE][b]["G_norm_eigs"])
        r[b]["pass_H"] = r[b]["in_band"] and r[b]["e_pr"] < 0.1
    for n in (16384, 262144):  # evaluation-sample check
        try:
            r[f"pr_Ch_n{n}"] = rec(name, LAST, f"analysis_n{n}")["sites"][SITE]["C_h"]["pr"]
        except StopIteration:
            r[f"pr_Ch_n{n}"] = None
    runs[name] = r

main = {k: v for k, v in runs.items() if k.startswith("main_")}
shuf = runs.get("ctrl_shuffled_s0")
delta_shuf = shuf["delta"] if shuf else None
for v in main.values():
    v["passing_tail"] = v["bar_tail"] and v["delta"] > 0.02          # A1 rule (graded)
    v["passing_single"] = v["bar_single"] and v["delta"] > 0.02      # original rule (reported alongside)
    v["delta_exceeds_shuffled"] = None if delta_shuf is None else v["delta"] > delta_shuf


def grade_H(b, passkey):
    by_q = {q: [v for v in main.values() if v["q2"] == q and v[passkey]] for q in Q2S}
    ok_q = {q: len(by_q[q]) >= 2 for q in Q2S}
    supported = all(ok_q.values()) and all(v[b]["pass_H"] for q in Q2S for v in by_q[q])
    whole_cond_top = [q for q in Q2S if ok_q[q] and all(v[b]["top"] > 0.6 for v in by_q[q])]
    share = {q: float(np.mean([v[b]["factor1_share"] for v in by_q[q]])) if by_q[q] else None for q in Q2S}
    trend = (abs(share[0.25] - share[0.8]) if ok_q[0.25] and ok_q[0.8] else None)
    not_supported = bool(whole_cond_top) or (trend is not None and trend > 0.1)
    verdict = ("CONFLICT" if supported and not_supported else "supported" if supported
               else "not supported" if not_supported else "inconclusive")
    return dict(verdict=verdict, n_passing={str(q): len(by_q[q]) for q in Q2S},
                conditions_all_top_above_0p6=whole_cond_top, factor1_share_mean=share, share_diff_025_08=trend,
                all_passing_runs_in_band_and_epr=all(v[b]["pass_H"] for q in Q2S for v in by_q[q]))


def m_vs_u(subset):
    closer = {k: ("u" if abs(v["u"]["top"] - .5) < abs(v["m"]["top"] - .5) else "m") for k, v in subset.items()}
    nu = sum(c == "u" for c in closer.values()); n = len(closer)
    verdict = "u favored" if nu >= 5 else "m favored" if n - nu >= 5 else "undecided"
    return dict(closer=closer, n_u=nu, n=n, verdict=verdict)


def seed_consistency(passkey=None):
    res = {}
    for q in Q2S:
        vs = [v for v in main.values() if v["q2"] == q and (passkey is None or v[passkey])]
        pr = np.array([v["pr_Ch"] for v in vs]); kl = np.array([v["kl_final_tailmean"] for v in vs])
        res[str(q)] = dict(n=len(vs), pr=pr.tolist(),
                           cv_ddof1=float(pr.std(ddof=1) / pr.mean()) if len(vs) > 1 else None,
                           cv_ddof0=float(pr.std() / pr.mean()) if len(vs) > 1 else None,
                           kl_tail_ratio=float(kl.max() / kl.min()) if len(vs) > 1 else None)
        res[str(q)]["below_5pct"] = None if res[str(q)]["cv_ddof1"] is None else res[str(q)]["cv_ddof1"] < 0.05
        res[str(q)]["kl_matched_2x"] = None if res[str(q)]["kl_tail_ratio"] is None else res[str(q)]["kl_tail_ratio"] <= 2
    return res


aniso = {k: v for k, v in main.items() if v["q2"] != 0.5}
validity = {k: (None if v["pr_Ch_n262144"] is None else abs(v["pr_Ch"] - v["pr_Ch_n262144"]) / v["pr_Ch_n262144"])
            for k, v in main.items()}

perm = {}
for name in ("ctrl_perm777_q0.25_s0", "ctrl_perm777_q0.8_s0"):
    if name not in runs:
        continue
    c = runs[name]; ref = [v for v in main.values() if v["q2"] == c["q2"]]
    checks = {}
    for key, get, widen in (("top_m", lambda v: v["m"]["top"], "abs"), ("top_u", lambda v: v["u"]["top"], "abs"),
                            ("pr_Ch", lambda v: v["pr_Ch"], "rel")):
        lo, hi = min(map(get, ref)), max(map(get, ref))
        lo, hi = (lo - .05, hi + .05) if widen == "abs" else (lo * .95, hi * 1.05)
        checks[key] = dict(value=get(c), range=[lo, hi], inside=lo <= get(c) <= hi)
    perm[name] = dict(checks=checks, passes=all(x["inside"] for x in checks.values()))

h4 = defaultdict(dict)  # explained-trace fraction, seed mean and min/max, per q2, site, coordinate set, step
for step in (0, LAST):
    for q in Q2S:
        rs = [rec(k, step) for k, v in main.items() if v["q2"] == q]
        for site in rs[0]["sites"]:
            for cs in ("last_token", "m", "m_token"):
                x = [r["sites"][site][cs]["frac_explained"] for r in rs]
                h4[f"step{step}"][f"q{q}|{site}|{cs}"] = [float(np.mean(x)), min(x), max(x)]

G = dict(
    runs=runs, delta_shuffled=delta_shuf,
    H1=grade_H("m", "passing_tail"), H1u=grade_H("u", "passing_tail"),
    H1_single_final_bar=grade_H("m", "passing_single"), H1u_single_final_bar=grade_H("u", "passing_single"),
    m_vs_u_all_six=m_vs_u(aniso), m_vs_u_passing=m_vs_u({k: v for k, v in aniso.items() if v["passing_tail"]}),
    seed_consistency_all=seed_consistency(), seed_consistency_passing=seed_consistency("passing_tail"),
    measurement_validity=dict(rel_diff=validity, all_within_1pct=all(x is not None and x < .01 for x in validity.values())),
    permutation=perm, h4=h4,
)
out.write_text(json.dumps(G, indent=1))

# Console table
print(f"{'run':16s} {'tailKL':>8s} {'final':>8s} {'floor4':>8s} bar(t/s) {'Delta':>7s} pass  "
      f"{'top_m':>6s} {'sh_m':>5s} {'ePR_m':>6s}  {'top_u':>6s} {'sh_u':>5s} {'ePR_u':>6s} {'PR(Ch)':>7s}")
for k, v in runs.items():
    print(f"{k:16s} {v['kl_final_tailmean']:8.5f} {v['kl_final_last']:8.5f} {v['floor4']:8.5f} "
          f"{'Y' if v['bar_tail'] else 'n'}/{'Y' if v['bar_single'] else 'n'}      {v['delta']:7.4f} "
          f"{'Y' if v.get('passing_tail') else '-':4s}  {v['m']['top']:6.3f} {v['m']['factor1_share']:5.3f} {v['m']['e_pr']:6.3f}"
          f"  {v['u']['top']:6.3f} {v['u']['factor1_share']:5.3f} {v['u']['e_pr']:6.3f} {v['pr_Ch']:7.3f}")
print("Delta_shuffled", delta_shuf)
for k in ("H1", "H1u", "H1_single_final_bar", "H1u_single_final_bar"):
    print(k, json.dumps(G[k]))
print("m vs u (all six):", G["m_vs_u_all_six"]["verdict"], G["m_vs_u_all_six"]["closer"])
print("seed consistency:", json.dumps(G["seed_consistency_all"]))
print("validity:", json.dumps(G["measurement_validity"]))
print("permutation:", json.dumps({k: v["passes"] for k, v in perm.items()}))

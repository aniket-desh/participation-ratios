"""Apply the definition's grading rules (EXPERIMENT.md, "Success and failure criteria") to results.json.

Usage: python grade.py results.json grades.json   (prints a readable summary, writes the numbers to grades.json)

Rules, as written in the definition:
- A main run passes the prediction-quality bar if kl_final_tailmean < kl_lastk_final["4"]; a cell (arch, q2) with
  fewer than two passing seeds makes its comparisons inconclusive.
- Excess KL = kl_final_tailmean (KL(Bayes||model) at the final position, mean over the last quarter of evaluations).
  Ratio = larger / smaller of the two cells' seed means over passing runs; > 2 labels a difference "confounded".
- Belief metric: factor-1 share of the token-partialled normalized E_b^T E_b, in m and in u, read only where
  R2(m_token) - R2(last_token) >= 0.01 (the definition's gate, applied to both m and u). Per condition, the
  coordinate (m or u) with the larger |Delta| decides. Threshold 0.1 absolute.
- PR(C_h): threshold 5% relative to the reference cell's seed mean (the second-named cell, e.g. base in A1).
- Delta = seed mean(cell A) - seed mean(cell B) over passing runs at step 20,000; SD = larger of the two cells'
  sample SDs (ddof = 1). Supported if in >= 1 condition |Delta| > thr, |Delta| > 2 SD, |Delta| > |Delta at step 0|
  (met when the step-0 value is undefined under the gate), and KL ratio <= 2. Not supported if in both conditions
  |Delta| < thr and 2 SD < thr. Otherwise inconclusive. A supported-by-size difference with ratio > 2 is reported as
  "confounded with prediction quality".
"""
import json
import sys
from pathlib import Path

import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text())
GATE, THR_F1, THR_PR = 0.01, 0.1, 0.05
Q2 = [0.25, 0.8]
main = [r for r in recs if "_q" in r["run"] and r["analysis"] == "analysis"]
STEP = max(r["step"] for r in main)


def rec(arch, q2, seed, step):
    m = [r for r in main if r["arch"] == arch and r["q2"] == q2 and r["seed"] == seed and r["step"] == step]
    return m[0] if m else None


def passes(r):
    return r["kl_final_tailmean"] < r["kl_lastk_final"]["4"]


def gain(s):
    return s["m_token"]["r2_heldout"] - s["last_token"]["r2_heldout"]


def f1(s, cs):  # gated factor-1 share; nan where beliefs add < 0.01 R2 beyond the final token
    return s[cs]["partialled"]["factor1_share"] if gain(s) >= GATE else np.nan


QTY = {"pr": lambda s: s["C_h"]["pr"], "f1_m": lambda s: f1(s, "m_token"), "f1_u": lambda s: f1(s, "u_token")}

# --- per-run table and cell pass counts
runs, cells = [], {}
for arch in ["base", "nomlp", "preln"]:
    for q2 in Q2:
        ok = []
        for seed in [0, 1, 2]:
            r = rec(arch, q2, seed, STEP)
            if r is None:
                continue
            p = passes(r)
            ok.append(seed) if p else None
            runs.append(dict(arch=arch, q2=q2, seed=seed, lr=r["lr"], kl_final_tailmean=r["kl_final_tailmean"],
                             kl_final_last=r["kl_final"], floors={k: r["kl_lastk_final"][k] for k in "1234"}, passes=p,
                             sites={s: dict(pr=v["C_h"]["pr"], gain=gain(v), f1_m=f1(v, "m_token"), f1_u=f1(v, "u_token"))
                                    for s, v in r["sites"].items()}))
        cells[(arch, q2)] = ok


def vals(arch, site, q2, step, qty, seeds):
    out = []
    for sd in seeds:
        r = rec(arch, q2, sd, step)
        out.append(QTY[qty](r["sites"][site]) if r is not None and site in r["sites"] else np.nan)
    return np.array(out, float)


def stats(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan, len(x)


def kl_mean(arch, q2):
    return float(np.mean([next(r["kl_final_tailmean"] for r in runs if r["arch"] == arch and r["q2"] == q2 and r["seed"] == s)
                          for s in cells[(arch, q2)]])) if cells[(arch, q2)] else np.nan


def compare(a, b, qty, q2):
    """a, b = (arch, site). Returns the per-condition numbers for one quantity."""
    sa, sb = cells[(a[0], q2)], cells[(b[0], q2)]
    ma, sda, na = stats(vals(*a, q2, STEP, qty, sa)); mb, sdb, nb = stats(vals(*b, q2, STEP, qty, sb))
    m0a, _, _ = stats(vals(*a, q2, 0, qty, sa)); m0b, _, _ = stats(vals(*b, q2, 0, qty, sb))
    d, d0 = ma - mb, m0a - m0b
    sd = np.nanmax([sda, sdb]) if np.isfinite([sda, sdb]).any() else np.nan
    thr = THR_PR * mb if qty == "pr" else THR_F1
    # paired per-seed differences (seeds passing in both cells)
    both = sorted(set(sa) & set(sb))
    paired = (vals(*a, q2, STEP, qty, both) - vals(*b, q2, STEP, qty, both)).tolist()
    return dict(qty=qty, q2=q2, mean_a=ma, mean_b=mb, n_a=na, n_b=nb, sd=sd, delta=d, delta_step0=d0, thr=thr,
                paired=dict(zip(both, paired)))


def verdict(a, b, metric):
    per = []
    for q2 in Q2:
        ka, kb = kl_mean(a[0], q2), kl_mean(b[0], q2)
        ratio = max(ka, kb) / min(ka, kb) if min(ka, kb) > 0 else np.nan
        enough = len(cells[(a[0], q2)]) >= 2 and len(cells[(b[0], q2)]) >= 2
        if metric == "pr":
            c = compare(a, b, "pr", q2); alt = None
        else:  # the larger |Delta| of m and u decides
            cm, cu = compare(a, b, "f1_m", q2), compare(a, b, "f1_u", q2)
            fin = [c for c in (cm, cu) if np.isfinite(c["delta"])]
            c = max(fin, key=lambda c: abs(c["delta"])) if fin else cm
            alt = cu if c is cm else cm
        d, sd, thr, d0 = c["delta"], c["sd"], c["thr"], c["delta_step0"]
        defined = np.isfinite(d) and np.isfinite(sd) and enough
        size = defined and abs(d) > thr and abs(d) > 2 * sd and (not np.isfinite(d0) or abs(d) > abs(d0))
        null = defined and abs(d) < thr and 2 * sd < thr
        per.append(dict(c, kl_a=ka, kl_b=kb, kl_ratio=ratio, enough_passing=enough, size_ok=size, null_ok=null,
                        other_coord=alt))
    if any(p["size_ok"] and p["kl_ratio"] <= 2 for p in per):
        v = "supported"
    elif any(p["size_ok"] for p in per):
        v = "confounded with prediction quality"
    elif all(p["null_ok"] for p in per):
        v = "not supported"
    else:
        v = "inconclusive"
    return dict(verdict=v, per_condition=per)


COMPS = {"A1 nomlp vs base @resid_post": (("nomlp", "L0.resid_post"), ("base", "L0.resid_post")),
         "A2 preln vs base @resid_post": (("preln", "L0.resid_post"), ("base", "L0.resid_post")),
         "A2 preln ln2 vs resid_mid": (("preln", "L0.ln2"), ("preln", "L0.resid_mid")),
         "A3 base resid_post vs resid_mid": (("base", "L0.resid_post"), ("base", "L0.resid_mid"))}
grades = {name: {m: verdict(a, b, m) for m in ["belief", "pr"]} for name, (a, b) in COMPS.items()}

# --- within-pilot lr effect (pilots analyzed at steps 0 and 20,000), at resid_post
pil = {}
for r in recs:
    if "_pilot_" in r["run"] and r["step"] == STEP:
        s = r["sites"]["L0.resid_post"]
        pil.setdefault(r["arch"], {})[r["lr"]] = dict(pr=s["C_h"]["pr"], gain=gain(s), f1_m=f1(s, "m_token"),
                                                      f1_u=f1(s, "u_token"), kl_final=r["kl_final"], kl=r["kl"])
lrs = {}
for a in ["base", "nomlp", "preln"]:
    p = Path(sys.argv[1]).parent / f"lr_{a}.json"
    lrs[a] = json.loads(p.read_text())["lr"] if p.exists() else None

out = dict(step=STEP, selected_lr=lrs, cells_passing={f"{a}_q{q}": s for (a, q), s in cells.items()}, runs=runs,
           grades=grades, pilots=pil)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))

# --- readable summary
f = lambda x: "  nan" if x is None or not np.isfinite(x) else f"{x:.4g}"
print("selected lr:", lrs)
for r in runs:
    print(f"{r['arch']:5s} q2={r['q2']:<4} s{r['seed']} lr={r['lr']:g} tailKL={f(r['kl_final_tailmean'])} "
          f"floor4={f(r['floors']['4'])} pass={r['passes']}  resid_post PR={f(r['sites']['L0.resid_post']['pr'])} "
          f"gain={f(r['sites']['L0.resid_post']['gain'])} f1m={f(r['sites']['L0.resid_post']['f1_m'])} "
          f"f1u={f(r['sites']['L0.resid_post']['f1_u'])}")
for name, g in grades.items():
    for m, v in g.items():
        print(f"\n{name} [{m}]: {v['verdict']}")
        for p in v["per_condition"]:
            print(f"  q2={p['q2']} {p['qty']}: A={f(p['mean_a'])} (n{p['n_a']}) B={f(p['mean_b'])} (n{p['n_b']}) "
                  f"D={f(p['delta'])} SD={f(p['sd'])} thr={f(p['thr'])} D0={f(p['delta_step0'])} "
                  f"KLratio={f(p['kl_ratio'])} paired={ {k: round(x, 4) for k, x in p['paired'].items()} }")
            if p["other_coord"] is not None:
                o = p["other_coord"]
                print(f"     (other coord {o['qty']}: D={f(o['delta'])} SD={f(o['sd'])} D0={f(o['delta_step0'])})")
print("\npilots @resid_post step", STEP)
for a, d in pil.items():
    for lr, v in sorted(d.items()):
        print(f"  {a} lr={lr:g}: " + " ".join(f"{k}={f(x)}" for k, x in v.items()))

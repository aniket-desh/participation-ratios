"""Package 5 numerical study (derivations §5.8-5.13 and "what to implement next" item 5). CPU only.

    uv run python experiments/heads-continuum/run.py [stage ...]     stages: onset weak quad finite sweep controls

Restricted many-head model of derivations §5.1 on exact T=3 populations (64 contexts). Continuum predictions use
tensor Gauss–Legendre cubature over the initial law (beta, kappa) ~ U([-1,1] x [-1/2,1/2]), a = 0, g = 1; finite
networks draw H iid heads from the same law. Nothing is fitted: the continuum is computed before and independently
of the finite networks. Each stage writes results/<stage>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from prt import heads as hd

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PRIMARY = (0.7, 0.4, 0.8, 0.8)          # the derivations' numerical check generator
TAU, DT, REC = 100.0, 0.2, 25           # horizon, RK4 step, record every REC steps (tau = 5)


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj))
    print(f"[{time.strftime('%H:%M:%S')}] wrote {name}", flush=True)


def traj(pop, th, w, tau=TAU, dt=DT, rec=REC, mask=None):
    _, r = hd.integrate(th, w, pop, dt=dt, steps=int(round(tau / dt)), record_every=rec, mask=mask)
    return r


def summary(r):
    S = r["S"]
    return dict(tau=r["tau"].tolist(), loss=r["loss"].tolist(), pr=[hd.pr_S(s) for s in S],
                eigs=[np.sort(np.linalg.eigvalsh(s))[::-1][:6].tolist() for s in S],
                theta_mean=r["theta_mean"].tolist())


stages = sys.argv[1:] or ["onset", "weak", "quad", "finite", "sweep", "controls"]
pop = hd.Population.independent(*PRIMARY, 3)

if "onset" in stages:   # zero-time coefficient table (derivations' numerical check)
    save("onset", [dict(n=n, **{k: v for k, v in hd.onset(*hd.quadrature_init(n), pop).items() if k in ("pr0", "prdd0")})
                   for n in [5, 9, 17, 33]])

if "weak" in stages:    # weak-signal scaling: q_j = eps * qbar_j => A = O(eps^2), B, K'' = O(eps^4) (derivations §5.9)
    rows = []
    for eps in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        pe = hd.Population.independent(0.7, 0.4 * eps, 0.8, 0.8 * eps, 3)
        o = hd.onset(*hd.quadrature_init(17), pe)
        rows.append(dict(eps=eps, A=float(np.abs(o["A"]).max()), B=float(np.abs(o["B"]).max()),
                         Kdd=float(np.abs(o["Kdd"]).max()), pr0=o["pr0"], prdd0=o["prdd0"]))
    save("weak", rows)

if "quad" in stages:    # continuum: cubature-order and step-size convergence
    out = {}
    for n, dt in [(12, 0.2), (17, 0.2), (24, 0.2), (17, 0.1)]:
        out[f"n{n}_dt{dt}"] = summary(traj(pop, *hd.quadrature_init(n), dt=dt, rec=int(round(REC * DT / dt))))
    save("quad", out)

if "finite" in stages:  # finite-H networks vs the continuum: RMS ||S_H - S_nu||_F and |PR_H - PR_nu|
    ref = traj(pop, *hd.quadrature_init(24))
    out = dict(tau=ref["tau"].tolist(), pr_ref=[hd.pr_S(s) for s in ref["S"]], runs={})
    for H, seeds in [(16, 16), (64, 16), (256, 8), (1024, 8)]:
        rr = []
        for seed in range(seeds):
            r = traj(pop, *hd.random_init(H, np.random.default_rng(1000 * H + seed)))
            rr.append(dict(seed=seed, fro_err=[float(np.linalg.norm(a - b)) for a, b in zip(r["S"], ref["S"])],
                           pr=[hd.pr_S(s) for s in r["S"]], loss=r["loss"].tolist()))
        out["runs"][str(H)] = rr
        save("finite", out)

if "sweep" in stages:   # generator anisotropy at the main-sweep reliabilities (T=3 so the population is exact)
    out = {}
    for q2 in [0.25, 0.5, 0.8]:
        pg = hd.Population.independent(0.8, 0.5, 0.8, q2, 3)
        o = hd.onset(*hd.quadrature_init(17), pg)
        out[str(q2)] = dict(onset=dict(pr0=o["pr0"], prdd0=o["prdd0"]),
                            traj=summary(traj(pg, *hd.quadrature_init(17), tau=300.0, rec=50)))
        save("sweep", out)

if "controls" in stages:  # frozen features (readout only) and frozen attention (a, g train) at the primary generator
    out = {}
    for name, mask in [("readout_only", np.array([1, 1, 0, 0, 0.])), ("frozen_attention", np.array([1, 1, 0, 0, 1.])),
                       ("full", None)]:
        out[name] = summary(traj(pop, *hd.quadrature_init(17), tau=300.0, rec=50, mask=mask))
    save("controls", out)

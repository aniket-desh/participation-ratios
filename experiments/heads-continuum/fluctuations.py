"""Initialization fluctuations of PR (derivations §5.12-5.13). CPU only.

    uv run python experiments/heads-continuum/fluctuations.py

At tau = 0 heads are iid, so S_H - S = (1/H) sum_i (X_i - E X) with X = P^1/2 Xi(theta) P^1/2. Linearizing
R(S) = (tr S)^2 / tr(S^2) gives PR_H - PR ≈ DR_S[S_H - S], hence
    SD(PR_H) ≈ H^{-1/2} SD_{theta ~ nu0}( DR_S[X(theta)] ),
DR_S[E] = 2 tr S tr E / tr S^2 - 2 (tr S)^2 tr(S E) / (tr S^2)^2. The SD over the initial law is computed with
48x48 Gauss–Legendre cubature and compared with the seed spread of the finite networks in results/finite.json.
"""
import json
from pathlib import Path

import numpy as np

from prt import heads as hd

D = Path(__file__).parent / "results"
pop = hd.Population.independent(0.7, 0.4, 0.8, 0.8, 3)
th, w = hd.quadrature_init(48)
psi, _ = hd.features(th, pop)
psit = psi - np.einsum("m,qmj->qj", pop.p, psi)[:, None, :]
sp = np.sqrt(pop.p)
X = 0.5 * np.einsum("qmj,qnj->qmn", psit, psit) * sp[None, :, None] * sp[None, None, :]   # per-head P^1/2 Xi P^1/2
S = np.einsum("q,qmn->mn", w, X)
A, B = np.trace(S), np.sum(S * S)
dr = 2 * A * np.einsum("qmm->q", X) / B - 2 * A**2 * np.einsum("mn,qmn->q", S, X) / B**2
sd_unit = float(np.sqrt(w @ (dr - w @ dr) ** 2))
f = json.loads((D / "finite.json").read_text())
rows = []
for H, rr in f["runs"].items():
    prs = np.array([r["pr"][0] for r in rr])
    rows.append(dict(H=int(H), n_seeds=len(prs), sd_observed=float(prs.std(ddof=1)), sd_predicted=sd_unit / np.sqrt(int(H)),
                     ratio=float(prs.std(ddof=1) / (sd_unit / np.sqrt(int(H)))),
                     mean_minus_pr0=float(prs.mean() - f["pr_ref"][0])))
    print(rows[-1])
(D / "fluctuations.json").write_text(json.dumps(dict(sd_unit=sd_unit, pr0=float(A**2 / B), rows=rows), indent=1))

# Direct check with many initializations (no training is needed at tau = 0): 2,000 iid networks per H.
big = []
for H in [16, 64, 256, 1024]:
    rng = np.random.default_rng(12345 + H)
    prs = []
    for _ in range(2000):
        t, wh = hd.random_init(H, rng)
        prs.append(hd.pr_S(hd.kernel(t, wh, pop)[1]))
    prs = np.array(prs)
    se_sd = prs.std(ddof=1) / np.sqrt(2 * (len(prs) - 1))
    big.append(dict(H=H, n=len(prs), sd_observed=float(prs.std(ddof=1)), sd_se=float(se_sd),
                    sd_predicted=sd_unit / np.sqrt(H), ratio=float(prs.std(ddof=1) / (sd_unit / np.sqrt(H))),
                    mean_minus_pr0=float(prs.mean() - A**2 / B), skew=float(((prs - prs.mean()) ** 3).mean() / prs.std() ** 3)))
    print(big[-1], flush=True)
(D / "fluctuations.json").write_text(json.dumps(dict(sd_unit=sd_unit, pr0=float(A**2 / B), rows=rows, many_inits=big), indent=1))

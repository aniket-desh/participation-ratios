"""Stage C learned positional filter (experiment plan §7.2; derivations §3.3). CPU only.

    uv run python experiments/stagec-learned-filter/run.py

Model: z(c; beta) = sum_k softmax_k(-beta k) phi~(x_{T-k}) with fixed token features phi(x) = (x1, x2) (both factor
signs; the value/output geometry is fixed), readout f = w^T z. Target: the next sign of factor 1.
Prediction: integrate the exact gradient-flow ODE of derivations §3.3 using exact context sums at T = 6 with
y(c) = E[X_{T+1,1} | c] = rho1 q1 m1(c), and predict beta(tau), w(tau) and the statistic's covariance C_z(beta(tau)).
Test: minibatch SGD on independently sampled contexts with the actual next sign as target (same expected gradient),
lr eta, batch 256, time tau = eta * steps. Nothing from the SGD runs enters the prediction.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prt.attention_filter import filter_ode, lagged_features, statistic
from prt.generators import PAIRS, two_factor_hmm, token_pairs
from prt.spectra import pr, weighted_cov

OUT = Path(__file__).parent
T, RHO1, RHO2, Q2 = 6, 0.8, 0.5, 0.5
DK = -np.arange(T, dtype=float)
W0, BETA0, ETA, TAU_END = np.zeros(2), 0.0, 0.05, 400.0
res = {}
for q1 in [0.1, 0.25, 0.5, 0.8]:
    h = two_factor_hmm(RHO1, q1, RHO2, Q2, perm=[2, 0, 3, 1])
    toks, p, b = h.enumerate(T)
    phi = token_pairs(h).astype(float)            # stationary mean is zero by sign symmetry
    Xk = lagged_features(toks, phi)
    y = RHO1 * q1 * (b @ PAIRS[:, 0])
    ode = filter_ode(Xk, p, y, DK, W0, BETA0, dt=0.5, steps=int(TAU_END / 0.5), record_every=8)
    pr_pred = [pr(C) for C in ode["Cz"]]
    sgd_runs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        w, beta = W0.copy(), BETA0
        steps = int(TAU_END / ETA)
        rec = dict(tau=[], beta=[], w=[], pr=[])
        eval_t = h.sample(20_000, T, np.random.default_rng(999))
        eval_X = lagged_features(eval_t, phi)
        for n in range(steps + 1):
            if n % int(25 / ETA) == 0:
                z, _ = statistic(eval_X, beta, DK)
                rec["tau"].append(n * ETA); rec["beta"].append(beta); rec["w"].append(w.tolist())
                rec["pr"].append(pr(weighted_cov(z)[1]))
            if n == steps:
                break
            seq = h.sample(256, T + 1, rng)
            X = lagged_features(seq[:, :T], phi)
            target = phi[seq[:, T], 0]              # actual next factor-1 sign
            z, zb = statistic(X, beta, DK)
            e = z @ w - target
            w, beta = w - ETA * (e @ z) / 256, beta - ETA * (e @ (zb @ w)) / 256
        sgd_runs.append(rec)
    res[str(q1)] = dict(ode=dict(tau=ode["tau"].tolist(), beta=ode["beta"].tolist(), w=ode["w"].tolist(),
                                 loss=ode["loss"].tolist(), pr=pr_pred),
                        sgd=sgd_runs, beta_weak_signal=float(-np.log(RHO1)))
    fb = [r["beta"][-1] for r in sgd_runs]; fp = [r["pr"][-1] for r in sgd_runs]
    print(f"q1={q1}: ODE beta={ode['beta'][-1]:.4f} PR={pr_pred[-1]:.4f} | SGD beta={np.mean(fb):.4f}±{np.std(fb):.4f} "
          f"PR={np.mean(fp):.4f}±{np.std(fp):.4f} | weak-signal beta=-log rho1={-np.log(RHO1):.4f}", flush=True)
(OUT / "results.json").write_text(json.dumps(res))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, (q1, r) in enumerate(res.items()):
    c = f"C{i}"
    ax[0].plot(r["ode"]["tau"], r["ode"]["beta"], c, lw=3, alpha=.35, label=f"q1={q1} ODE")
    ax[1].plot(r["ode"]["tau"], r["ode"]["pr"], c, lw=3, alpha=.35)
    for s in r["sgd"]:
        ax[0].plot(s["tau"], s["beta"], c + "--", lw=1); ax[1].plot(s["tau"], s["pr"], c + "--", lw=1)
ax[0].axhline(-np.log(RHO1), c="k", ls=":", label="-log rho1"); ax[0].legend(fontsize=7)
ax[0].set_title("attention decay beta (thick: exact ODE, dashed: SGD seeds)"); ax[1].set_title("PR(C_z) of the attention statistic")
for a in ax: a.set_xlabel("tau = lr x steps")
fig.tight_layout(); fig.savefig(OUT / "filter.png", dpi=130)

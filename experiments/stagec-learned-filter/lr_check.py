"""Follow-up: is SGD's beta excess over the exact ODE at low reliability a finite-learning-rate effect?

Reruns the SGD arm of run.py at q1 in {0.1, 0.25} with lr 0.05 and 0.025 (same tau = 400, same 3 seeds) and writes
lr_check.json. If the excess shrinks roughly in proportion to lr, it is a discretization effect of SGD.
"""
import json
from pathlib import Path

import numpy as np

from prt.attention_filter import lagged_features, statistic
from prt.generators import two_factor_hmm, token_pairs

T, RHO1, RHO2, Q2, TAU_END = 6, 0.8, 0.5, 0.5, 400.0
DK = -np.arange(T, dtype=float)
ode = json.loads((Path(__file__).parent / "results.json").read_text())
out = {}
for q1 in [0.1, 0.25]:
    h = two_factor_hmm(RHO1, q1, RHO2, Q2, perm=[2, 0, 3, 1])
    phi = token_pairs(h).astype(float)
    for eta in [0.05, 0.025]:
        betas = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            w, beta = np.zeros(2), 0.0
            for n in range(int(TAU_END / eta)):
                seq = h.sample(256, T + 1, rng)
                X = lagged_features(seq[:, :T], phi)
                z, zb = statistic(X, beta, DK)
                e = z @ w - phi[seq[:, T], 0]
                w, beta = w - eta * (e @ z) / 256, beta - eta * (e @ (zb @ w)) / 256
            betas.append(beta)
        b_ode = ode[str(q1)]["ode"]["beta"][-1]
        out[f"q1={q1},lr={eta}"] = dict(sgd_beta=betas, mean=float(np.mean(betas)), sd=float(np.std(betas, ddof=1)),
                                         ode_beta=b_ode, rel_excess=float(np.mean(betas) / b_ode - 1))
        print(f"q1={q1} lr={eta}: SGD beta {np.mean(betas):.4f} ± {np.std(betas, ddof=1):.4f} vs ODE {b_ode:.4f} "
              f"(excess {100 * (np.mean(betas) / b_ode - 1):.1f}%)", flush=True)
(Path(__file__).parent / "lr_check.json").write_text(json.dumps(out, indent=1))

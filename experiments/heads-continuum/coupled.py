"""Package 5.14 on coupled generators (plan §4.2, H5 in the restricted model). CPU only.

    uv run python experiments/heads-continuum/coupled.py

Continuum (17x17 cubature) of the restricted many-head model at T = 3 on the coupled two-factor generator
P_eps = (1-eps) P1⊗P2 + eps P_corr (c = 0.8), rho = 0.8, q1 = q2 = 0.5, eps in {0, 0.05, 0.2}. Two output families:
factorized (logits y1 f1 + y2 f2) and with the interaction logit y1 y2 f12 (b(0) = 0). Reports the loss of each
against the exact population Bayes cross-entropy, the learned mean interaction amplitude, and PR of the residual
kernel S. The residual representation is psi in both models; the interaction acts through training of the shared
head parameters, never by adding chi to the measured residual (derivations §5.14).
"""
import json
from pathlib import Path

import numpy as np

from prt import heads as hd
from prt.generators import two_factor_hmm

OUT = Path(__file__).parent / "results"
TAU, DT, REC = 300.0, 0.2, 50
res = {}
for eps in [0.0, 0.05, 0.2]:
    hmm = two_factor_hmm(0.8, 0.5, 0.8, 0.5, eps=eps, c=0.8)
    pop, mu12 = hd.population_from_hmm(hmm, 3)
    toks, p, b = hmm.enumerate(3)
    P = hmm.next_token_probs(b)
    bayes_ce = float(-p @ np.sum(P * np.log(P), 1))       # exact conditional entropy of the next token
    th, w = hd.quadrature_init(17)
    steps = int(TAU / DT)
    _, rf = hd.integrate(th, w, pop, dt=DT, steps=steps, record_every=REC)
    # For the factorized family the exact expected cross-entropy is sum_j[log 2cosh f_j - mu_j f_j]: the Y1 Y2
    # moment does not enter, so both losses are directly comparable with the Bayes cross-entropy.
    _, ri = hd.integrate_int(hd.with_interaction(th), w, pop, mu12, dt=DT, steps=steps, record_every=REC)
    res[str(eps)] = dict(
        bayes_ce=bayes_ce, tau=rf["tau"].tolist(),
        factorized=dict(loss=rf["loss"].tolist(), pr=[hd.pr_S(s) for s in rf["S"]]),
        interaction=dict(loss=ri["loss"].tolist(), pr=[hd.pr_S(s) for s in ri["S"]],
                         b_mean=[float(t[5]) for t in ri["theta_mean"]]),
        mean_abs_mu12_minus_mu1mu2=float(p @ np.abs(mu12 - pop.mu[:, 0] * pop.mu[:, 1])))
    r = res[str(eps)]
    print(f"eps={eps}: Bayes CE {bayes_ce:.6f} | factorized loss {r['factorized']['loss'][-1]:.6f} PR {r['factorized']['pr'][-1]:.5f}"
          f" | interaction loss {r['interaction']['loss'][-1]:.6f} PR {r['interaction']['pr'][-1]:.5f} b_mean {r['interaction']['b_mean'][-1]:.4f}", flush=True)
(OUT / "coupled.json").write_text(json.dumps(res))

"""Batch 0 / Gate A (experiment plan §5, §7.1, §4.2): generator, filter and covariance-pipeline validation. CPU only.

    uv run python experiments/batch0-generator-validation/run.py

Writes results.json next to this file. Every number is a generator or prescribed-encoding calculation; no training.
"""
import json
from pathlib import Path

import numpy as np

from prt.attention_filter import imposed_cov, lagged_features, softmax_head_gain, weak_signal_variances
from prt.generators import PAIRS, belief_coordinates, factor_signs, magnetization_filter, token_pairs, two_factor_hmm
from prt.spectra import eigvals_psd, pr, summarize, weighted_cov

OUT = Path(__file__).parent
res = {}
PERM = [2, 0, 3, 1]

# 1. Exact T=6 benchmark, including the reserved held-out reliabilities 0.35 and 0.65 (locked here).
rows = []
for q2 in [0.25, 0.35, 0.5, 0.65, 0.8]:
    h = two_factor_hmm(0.8, 0.5, 0.8, q2, perm=PERM)
    toks, p, b = h.enumerate(6)
    c = belief_coordinates(h, b)
    _, Cm = weighted_cov(c["m"], p); _, Cw = weighted_cov(c["walsh"], p); _, Cu = weighted_cov(c["u"], p)
    _, CB = weighted_cov(b, p)
    xs = factor_signs(toks, h)
    m1, lp1 = magnetization_filter(xs[..., 0], 0.8, 0.5); m2, lp2 = magnetization_filter(xs[..., 1], 0.8, q2)
    v1, v2 = Cm[0, 0], Cm[1, 1]
    lamB = eigvals_psd(CB)
    rows.append(dict(q2=q2, v1=v1, v2=v2, pr_factored=pr(Cm), pr_joint=pr(Cw), pr_u=pr(Cu),
                     walsh_eig_err=float(np.abs(np.sort(lamB[:3]) - np.sort([v1 / 4, v2 / 4, v1 * v2 / 4])).max()),
                     filter_disagreement=float(max(np.abs(m1 - c["m"][:, 0]).max(), np.abs(m2 - c["m"][:, 1]).max())),
                     prob_disagreement=float(np.abs(np.exp(lp1 + lp2) - p).max()), sum_p_minus_1=float(p.sum() - 1),
                     mean_belief_minus_pi=float(np.abs(p @ b - h.pi).max()), heldout=q2 in (0.35, 0.65)))
res["exact_T6"] = rows

# 2. Exact vs sampled moments at T=6 with increasing N; z = (sample - exact) / bootstrap-free analytic SE.
h = two_factor_hmm(0.8, 0.5, 0.8, 0.25, perm=PERM)
toks, p, b = h.enumerate(6)
m_ex = belief_coordinates(h, b)["m"]
_, C_ex = weighted_cov(m_ex, p)
conv = []
rng = np.random.default_rng(0)
for N in [10_000, 100_000, 1_000_000]:
    m = belief_coordinates(h, h.filter(h.sample(N, 6, rng)))["m"]
    mm = m - m.mean(0)
    C = mm.T @ mm / N
    # SE of each second moment from the per-sample variance of the product (independent contexts).
    se = np.array([[np.std(mm[:, i] * mm[:, j]) / np.sqrt(N) for j in range(2)] for i in range(2)])
    conv.append(dict(N=N, C=C.tolist(), z=((C - C_ex) / se).tolist(), pr=pr(C), pr_exact=pr(C_ex)))
res["sampled_vs_exact_T6"] = conv

# 3. T=16 references by Monte Carlo (Batch 2 context length), with batch-means standard errors.
t16 = []
for q2 in [0.25, 0.35, 0.5, 0.65, 0.8]:
    h = two_factor_hmm(0.8, 0.5, 0.8, q2, perm=PERM)
    vals = []
    for k in range(10):
        m = belief_coordinates(h, h.filter(h.sample(200_000, 16, np.random.default_rng(100 + k))))["m"]
        _, Cm = weighted_cov(m)
        vals.append([Cm[0, 0], Cm[1, 1], pr(Cm)])
    vals = np.array(vals)
    t16.append(dict(q2=q2, v1=vals[:, 0].mean(), v2=vals[:, 1].mean(), pr_factored=vals[:, 2].mean(),
                    se=(vals.std(0, ddof=1) / np.sqrt(10)).tolist(), N=2_000_000))
res["mc_T16"] = t16

# 4. Prescribed encodings applied to exact T=6 beliefs (q2 = 0.25): isometry, stretch, rotation, embedding, nonlinear.
h = two_factor_hmm(0.8, 0.5, 0.8, 0.25, perm=PERM)
toks, p, b = h.enumerate(6)
m = belief_coordinates(h, b)["m"]
_, Sb = weighted_cov(m, p)
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((64, 2)))
R, _ = np.linalg.qr(rng.standard_normal((2, 2)))
enc = {}
for name, E in dict(isometry_d64=Q, stretch_diag_3_1=np.diag([3.0, 1.0]), rotation=R, stretch_embed_d64=Q @ np.diag([3.0, 1.0])).items():
    _, C = weighted_cov(m @ E.T, p)
    s = summarize(C)
    enc[name] = dict(pr=s["pr"], npr=s["npr"], rank=int(np.sum(s["eigs"] > 0)),
                     max_err_vs_ESigmaET=float(np.abs(C - E @ Sb @ E.T).max()))
_, Cn = weighted_cov(np.c_[m, m[:, 0] * m[:, 1]], p)
enc["nonlinear_m1_m2_m1m2"] = dict(pr=pr(Cn), rank=int(np.sum(eigvals_psd(Cn) > 0)))
enc["belief_pr"] = pr(Sb)
res["prescribed_encodings"] = enc

# 5. Imposed positional filters (plan §7.1): exact covariance identity vs enumeration, with temporal correlations.
T = 6
phi = token_pairs(h).astype(float)
Gam = h.feature_lag_cov(phi, T)
Xk = lagged_features(toks, phi)
filt = {}
A = rng.standard_normal((8, 2))  # fixed output geometry (imposed, not learned)
for name, a in dict(exponential_rho0p8=0.8 ** np.arange(T) / np.sum(0.8 ** np.arange(T)),
                    uniform_last3=np.r_[np.ones(3) / 3, np.zeros(T - 3)]).items():
    B = a[:, None, None] * A[None]
    _, C_enum = weighted_cov(np.einsum("kdr,mkr->md", B, Xk), p)
    C_th = imposed_cov(B, Gam)
    C_indep = imposed_cov(B, Gam * np.eye(T)[:, :, None, None])
    filt[name] = dict(max_err=float(np.abs(C_th - C_enum).max()), pr_theory=pr(C_th), pr_enum=pr(C_enum),
                      pr_independent_token_approx=pr(C_indep), rank_imposed=int(np.sum(eigvals_psd(C_th) > 0)))
res["imposed_filter"] = filt

# 6. Weak-signal benchmark (plan §7.1): exact posterior variance vs linear filter vs leading formula.
ws = []
for rho in [0.5, 0.8]:
    for q in [0.05, 0.1, 0.2]:
        v = weak_signal_variances(rho, q, 16 if rho == 0.5 else 12)
        wts, gain = softmax_head_gain(rho, q, 16 if rho == 0.5 else 12)
        ws.append(dict(rho=rho, q=q, T=16 if rho == 0.5 else 12, **v, head_gain=gain,
                       rel_err_ws=abs(v["v_ws"] / v["v_exact"] - 1), rel_err_lin=abs(v["v_lin"] / v["v_exact"] - 1)))
res["weak_signal"] = ws

# 7. Coupled generators (plan §4.2): stationary law, interaction coordinate and the factorized-predictor loss gap.
coup = []
for eps in [0.0, 0.05, 0.2]:
    hc = two_factor_hmm(0.8, 0.5, 0.8, 0.5, eps=eps, c=0.8, perm=PERM)
    toks, p, b = hc.enumerate(6)
    co = belief_coordinates(hc, b)
    P = hc.next_token_probs(b)                       # (M, 4) by token label
    pairs = token_pairs(hc)
    p1 = np.stack([P[:, pairs[:, 0] == s].sum(1) for s in (-1, 1)], 1)
    p2 = np.stack([P[:, pairs[:, 1] == s].sum(1) for s in (-1, 1)], 1)
    Pf = p1[:, (pairs[:, 0] + 1) // 2] * p2[:, (pairs[:, 1] + 1) // 2]   # product of next-token marginals
    gap = float(p @ np.sum(P * np.log(P / Pf), 1))   # E_c KL(P || P1 P2), nats per token
    _, Cw = weighted_cov(co["walsh"], p)
    coup.append(dict(eps=eps, pi=hc.pi.tolist(), var_interaction=float(p @ (co["interaction"] - p @ co["interaction"]) ** 2),
                     loss_gap_factorized_nats=gap, pr_joint=pr(Cw), corr_m1_m2=float(weighted_cov(co["m"], p)[1][0, 1])))
res["coupled"] = coup


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


(OUT / "results.json").write_text(json.dumps(clean(res), indent=1))
print(json.dumps(clean(res), indent=1)[:6000])

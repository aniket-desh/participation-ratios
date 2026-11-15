"""Gate A checks for the generator and belief statistics (experiment plan §5; derivations Package 1)."""
import numpy as np
import pytest

from prt.generators import (PAIRS, belief_coordinates, factor_signs, magnetization_filter, two_factor_hmm)
from prt.spectra import eigvals_psd, pr, weighted_cov

# Experiment plan §3.4, exact enumeration at T=6 with (rho1,q1)=(0.8,0.5), rho2=0.8.
TABLE = {0.25: (0.414752, 0.139888, 1.605662, 1.925329),
         0.50: (0.414752, 0.414752, 2.000000, 2.684611),
         0.80: (0.414752, 0.736687, 1.854991, 2.626951)}


@pytest.mark.parametrize("q2", sorted(TABLE))
def test_plan_table_and_two_filters(q2):
    h = two_factor_hmm(0.8, 0.5, 0.8, q2, perm=[3, 1, 0, 2])
    toks, p, b = h.enumerate(6)
    assert len(p) == 4**6 and abs(p.sum() - 1) < 1e-12
    # Tower property: E[b] = pi (uniform) at any finite T.
    assert np.allclose(p @ b, h.pi, atol=1e-12)
    c = belief_coordinates(h, b)
    _, Cm = weighted_cov(c["m"], p)
    _, Cw = weighted_cov(c["walsh"], p)
    v1, v2 = np.diag(Cm)
    assert np.allclose([v1, v2, pr(Cm), pr(Cw)], TABLE[q2], atol=5e-7)
    assert abs(Cm[0, 1]) < 1e-12
    # Independent implementation: scalar magnetization recursion and its context probability.
    xs = factor_signs(toks, h)
    m1, lp1 = magnetization_filter(xs[..., 0], 0.8, 0.5)
    m2, lp2 = magnetization_filter(xs[..., 1], 0.8, q2)
    assert np.allclose(m1, c["m"][:, 0], atol=1e-13) and np.allclose(m2, c["m"][:, 1], atol=1e-13)
    assert np.allclose(np.exp(lp1 + lp2), p, atol=1e-15)
    # Walsh eigenvalues of the full 4-state posterior covariance are v1/4, v2/4, v1 v2/4.
    _, CB = weighted_cov(b, p)
    lam = eigvals_psd(CB)
    assert np.allclose(np.sort(lam[:3]), np.sort([v1 / 4, v2 / 4, v1 * v2 / 4]), atol=1e-12)
    assert lam[3] == 0


def test_additive_logits_reproduce_next_token_probs():
    h = two_factor_hmm(0.8, 0.5, 0.7, 0.8, perm=[2, 0, 3, 1])
    toks, p, b = h.enumerate(5)
    u = belief_coordinates(h, b)["u"]
    pairs = np.zeros((4, 2)); pairs[h.meta["perm"]] = PAIRS
    logit = u @ pairs.T                      # x1 u1 + x2 u2 for each token label
    soft = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
    assert np.allclose(soft, h.next_token_probs(b), atol=1e-12)


def test_sampled_moments_match_exact():
    h = two_factor_hmm(0.8, 0.5, 0.8, 0.25, perm=[1, 2, 3, 0])
    _, p, b = h.enumerate(6)
    _, C_exact = weighted_cov(belief_coordinates(h, b)["m"], p)
    rng = np.random.default_rng(0)
    toks = h.sample(200_000, 6, rng)
    m = belief_coordinates(h, h.filter(toks))["m"]
    C = np.cov(m.T)
    se = np.sqrt((m[:, 0] ** 4).mean() / len(m))  # rough standard error scale of a second moment
    assert np.all(np.abs(C - C_exact) < 6 * se)
    # Token frequencies of the sampler match the generator's stationary single-token distribution.
    freq = np.bincount(toks.ravel(), minlength=4) / toks.size
    assert np.allclose(freq, h.pi @ h.T.sum(2), atol=5e-3)


def test_lag_moments_match_enumeration():
    h = two_factor_hmm(0.7, 0.6, 0.9, 0.3, eps=0.2, perm=[0, 2, 1, 3])
    T = 5
    toks, p, _ = h.enumerate(T)
    pairs = np.zeros((4, 2)); pairs[h.meta["perm"]] = PAIRS
    f, g = pairs[:, 0], pairs[:, 1]
    for n in range(T):
        direct = p @ (f[toks[:, 0]] * g[toks[:, n]])
        assert abs(direct - h.lag_moment(f, g, n)) < 1e-12
    # Binary closed form for an independent factor: E[X_t X_{t+n}] = q^2 rho^n.
    hi = two_factor_hmm(0.7, 0.6, 0.9, 0.3)
    pi_ = np.zeros((4, 2)); pi_[hi.meta["perm"]] = PAIRS
    for n in range(1, 5):
        assert abs(hi.lag_moment(pi_[:, 0], pi_[:, 0], n) - 0.36 * 0.7**n) < 1e-12


def test_coupled_generator_is_valid_and_couples():
    for eps in [0.0, 0.05, 0.2]:
        h = two_factor_hmm(0.8, 0.5, 0.8, 0.5, eps=eps)
        assert np.allclose(h.pi @ h.P, h.pi) and np.all(h.T >= 0)
        _, p, b = h.enumerate(4)
        inter = belief_coordinates(h, b)["interaction"]
        if eps == 0:
            assert np.abs(inter).max() < 1e-12
        else:
            assert p @ inter**2 > 1e-6

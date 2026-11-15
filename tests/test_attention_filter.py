"""Stage C: positional attention algebra, imposed-filter covariance and weak-signal orders (derivations Package 3)."""
import numpy as np

from prt.attention_filter import (alpha, alpha_derivs, filter_ode, imposed_cov, lagged_features, statistic,
                                  weak_signal_variances)
from prt.generators import PAIRS, two_factor_hmm
from prt.spectra import weighted_cov


def test_alpha_derivatives_finite_difference():
    dk = -np.arange(6.0)
    a, da, dda = alpha_derivs(0.37, dk)
    h = 1e-5
    assert np.allclose(da, (alpha(0.37 + h, dk) - alpha(0.37 - h, dk)) / (2 * h), atol=1e-9)
    assert np.allclose(dda, (alpha(0.37 + h, dk) - 2 * a + alpha(0.37 - h, dk)) / h**2, atol=1e-5)
    assert abs(da.sum()) < 1e-15


def test_imposed_cov_matches_enumeration_with_temporal_correlations():
    h = two_factor_hmm(0.8, 0.5, 0.6, 0.8, eps=0.1, perm=[1, 3, 0, 2])
    T = 5
    toks, p, _ = h.enumerate(T)
    phi = np.zeros((4, 2)); phi[h.meta["perm"]] = PAIRS            # token -> (x1, x2)
    Gam = h.feature_lag_cov(phi, T)
    rng = np.random.default_rng(0)
    B = rng.standard_normal((T, 7, 2))
    Xk = lagged_features(toks, phi)
    H = np.einsum("kdr,mkr->md", B, Xk)
    _, C_enum = weighted_cov(H, p)
    assert np.allclose(imposed_cov(B, Gam), C_enum, atol=1e-12)
    # The independent-token approximation (k = l only) is not the identity for HMM data.
    Gdiag = Gam * np.eye(T)[:, :, None, None]
    assert np.abs(imposed_cov(B, Gdiag) - C_enum).max() > 1e-3


def test_weak_signal_error_orders():
    """Both approximations differ from the exact posterior variance at O(q^4) (derivations §1.4)."""
    rho, T = 0.6, 6
    errs = []
    for q in [0.05, 0.1, 0.2]:
        v = weak_signal_variances(rho, q, T)
        errs.append((abs(v["v_ws"] - v["v_exact"]), abs(v["v_lin"] - v["v_exact"])))
    errs = np.array(errs)
    slopes = np.log(errs[1:] / errs[:-1]) / np.log(2)
    assert np.all(np.abs(slopes - 4) < 0.3)


def test_filter_ode_reduces_loss():
    h = two_factor_hmm(0.8, 0.5, 0.8, 0.5)
    T = 4
    toks, p, b = h.enumerate(T)
    phi = np.zeros((4, 2)); phi[h.meta["perm"]] = PAIRS
    Xk = lagged_features(toks, phi)
    y = 0.8 * 0.5 * (b @ PAIRS[:, 0])                              # mu_1(c) = rho q m_1
    rec = filter_ode(Xk, p, y, -np.arange(T, dtype=float), w0=[0.1, 0.0], beta0=0.0, dt=0.05, steps=400)
    assert np.all(np.diff(rec["loss"]) <= 1e-14) and rec["loss"][-1] < 0.5 * rec["loss"][0]
    z, _ = statistic(Xk, rec["beta"][-1], -np.arange(T, dtype=float))
    assert np.allclose(weighted_cov(z, p)[1], rec["Cz"][-1])

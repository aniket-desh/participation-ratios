"""Calibration of the covariance observables against known spectra (experiment plan §5, §9.4; derivations §5.13)."""
import numpy as np

from prt.spectra import (MomentAccumulator, center_kernel, d_threshold, e_spec, eigvals_psd, kernel_spectrum,
                         moment_estimators_meanfree, pr, pr_from_eigs, summarize, weighted_cov)


def test_known_spectra():
    assert np.isclose(pr(np.eye(7)), 7)                         # isotropic: PR = d
    P = np.diag([1, 1, 1, 0, 0])
    assert np.isclose(pr(P), 3)                                 # rank-k projection: PR = k
    assert np.isclose(pr(np.diag([4.0, 1.0])), 25 / 17)        # (g^2+1)^2/(g^4+1) with g^2 = 4
    assert np.isnan(pr(np.zeros((3, 3))))                       # degenerate, not 0 or 1
    assert d_threshold(np.array([0.9, 0.06, 0.04])) == 2


def test_prescribed_encodings_transport_covariance():
    """Isometry preserves PR and lowers nPR; stretch changes PR at fixed rank; rotation changes nothing."""
    rng = np.random.default_rng(1)
    b = rng.standard_normal((50_000, 2)) * [1.0, 0.5]
    _, Sb = weighted_cov(b)
    Q, _ = np.linalg.qr(rng.standard_normal((32, 2)))           # isometric embedding into d = 32
    s_b, s_iso = summarize(Sb), summarize(weighted_cov(b @ Q.T)[1])
    assert np.isclose(s_b["pr"], s_iso["pr"]) and np.isclose(s_iso["npr"], s_b["pr"] / 32)
    E = Q @ np.diag([3.0, 1.0])
    C = weighted_cov(b @ E.T)[1]
    assert np.allclose(C, E @ Sb @ E.T)                         # C_h = E Sigma_b E^T
    assert not np.isclose(pr(C), s_b["pr"]) and np.sum(eigvals_psd(C) > 0) == 2
    R, _ = np.linalg.qr(rng.standard_normal((2, 2)))
    assert np.isclose(pr(weighted_cov(b @ R.T)[1]), s_b["pr"])
    # A nonlinear encoding of a two-variable manifold has linear rank three.
    m = rng.uniform(-1, 1, (10_000, 2))
    assert np.sum(eigvals_psd(weighted_cov(np.c_[m, m[:, 0] * m[:, 1]])[1]) > 1e-8) == 3


def test_kernel_spectrum_equals_covariance_spectrum():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((20, 9))
    p = rng.random(20); p /= p.sum()
    mu, C = weighted_cov(F, p)
    K = center_kernel(F @ F.T / 9, p)                           # centering the uncentered kernel
    lam_k = kernel_spectrum(K, p)
    lam_c = eigvals_psd(C / 9)
    n = min(len(lam_k), len(lam_c))
    assert np.allclose(lam_k[:n], lam_c[:n], atol=1e-12)
    assert np.isclose(pr_from_eigs(lam_k), pr(C))


def test_streaming_and_meanfree_estimators():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6))
    X = 100 + rng.standard_normal((200_000, 6)) @ A.T            # large mean, to test cancellation
    acc = MomentAccumulator(6)
    for i in range(0, len(X), 7_000):
        acc.update(X[i:i + 7_000])
    assert np.allclose(acc.cov(), np.cov(X.T, bias=True), rtol=1e-9, atol=1e-9)
    t1, t2 = moment_estimators_meanfree(X)
    C = A @ A.T
    assert abs(t1 / np.trace(C) - 1) < 0.02 and abs(t2 / np.trace(C @ C) - 1) < 0.05


def test_e_spec_pads():
    assert np.isclose(e_spec(np.array([1.0]), np.array([0.5, 0.5])), 0.5)

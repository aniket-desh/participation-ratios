"""Covariance-spectrum observables (experiment plan §2.1, §10; derivations §1.2, §5.7, §5.13).

Every covariance here is a covariance across contexts at one declared layer and position, never across seeds.
All accumulation is float64.
"""
from __future__ import annotations

import numpy as np

# Eigenvalues below EIG_TOL * max eigenvalue are treated as floating-point zeros (plan §2.1 asks for a
# documented tolerance). Negative eigenvalues of a PSD matrix above this size raise instead of being clipped.
EIG_TOL = 1e-12


def eigvals_psd(C: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of a symmetric PSD matrix, with round-off negatives set to zero."""
    C = 0.5 * (C + C.T)
    lam = np.linalg.eigvalsh(C)[::-1]
    scale = max(lam[0], 0.0)
    if lam[-1] < -1e-8 * max(scale, 1e-300):
        raise ValueError(f"matrix is not PSD: min eigenvalue {lam[-1]:.3e}, max {lam[0]:.3e}")
    lam[lam < EIG_TOL * scale] = 0.0
    return lam


def pr_from_eigs(lam: np.ndarray) -> float:
    """PR = (sum λ)^2 / sum λ^2. NaN marks the degenerate zero covariance (plan §2.1: undefined, not 0 or 1)."""
    s2 = float(np.sum(lam**2))
    return float(np.sum(lam)) ** 2 / s2 if s2 > 0 else float("nan")


def pr(C: np.ndarray) -> float:
    """PR(C) = (tr C)^2 / tr(C^2), computed from traces (no eigendecomposition)."""
    C = np.asarray(C, dtype=np.float64)
    t2 = float(np.sum(C * C.T))
    return float(np.trace(C)) ** 2 / t2 if t2 > 0 else float("nan")


def d_threshold(lam: np.ndarray, frac: float = 0.95) -> int:
    """d_95 = min{k : sum_{i<=k} p_i >= frac} for descending eigenvalues."""
    p = lam / lam.sum()
    return int(np.searchsorted(np.cumsum(p), frac - 1e-12) + 1)


def summarize(C: np.ndarray) -> dict:
    """The observables the plan asks to report together: trace, PR, nPR, d95 and the normalized spectrum."""
    lam = eigvals_psd(C)
    tr = float(lam.sum())
    return {
        "trace": tr,
        "pr": pr_from_eigs(lam),
        "npr": pr_from_eigs(lam) / C.shape[0],
        "d95": d_threshold(lam) if tr > 0 else None,
        "eigs": lam,
        "p": lam / tr if tr > 0 else lam,
    }


def weighted_cov(X: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Probability-weighted mean and covariance of rows of X (population, not unbiased-sample, normalization)."""
    X = np.asarray(X, dtype=np.float64)
    w = np.full(len(X), 1.0 / len(X)) if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    mu = w @ X
    Xc = X - mu
    return mu, (Xc * w[:, None]).T @ Xc


class MomentAccumulator:
    """Streams first and second moments so covariance needs O(d^2) memory (plan §9.4).

    Uses a shifted accumulation (shift = first batch mean) to avoid cancellation when |mean| >> std.
    """

    def __init__(self, d: int):
        self.n = 0
        self.shift = None
        self.s1 = np.zeros(d)
        self.s2 = np.zeros((d, d))

    def update(self, X) -> None:
        X = np.asarray(X, dtype=np.float64)
        if self.shift is None:
            self.shift = X.mean(0)
        Xs = X - self.shift
        self.n += len(X)
        self.s1 += Xs.sum(0)
        self.s2 += Xs.T @ Xs

    def mean(self) -> np.ndarray:
        return self.shift + self.s1 / self.n

    def cov(self) -> np.ndarray:
        m = self.s1 / self.n
        return self.s2 / self.n - np.outer(m, m)


def kernel_spectrum(K: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Nonzero covariance/d eigenvalues from a centered context kernel: eig(P^1/2 K P^1/2) (roadmap §7, derivations §5.7)."""
    sp = np.sqrt(p)
    return eigvals_psd(sp[:, None] * K * sp[None, :])


def center_kernel(K: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Population-center an uncentered kernel: subtract weighted row/column means, add weighted grand mean (roadmap §7)."""
    r = K @ p
    return K - r[:, None] - r[None, :] + p @ K @ p


def moment_estimators_meanfree(H: np.ndarray) -> tuple[float, float]:
    """Unbiased tr C and tr C^2 from disjoint independent quadruples of uncentered samples (derivations §5.13).

    tr C = E||h1-h2||^2 / 2 and tr C^2 = E[((h1-h2)·(h3-h4))^2] / 4. Rows of H must be independent draws.
    """
    n = (len(H) // 4) * 4
    h1, h2, h3, h4 = (H[i:n:4] for i in range(4))
    D12, D34 = h1 - h2, h3 - h4
    t1 = 0.25 * (np.sum(D12**2, 1).mean() + np.sum(D34**2, 1).mean())
    t2 = 0.25 * np.mean(np.sum(D12 * D34, 1) ** 2)
    return float(t1), float(t2)


def e_pr(pr_pred: float, pr_obs: float) -> float:
    """Relative PR error (plan §10)."""
    return abs(pr_pred - pr_obs) / pr_obs


def e_spec(p_pred: np.ndarray, p_obs: np.ndarray) -> float:
    """Half L1 distance between normalized spectra, zero-padding the shorter one (plan §10)."""
    n = max(len(p_pred), len(p_obs))
    a = np.zeros(n); a[: len(p_pred)] = np.sort(p_pred)[::-1]
    b = np.zeros(n); b[: len(p_obs)] = np.sort(p_obs)[::-1]
    return 0.5 * float(np.abs(a - b).sum())


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between column spans of A and B."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))

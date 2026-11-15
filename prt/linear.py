"""Balanced two-layer linear learning on whitened beliefs (derivations Package 2; experiment plan §6, Stage B).

Loss L = 1/2 ||D E - M||_F^2 for whitened inputs z (E[z z^T] = I), hidden h = E z, so C_h = E E^T.
Gradient flow with unit rates: dD/dt = (M - D E) E^T, dE/dt = D^T (M - D E).
"""
from __future__ import annotations

import numpy as np


def whiten(b: np.ndarray, p: np.ndarray | None = None, tol: float = 1e-12):
    """z = Sigma_b^{-1/2} (b - mean) on the positive-eigenvalue support (derivations §2.1). Returns z, W, mean."""
    p = np.full(len(b), 1 / len(b)) if p is None else p / p.sum()
    mu = p @ b
    bc = b - mu
    S = (bc * p[:, None]).T @ bc
    lam, V = np.linalg.eigh(S)
    keep = lam > tol * lam.max()
    W = np.diag(lam[keep] ** -0.5) @ V[:, keep].T  # (r_*, r): Lambda^{-1/2} V^T on the support
    if keep.all():
        W = V @ W                                  # symmetric inverse root V Lambda^{-1/2} V^T when full rank
    return bc @ W.T, W, mu


def sigmoid_modes(s: np.ndarray, a0: np.ndarray, tau: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Aligned balanced mode amplitudes a_i(tau) (derivations §2.5-2.6).

    da/dt = 2 a (s - lam - a). For s_eff = s - lam != 0: a = s_eff / (1 + (s_eff/a0 - 1) e^{-2 s_eff tau});
    for s_eff = 0: a = a0 / (1 + 2 a0 tau). Returns (len(tau), len(s)).
    """
    s_eff = np.asarray(s, float) - lam
    a0 = np.broadcast_to(np.asarray(a0, float), s_eff.shape)
    tau = np.asarray(tau, float)[:, None]
    out = np.empty((tau.shape[0], len(s_eff)))
    for i, (se, a) in enumerate(zip(s_eff, a0)):
        out[:, i] = (a / (1 + 2 * a * tau[:, 0])) if se == 0 else se / (1 + (se / a - 1) * np.exp(-2 * se * tau[:, 0]))
    return out


def pr_of_modes(a: np.ndarray) -> np.ndarray:
    return a.sum(-1) ** 2 / (a**2).sum(-1)


def balanced_endpoint_pr(M: np.ndarray) -> float:
    """PR(C_h) = (sum s_i)^2 / sum s_i^2 at the minimum-norm exact factorization (derivations §2.4)."""
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-12 * s.max()]
    return float(s.sum() ** 2 / (s**2).sum())


def aligned_init(M: np.ndarray, d: int, a0, rng=None):
    """D = U diag(sqrt a0) R^T, E = R diag(sqrt a0) V^T with a random (or identity) orthonormal R in R^{d x r}."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = len(s)
    if rng is None:
        R = np.eye(d, r)
    else:
        R, _ = np.linalg.qr(rng.standard_normal((d, r)))
    sq = np.sqrt(np.broadcast_to(np.asarray(a0, float), (r,)))
    return U @ np.diag(sq) @ R.T, R @ np.diag(sq) @ Vt


def random_balanced_init(p: int, r: int, d: int, scale: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Small random init made exactly balanced (E E^T = D^T D): E = R S Q^T, D = P S R^T from one SVD-like draw."""
    k = min(p, r, d)
    R, _ = np.linalg.qr(rng.standard_normal((d, k)))
    Q, _ = np.linalg.qr(rng.standard_normal((r, k)))
    P, _ = np.linalg.qr(rng.standard_normal((p, k)))
    S = np.diag(scale * (0.5 + rng.random(k)))
    return P @ S @ R.T, R @ S @ Q.T


def gradient_flow(D, E, M, dt: float, steps: int, lam: float = 0.0, eta_D: float = 1.0, eta_E: float = 1.0,
                  record_every: int = 1, method: str = "rk4"):
    """Integrate the exact finite-width ODE (optionally with factor penalty lam and layer rates).

    Records tau, hidden covariance eigenvalues eig(E E^T), singular values of D E, imbalance ||E E^T - D^T D||_F, loss.
    """
    def f(D, E):
        R = M - D @ E
        return eta_D * (R @ E.T - lam * D), eta_E * (D.T @ R - lam * E)

    rec = {k: [] for k in ["tau", "eig_h", "sv_DE", "imbalance", "loss"]}

    def log(n):
        rec["tau"].append(n * dt)
        rec["eig_h"].append(np.sort(np.linalg.eigvalsh(E @ E.T))[::-1])
        rec["sv_DE"].append(np.linalg.svd(D @ E, compute_uv=False))
        rec["imbalance"].append(np.linalg.norm(E @ E.T - D.T @ D))
        rec["loss"].append(0.5 * np.linalg.norm(D @ E - M) ** 2)

    log(0)
    for n in range(1, steps + 1):
        if method == "euler":
            dD, dE = f(D, E)
            D, E = D + dt * dD, E + dt * dE
        else:
            k1 = f(D, E)
            k2 = f(D + 0.5 * dt * k1[0], E + 0.5 * dt * k1[1])
            k3 = f(D + 0.5 * dt * k2[0], E + 0.5 * dt * k2[1])
            k4 = f(D + dt * k3[0], E + dt * k3[1])
            D = D + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            E = E + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if n % record_every == 0:
            log(n)
    return D, E, {k: np.array(v) for k, v in rec.items()}

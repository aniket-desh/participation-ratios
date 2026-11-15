"""Restricted positional attention (derivations Package 3; experiment plan §7, Stage C).

Final-position statistic z(c; beta) = sum_k alpha_k(beta) phi~(x_{T-k}), alpha_k = softmax_k(beta d_k), with
relative-position features d_k (default d_k = -k, so beta = -log rho gives alpha_k ∝ rho^k).
Contexts are enumerated exactly; arrays use lag index k = 0..T-1 where k = 0 is the final token.
"""
from __future__ import annotations

import numpy as np



def lagged_features(toks: np.ndarray, phi: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Xk[c, k, :] = phi(x_{T-k}) - mean, shape (M, T, r)."""
    X = phi[toks[:, ::-1]]
    return X if mean is None else X - mean


def alpha(beta: float, dk: np.ndarray) -> np.ndarray:
    a = np.exp(beta * dk - np.max(beta * dk))
    return a / a.sum()


def alpha_derivs(beta: float, dk: np.ndarray):
    """alpha, d alpha/d beta = alpha (d - dbar), d^2 alpha/d beta^2 = alpha[(d - dbar)^2 - Var_alpha(d)] (derivations §3.1)."""
    a = alpha(beta, dk)
    dbar = a @ dk
    var = a @ (dk - dbar) ** 2
    return a, a * (dk - dbar), a * ((dk - dbar) ** 2 - var)


def statistic(Xk: np.ndarray, beta: float, dk: np.ndarray):
    """z(c) and z_beta(c) for all contexts: (M, r) each."""
    a, da, _ = alpha_derivs(beta, dk)
    return np.einsum("k,mkr->mr", a, Xk), np.einsum("k,mkr->mr", da, Xk)


def imposed_cov(B: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """C_h = sum_{k,l} B_k Gamma_{kl} B_l^T for deterministic positional maps B (T, d, r) (derivations §3.2)."""
    return np.einsum("kdr,klrs,les->de", B, Gamma, B)


# ---------------------------------------------------------------------------
# Weak-signal benchmark for one binary factor (derivations §1.4, plan §7.1)
# ---------------------------------------------------------------------------

def weak_signal_variances(rho: float, q: float, T: int) -> dict:
    """Three variances that the plan asks to compare:

    v_lin: exact variance of the linear filter q sum_k rho^k X_{T-k} using Gamma_{kl} = 1 (k=l), q^2 rho^|k-l| else;
    v_ws:  leading weak-signal formula q^2 (1 - rho^{2T}) / (1 - rho^2);
    v_exact: exact posterior variance E[m_T^2] by enumeration of 2^T sign strings.
    """
    from .generators import magnetization_filter
    k = np.arange(T)
    Gam = np.where(k[:, None] == k[None, :], 1.0, q**2 * rho ** np.abs(k[:, None] - k[None, :]))
    w = rho**k
    v_lin = q**2 * w @ Gam @ w
    v_ws = q**2 * (1 - rho ** (2 * T)) / (1 - rho**2)
    x = np.array(np.meshgrid(*[[-1, 1]] * T, indexing="ij")).reshape(T, -1).T
    m, logp = magnetization_filter(x, rho, q)
    v_exact = float(np.exp(logp) @ m**2)
    return dict(v_lin=float(v_lin), v_ws=float(v_ws), v_exact=v_exact)


def softmax_head_gain(rho: float, q: float, T: int):
    """Positional softmax head implementing the leading filter: weights rho^k / Z_T and value/output gain q Z_T."""
    k = np.arange(T)
    Z = np.sum(rho**k)
    return rho**k / Z, q * Z


# ---------------------------------------------------------------------------
# Trainable positional kernel and linear readout (derivations §3.3)
# ---------------------------------------------------------------------------

def filter_ode(Xk, p, y, dk, w0, beta0, eta_w=1.0, eta_beta=1.0, dt=1e-2, steps=1000, record_every=10):
    """RK4 integration of dw/dt = -eta_w sum_c p_c e_c z(c), dbeta/dt = -eta_beta sum_c p_c e_c w^T z_beta(c).

    e_c = w^T z(c; beta) - y(c). All coefficients are exact sums over the enumerated generator population.
    Records tau, beta, w, loss and the predicted covariance C_z(beta) of the statistic.
    """
    p = p / p.sum()

    def rhs(w, beta):
        z, zb = statistic(Xk, beta, dk)
        e = z @ w - y
        return -eta_w * (p * e) @ z, -eta_beta * (p * e) @ (zb @ w)

    def loss(w, beta):
        z, _ = statistic(Xk, beta, dk)
        return 0.5 * p @ (z @ w - y) ** 2

    def cov(beta):
        z, _ = statistic(Xk, beta, dk)
        zc = z - p @ z
        return (zc * p[:, None]).T @ zc

    w, beta = np.array(w0, float), float(beta0)
    rec = {k: [] for k in ["tau", "beta", "w", "loss", "Cz"]}
    for n in range(steps + 1):
        if n % record_every == 0:
            rec["tau"].append(n * dt); rec["beta"].append(beta); rec["w"].append(w.copy())
            rec["loss"].append(loss(w, beta)); rec["Cz"].append(cov(beta))
        if n == steps:
            break
        k1 = rhs(w, beta)
        k2 = rhs(w + 0.5 * dt * k1[0], beta + 0.5 * dt * k1[1])
        k3 = rhs(w + 0.5 * dt * k2[0], beta + 0.5 * dt * k2[1])
        k4 = rhs(w + dt * k3[0], beta + dt * k3[1])
        w = w + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        beta = beta + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return {k: np.array(v) for k, v in rec.items()}

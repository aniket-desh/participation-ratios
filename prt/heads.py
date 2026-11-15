"""Restricted many-head nonlinear transformer and its continuum (mean-field) limit (derivations Package 5).

Each head has theta = (a1, a2, beta, kappa, g). For a context of signed pairs x_s in {-1,+1}^2, s = 1..T:
    l_s   = beta t_s + kappa d_s(c),        t_s = s - T,  d_s = (w.x_T)(w.x_s),  w = (1,1)/sqrt2
    alpha = softmax_s(l),  z_j = sum_s alpha_s x_sj,  r_j = x_Tj + z_j,  psi_j = r_j + tanh(g r_j)
Signed marginal logits f_j(c) = sum_n w_n a_nj psi_j(c; theta_n) with weights w_n (1/H for a finite network,
quadrature weights for the continuum). Loss per context: sum_j [log(2 cosh f_j) - mu_j(c) f_j].
Mean-field gradient flow: dtheta_n/dtau = v(theta_n) = -sum_c p_c sum_j e_j(c) grad_theta phi_j(c; theta_n),
e_j = tanh f_j - mu_j, phi_j = a_j psi_j (derivations §5.4-5.5).

The measured representation is the concatenation of psi over heads; its centered kernel is
K(c,c') = sum_n w_n Xi_cc'(theta_n), Xi = 1/2 psi~(c).psi~(c'), and S = P^1/2 K P^1/2 has the covariance spectrum / 2H.

Parameter index order everywhere: A1, A2, BETA, KAPPA, G = 0..4.
"""
from __future__ import annotations

import numpy as np

A1, A2, BETA, KAPPA, G = range(5)
W_DIR = np.array([1.0, 1.0]) / np.sqrt(2)


class Population:
    """Exact finite context population for the two-factor binary generator: xs (M,T,2), p (M,), mu (M,2)."""

    def __init__(self, xs: np.ndarray, p: np.ndarray, mu: np.ndarray):
        self.xs, self.p, self.mu = xs.astype(np.float64), p / p.sum(), mu
        M, T, _ = xs.shape
        self.T = T
        self.ts = np.arange(1, T + 1) - T                        # t_s = s - T
        self.ds = (self.xs[:, -1] @ W_DIR)[:, None] * (self.xs @ W_DIR)  # (M, T)

    @classmethod
    def independent(cls, rho1, q1, rho2, q2, T):
        """Enumerate all 4^T contexts; mu_j = rho_j q_j m_j(c) (derivations §5.1)."""
        from .generators import magnetization_filter
        xs = np.array(np.meshgrid(*[[-1, 1]] * (2 * T), indexing="ij")).reshape(2 * T, -1).T.reshape(-1, T, 2)
        m1, lp1 = magnetization_filter(xs[..., 0], rho1, q1)
        m2, lp2 = magnetization_filter(xs[..., 1], rho2, q2)
        return cls(xs, np.exp(lp1 + lp2), np.stack([rho1 * q1 * m1, rho2 * q2 * m2], -1))


def features(theta: np.ndarray, pop: Population, second: bool = False):
    """psi (Q,M,2) and first derivatives dpsi (Q,M,2,3) w.r.t. (beta, kappa, g) (derivations §5.2).

    With second=True also returns z, r, zu (Q,M,2,2) for (beta,kappa) and the attention weights.
    """
    beta, kappa, g = theta[:, BETA], theta[:, KAPPA], theta[:, G]
    xs, ts, ds = pop.xs, pop.ts, pop.ds
    logits = beta[:, None, None] * ts[None, None, :] + kappa[:, None, None] * ds[None]   # (Q,M,T)
    logits -= logits.max(-1, keepdims=True)
    al = np.exp(logits)
    al /= al.sum(-1, keepdims=True)
    z = np.einsum("qms,msj->qmj", al, xs)
    r = xs[None, :, -1, :] + z
    gr = g[:, None, None] * r
    th = np.tanh(gr)
    sech2 = 1 - th**2
    psi = r + th
    # z_{j,u} = sum_s alpha_s (x_sj - z_j) e^u_s with e^beta = t_s, e^kappa = d_s
    dev = xs[None] - z[:, :, None, :]                                                    # (Q,M,T,2)
    z_beta = np.einsum("qms,qmsj,s->qmj", al, dev, ts.astype(float))
    z_kappa = np.einsum("qms,qmsj,ms->qmj", al, dev, ds)
    D = 1 + g[:, None, None] * sech2
    dpsi = np.stack([D * z_beta, D * z_kappa, r * sech2], -1)
    if second:
        return psi, dpsi, dict(al=al, z=z, r=r, th=th, sech2=sech2, zu=np.stack([z_beta, z_kappa], -1))
    return psi, dpsi


def logits(theta, w, psi):
    """f_j(c) = sum_n w_n a_nj psi_j(c; theta_n): (M, 2)."""
    return np.einsum("q,qj,qmj->mj", w, theta[:, :2], psi)


def loss(theta, w, pop):
    psi, _ = features(theta, pop)
    f = logits(theta, w, psi)
    return float(pop.p @ np.sum(np.log(2 * np.cosh(f)) - pop.mu * f, 1))


def velocity(theta, w, pop, mask=None):
    """v(theta_n) for every head/characteristic n: (Q, 5). mask (5,) zeroes frozen coordinates (controls)."""
    psi, dpsi = features(theta, pop)
    f = logits(theta, w, psi)
    e = np.tanh(f) - pop.mu                                       # (M, 2)
    pe = pop.p[:, None] * e
    v = np.empty_like(theta)
    v[:, :2] = -np.einsum("mj,qmj->qj", pe, psi)
    v[:, 2:] = -np.einsum("mj,qj,qmju->qu", pe, theta[:, :2], dpsi)
    return v if mask is None else v * mask


def kernel(theta, w, pop):
    """Centered kernel K(c,c') = sum_n w_n 1/2 psi~_n(c).psi~_n(c') and S = P^1/2 K P^1/2."""
    psi, _ = features(theta, pop)
    psit = psi - np.einsum("m,qmj->qj", pop.p, psi)[:, None, :]
    K = 0.5 * np.einsum("q,qmj,qnj->mn", w, psit, psit)
    sp = np.sqrt(pop.p)
    return K, sp[:, None] * K * sp[None, :]


def pr_S(S):
    return float(np.trace(S) ** 2 / np.sum(S * S))


def integrate(theta0, w, pop, dt, steps, record_every=1, mask=None, record=("loss", "S")):
    """RK4 for the characteristic/head ODE. Returns final theta and records at tau = n dt."""
    th = theta0.copy()
    rec = {"tau": [], **{k: [] for k in record}, "theta_mean": []}

    def log(n):
        rec["tau"].append(n * dt)
        rec["theta_mean"].append(w @ th)
        if "loss" in rec:
            rec["loss"].append(loss(th, w, pop))
        if "S" in rec:
            rec["S"].append(kernel(th, w, pop)[1])

    log(0)
    for n in range(1, steps + 1):
        k1 = velocity(th, w, pop, mask)
        k2 = velocity(th + 0.5 * dt * k1, w, pop, mask)
        k3 = velocity(th + 0.5 * dt * k2, w, pop, mask)
        k4 = velocity(th + dt * k3, w, pop, mask)
        th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if n % record_every == 0:
            log(n)
    return th, {k: np.array(v) for k, v in rec.items()}


# ---------------------------------------------------------------------------
# Initial law and quadrature (derivations §5.8)
# ---------------------------------------------------------------------------
BETA_RANGE, KAPPA_RANGE = (-1.0, 1.0), (-0.5, 0.5)


def theta_from_labels(beta0, kappa0):
    th = np.zeros((len(beta0), 5))
    th[:, BETA], th[:, KAPPA], th[:, G] = beta0, kappa0, 1.0
    return th


def quadrature_init(n_per_dim: int):
    """Tensor-product Gauss–Legendre nodes on Unif([-1,1] x [-1/2,1/2]); weights sum to one."""
    x, wx = np.polynomial.legendre.leggauss(n_per_dim)
    b = 0.5 * (BETA_RANGE[1] - BETA_RANGE[0]) * x + 0.5 * sum(BETA_RANGE)
    k = 0.5 * (KAPPA_RANGE[1] - KAPPA_RANGE[0]) * x + 0.5 * sum(KAPPA_RANGE)
    B, Kp = np.meshgrid(b, k, indexing="ij")
    W = np.outer(wx, wx) / 4.0
    return theta_from_labels(B.ravel(), Kp.ravel()), W.ravel()


def random_init(H: int, rng):
    """H iid heads from the same initial law, each weight 1/H."""
    th = theta_from_labels(rng.uniform(*BETA_RANGE, H), rng.uniform(*KAPPA_RANGE, H))
    return th, np.full(H, 1.0 / H)


# ---------------------------------------------------------------------------
# Feature-learning onset (derivations §5.9)
# ---------------------------------------------------------------------------

def onset(theta0, w, pop):
    """Zero-time coefficients: A_j, B_u, K''(0), S_0, S_2 and PR(0), PR''(0).

    A_j(xi) = sum_c p_c mu_j psi_j;  B_u(xi) = sum_j A_j sum_c p_c mu_j psi_{j,u};
    K''(0) = int sum_u d_u Xi B_u;   PR'' = 2 A0 tr S2 / B20 - 2 A0^2 tr(S0 S2) / B20^2.
    """
    assert np.all(theta0[:, :2] == 0), "onset formulas assume zero initial readouts"
    psi, dpsi = features(theta0, pop)
    p, mu = pop.p, pop.mu
    A = np.einsum("m,mj,qmj->qj", p, mu, psi)                    # (Q,2)
    Bu = np.einsum("qj,m,mj,qmju->qu", A, p, mu, dpsi)            # (Q,3)
    psit = psi - np.einsum("m,qmj->qj", p, psi)[:, None, :]
    dpsit = dpsi - np.einsum("m,qmju->qju", p, dpsi)[:, None]
    # d_u Xi_cc' = 1/2 sum_j [d_u psi~_j(c) psi~_j(c') + psi~_j(c) d_u psi~_j(c')]
    dpsit_B = np.einsum("qmju,qu->qmj", dpsit, Bu)
    Kdd = 0.5 * np.einsum("q,qmj,qnj->mn", w, dpsit_B, psit)
    Kdd = Kdd + Kdd.T
    K0 = 0.5 * np.einsum("q,qmj,qnj->mn", w, psit, psit)
    sp = np.sqrt(p)
    S0 = sp[:, None] * K0 * sp[None]
    S2 = sp[:, None] * Kdd * sp[None]
    A0, B20 = np.trace(S0), np.sum(S0 * S0)
    prdd = 2 * A0 * np.trace(S2) / B20 - 2 * A0**2 * np.sum(S0 * S2) / B20**2
    return dict(A=A, B=Bu, Kdd=Kdd, S0=S0, S2=S2, pr0=float(A0**2 / B20), prdd0=float(prdd))


# ---------------------------------------------------------------------------
# Interaction extension for dependent generators (derivations §5.14)
# ---------------------------------------------------------------------------
# theta gains a sixth coordinate b (index B12); chi = psi_1 psi_2 and f12 = sum_n w_n b_n chi_n.
# Logits L_y = y1 f1 + y2 f2 + y1 y2 f12, Z = 2 e^{f12} cosh(f1+f2) + 2 e^{-f12} cosh(f1-f2),
# conditional loss log Z - mu1 f1 - mu2 f2 - mu12 f12 with mu12 = E[Y1 Y2 | c] from full joint filtering.
B12 = 5


def population_from_hmm(hmm, T: int) -> tuple[Population, np.ndarray]:
    """Exact population of a (possibly coupled) two-factor HMM: returns Population(xs, p, mu) and mu12 (M,).

    Next-token moments come from the joint posterior: mu_j = E[Y_j | c], mu12 = E[Y1 Y2 | c].
    """
    from .generators import token_pairs
    toks, p, b = hmm.enumerate(T)
    pairs = token_pairs(hmm).astype(float)        # (V, 2) signs per token label
    P = hmm.next_token_probs(b)                    # (M, V)
    mu = P @ pairs
    mu12 = P @ (pairs[:, 0] * pairs[:, 1])
    return Population(pairs[toks], p, mu), mu12


def logits_int(theta, w, psi):
    f = logits(theta, w, psi)
    f12 = np.einsum("q,q,qm->m", w, theta[:, B12], psi[..., 0] * psi[..., 1])
    return f, f12


def _logZ_and_grads(f, f12):
    """log Z and its gradient w.r.t. (f1, f2, f12), computed stably via log-sum-exp over the four tokens."""
    y = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    L = f @ y.T + f12[:, None] * (y[:, 0] * y[:, 1])[None]   # (M, 4)
    Lmax = L.max(1, keepdims=True)
    e = np.exp(L - Lmax)
    Z = e.sum(1, keepdims=True)
    q = e / Z                                                # model distribution over tokens
    return (Lmax + np.log(Z))[:, 0], q @ y, q @ (y[:, 0] * y[:, 1])


def loss_int(theta, w, pop, mu12):
    psi, _ = features(theta, pop)
    f, f12 = logits_int(theta, w, psi)
    logZ, _, _ = _logZ_and_grads(f, f12)
    return float(pop.p @ (logZ - np.sum(pop.mu * f, 1) - mu12 * f12))


def velocity_int(theta, w, pop, mu12, mask=None):
    """Mean-field velocity with the interaction logit: (Q, 6)."""
    psi, dpsi = features(theta, pop)
    f, f12 = logits_int(theta, w, psi)
    _, Ey, Ey12 = _logZ_and_grads(f, f12)
    e, e12 = Ey - pop.mu, Ey12 - mu12
    pe, pe12 = pop.p[:, None] * e, pop.p * e12
    chi = psi[..., 0] * psi[..., 1]
    dchi = dpsi[..., 0, :] * psi[..., 1, None] + psi[..., 0, None] * dpsi[..., 1, :]   # (Q,M,3)
    v = np.empty_like(theta)
    v[:, :2] = -np.einsum("mj,qmj->qj", pe, psi)
    v[:, 2:5] = -(np.einsum("mj,qj,qmju->qu", pe, theta[:, :2], dpsi) + np.einsum("m,q,qmu->qu", pe12, theta[:, B12], dchi))
    v[:, B12] = -np.einsum("m,qm->q", pe12, chi)
    return v if mask is None else v * mask


def integrate_int(theta0, w, pop, mu12, dt, steps, record_every=1):
    """RK4 for the interaction model. theta0 is (Q, 6) with b(0) = 0 by default."""
    th = theta0.copy()
    rec = {"tau": [], "loss": [], "S": [], "theta_mean": []}

    def log(n):
        rec["tau"].append(n * dt); rec["loss"].append(loss_int(th, w, pop, mu12))
        rec["S"].append(kernel(th, w, pop)[1]); rec["theta_mean"].append(w @ th)

    log(0)
    for n in range(1, steps + 1):
        k1 = velocity_int(th, w, pop, mu12)
        k2 = velocity_int(th + 0.5 * dt * k1, w, pop, mu12)
        k3 = velocity_int(th + 0.5 * dt * k2, w, pop, mu12)
        k4 = velocity_int(th + dt * k3, w, pop, mu12)
        th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if n % record_every == 0:
            log(n)
    return th, {k: np.array(v) for k, v in rec.items()}


def with_interaction(theta):
    """Append a zero interaction coordinate b to (Q, 5) parameters."""
    return np.hstack([theta, np.zeros((len(theta), 1))])

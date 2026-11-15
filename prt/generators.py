"""Sequence generators with exact filtering (derivations Package 1; experiment plan §3, §4.2, §5).

An edge-emitting HMM is given by token-labelled matrices T[x]_{ij} = P(X_t=x, S_t=j | S_{t-1}=i).
Contexts start from the declared prior pi (stationary by default), so beliefs are finite-window posteriors.
Row-vector convention throughout, as in the derivations: b_t = b_{t-1} T[x] / (b_{t-1} T[x] 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

SIGNS = np.array([-1, 1])
# Hidden/observation pairs are indexed k = 2*i1 + i2 with sign s = 2*i - 1, i.e. (-,-),(-,+),(+,-),(+,+).
PAIRS = np.array([(s1, s2) for s1 in SIGNS for s2 in SIGNS])  # (4, 2)


def stationary(P: np.ndarray) -> np.ndarray:
    """Row vector pi with pi P = pi, sum pi = 1 (ergodic chain)."""
    K = len(P)
    A = np.vstack([(P.T - np.eye(K)), np.ones(K)])
    b = np.zeros(K + 1); b[-1] = 1
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


@dataclass
class HMM:
    T: np.ndarray                      # (V, K, K) token-labelled transition matrices
    pi: np.ndarray | None = None       # initial row distribution; stationary if None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=np.float64)
        self.P = self.T.sum(0)
        assert np.allclose(self.P.sum(1), 1), "sum_x T[x] must be row-stochastic"
        if self.pi is None:
            self.pi = stationary(self.P)

    @property
    def V(self) -> int:
        return self.T.shape[0]

    @property
    def K(self) -> int:
        return self.T.shape[1]

    # ---------- exact enumeration (prefix tree of unnormalized state vectors) ----------
    def enumerate(self, T: int):
        """All V^T contexts with p_T(c) > 0: tokens (M,T), probabilities (M,), beliefs after the context (M,K).

        alpha(c) = pi T[x1]...T[xT]; p(c) = alpha 1; b(c) = alpha / p (derivations §1.1).
        Contexts are ordered lexicographically with x1 most significant.
        """
        alpha = self.pi[None, :]
        toks = np.zeros((1, 0), dtype=np.int64)
        for _ in range(T):
            alpha = np.einsum("mi,xij->mxj", alpha, self.T).reshape(-1, self.K)
            toks = np.concatenate([np.repeat(toks, self.V, 0), np.tile(np.arange(self.V), len(toks))[:, None]], 1)
        p = alpha.sum(1)
        keep = p > 0
        return toks[keep], p[keep], alpha[keep] / p[keep, None]

    # ---------- sampling ----------
    def sample(self, N: int, T: int, rng: np.random.Generator) -> np.ndarray:
        """N independent contexts of length T, each started from pi. Returns tokens (N,T)."""
        flat = self.T.transpose(1, 0, 2).reshape(self.K, -1)  # (K, V*K): joint (x, j) given i
        cum = np.cumsum(flat, 1)
        s = rng.choice(self.K, size=N, p=self.pi)
        toks = np.empty((N, T), dtype=np.int64)
        for t in range(T):
            u = rng.random(N)[:, None]
            idx = np.minimum((u > cum[s]).sum(1), flat.shape[1] - 1)
            toks[:, t], s = idx // self.K, idx % self.K
        return toks

    # ---------- filtering ----------
    def filter(self, toks: np.ndarray, return_all: bool = False):
        """Matrix filter. Returns beliefs after each prefix (N,T,K) if return_all else after the full context (N,K)."""
        N, T = toks.shape
        b = np.repeat(self.pi[None], N, 0)
        out = np.empty((N, T, self.K)) if return_all else None
        for t in range(T):
            a = np.einsum("ni,nij->nj", b, self.T[toks[:, t]])
            b = a / a.sum(1, keepdims=True)
            if return_all:
                out[:, t] = b
        return out if return_all else b

    def next_token_probs(self, b: np.ndarray) -> np.ndarray:
        """P(X_{T+1}=y | c) = b T[y] 1 (derivations §1.1)."""
        return b @ self.T.sum(2).T

    def bayes_predictive(self, toks: np.ndarray) -> np.ndarray:
        """Per-position Bayes predictive distributions for every prefix: (N, T, V); position t predicts x_{t+1}.

        Position 0 in the returned array is the prediction after seeing x_1, matching a causal LM's output at t=0.
        """
        B = self.filter(toks, return_all=True)
        return B @ self.T.sum(2).T

    # ---------- lagged token statistics (derivations §3.2) ----------
    def lag_moment(self, f: np.ndarray, g: np.ndarray, n: int) -> float:
        """E[f(X_t) g(X_{t+n})] under stationarity: pi M_f P^{n-1} M_g 1 for n>=1; pi (sum f g T[x]) 1 for n=0."""
        if n == 0:
            return float(self.pi @ np.einsum("x,x,xij->ij", f, g, self.T) @ np.ones(self.K))
        Mf = np.einsum("x,xij->ij", f, self.T)
        Mg = np.einsum("x,xij->ij", g, self.T)
        return float(self.pi @ Mf @ np.linalg.matrix_power(self.P, n - 1) @ Mg @ np.ones(self.K))

    def feature_lag_cov(self, phi: np.ndarray, T: int) -> np.ndarray:
        """Gamma[k,l] = Cov[phi(X_{T-k}), phi(X_{T-l})] for k,l in 0..T-1; phi is (V, r). Returns (T, T, r, r)."""
        V, r = phi.shape
        mean = np.array([self.lag_moment(phi[:, a], np.ones(V), 0) for a in range(r)])
        G = np.zeros((T, T, r, r))
        for k in range(T):
            for l in range(T):
                n = abs(k - l)
                for a in range(r):
                    for c in range(r):
                        # X_{T-k} is earlier than X_{T-l} when k > l.
                        fa, fc = (phi[:, a], phi[:, c]) if k >= l else (phi[:, c], phi[:, a])
                        G[k, l, a, c] = self.lag_moment(fa, fc, n) - mean[a] * mean[c]
        return G


# ---------------------------------------------------------------------------
# Binary factors and the two-factor product generator (plan §3.1)
# ---------------------------------------------------------------------------

def binary_transition(rho: float) -> np.ndarray:
    """P(S'=s'|S=s) = (1 + rho s s')/2 on states ordered (-1, +1)."""
    return 0.5 * (1 + rho * np.outer(SIGNS, SIGNS))


def binary_emission(q: float) -> np.ndarray:
    """E[s, x] = P(X=x|S=s) = (1 + q x s)/2."""
    return 0.5 * (1 + q * np.outer(SIGNS, SIGNS))


def magnetization_filter(x: np.ndarray, rho: float, q: float, return_all: bool = False):
    """Scalar recursion m_t = (rho m_{t-1} + q x_t)/(1 + rho q x_t m_{t-1}), m_0 = 0 (derivations §1.3).

    x: (..., T) signs. Also returns log p(x) accumulated as sum log[(1 + rho q x_t m_{t-1})/2].
    """
    m = np.zeros(x.shape[:-1])
    logp = np.zeros(x.shape[:-1])
    ms = []
    for t in range(x.shape[-1]):
        xt = x[..., t]
        logp += np.log(0.5 * (1 + rho * q * xt * m))
        m = (rho * m + q * xt) / (1 + rho * q * xt * m)
        ms.append(m)
    return (np.stack(ms, -1) if return_all else m), logp


def product_transition(rho1: float, rho2: float) -> np.ndarray:
    return np.kron(binary_transition(rho1), binary_transition(rho2))


def corr_reset_transition(c: float) -> np.ndarray:
    """P_corr(s1',s2'|s1,s2) = (1 + c s1' s2')/4, independent of the previous state (plan §4.2)."""
    row = 0.25 * (1 + c * PAIRS[:, 0] * PAIRS[:, 1])
    return np.tile(row, (4, 1))


def two_factor_hmm(rho1, q1, rho2, q2, eps: float = 0.0, c: float = 0.8, perm=None) -> HMM:
    """Four hidden states (s1,s2), four observed tokens (x1,x2) with label permutation `perm`.

    Hidden transition (1-eps) P1⊗P2 + eps P_corr; emission is the independent product of factor emissions
    from the new state. perm[k] is the token label given to observation pair PAIRS[k]; identity if None.
    """
    P = (1 - eps) * product_transition(rho1, rho2) + eps * corr_reset_transition(c)
    E1, E2 = binary_emission(q1), binary_emission(q2)
    # emit[j, k] = P(obs pair k | hidden state j)
    emit = np.array([[E1[j // 2, k // 2] * E2[j % 2, k % 2] for k in range(4)] for j in range(4)])
    perm = np.arange(4) if perm is None else np.asarray(perm)
    T = np.zeros((4, 4, 4))
    for k in range(4):
        T[perm[k]] = P * emit[:, k][None, :]
    return HMM(T, meta=dict(kind="two_factor", rho=(rho1, rho2), q=(q1, q2), eps=eps, c=c, perm=perm.tolist()))


def token_pairs(hmm: HMM) -> np.ndarray:
    """(V, 2) observation signs (x1, x2) carried by each token label."""
    perm = np.asarray(hmm.meta["perm"])
    out = np.zeros((4, 2), dtype=np.int64)
    out[perm] = PAIRS
    return out


def belief_coordinates(hmm: HMM, b: np.ndarray) -> dict:
    """Candidate belief coordinate sets for the two-factor generator (plan §3.2-3.3, §4.2).

    m: posterior magnetizations (m1, m2) = E[S_a | c]
    u: signed next-token logit coordinates atanh(rho_a q_a m_a)   (independent factors only)
    m_aug: (m1, m2, m1 m2)                                         (preregistered augmented set)
    walsh: orthonormal Walsh coordinates of the joint posterior (E[S1]/2, E[S2]/2, E[S1 S2]/2)
    interaction: E[S1 S2 | c] - m1 m2
    belief: full 4-state posterior
    """
    s1, s2 = PAIRS[:, 0], PAIRS[:, 1]
    m1, m2, m12 = b @ s1, b @ s2, b @ (s1 * s2)
    (rho1, rho2), (q1, q2) = hmm.meta["rho"], hmm.meta["q"]
    out = dict(
        m=np.stack([m1, m2], -1),
        m_aug=np.stack([m1, m2, m1 * m2], -1),
        walsh=0.5 * np.stack([m1, m2, m12], -1),
        interaction=m12 - m1 * m2,
        belief=b,
    )
    if hmm.meta["eps"] == 0:
        out["u"] = np.stack([np.arctanh(rho1 * q1 * m1), np.arctanh(rho2 * q2 * m2)], -1)
    return out


def shuffled_control(hmm: HMM) -> HMM:
    """IID tokens with the generator's one-token stationary frequencies: removes temporal prediction (plan §4.3)."""
    freq = hmm.pi @ hmm.T.sum(2)  # P(X=x) = sum_j pi T[x] 1
    T = np.zeros((hmm.V, 1, 1))
    T[:, 0, 0] = freq
    return HMM(T, pi=np.ones(1), meta=dict(hmm.meta, kind="shuffled"))


def factor_signs(toks: np.ndarray, hmm: HMM) -> np.ndarray:
    """Map token labels to observation signs: (N, T) -> (N, T, 2)."""
    return token_pairs(hmm)[toks]

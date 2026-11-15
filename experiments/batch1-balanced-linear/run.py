"""Batch 1 / Gate B (experiment plan §6; derivations Package 2): balanced two-layer linear learning. CPU only.

    uv run python experiments/batch1-balanced-linear/run.py

Inputs are exactly whitened belief coordinates z of the T=6 two-factor generator (q2 = 0.25), weighted by their exact
context probabilities, so E[z z^T] = I holds to round-off. The network h = E z, yhat = D h is trained on the
population loss 1/2 sum_c p_c ||D E z_c - M z_c||^2 by Euler steps on gradients computed from the data, and the
hidden covariance is measured from activations, C_h = Cov_c(E z_c), as an experiment would measure it.
Predictions (no fitting): aligned balanced modes follow a_i(tau) = s/(1 + (s/a0 - 1) e^{-2 s tau});
endpoint PR = (sum s)^2 / sum s^2; with factor penalty lam the endpoint spectrum is (s - lam)_+.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prt.generators import belief_coordinates, two_factor_hmm
from prt.linear import aligned_init, balanced_endpoint_pr, pr_of_modes, random_balanced_init, sigmoid_modes, whiten
from prt.spectra import eigvals_psd, pr, weighted_cov

OUT = Path(__file__).parent
h = two_factor_hmm(0.8, 0.5, 0.8, 0.25, perm=[2, 0, 3, 1])
_, p, b = h.enumerate(6)
z, W, _ = whiten(belief_coordinates(h, b)["m"], p)
assert np.allclose((z * p[:, None]).T @ z, np.eye(2), atol=1e-12)


def train(D, E, M, dt, tau_end, lam=0.0, n_rec=200):
    """Euler on data gradients. Returns tau, measured C_h eigenvalues, imbalance norm, loss."""
    steps = int(round(tau_end / dt))
    rec_at = set(np.unique(np.linspace(0, steps, n_rec).astype(int)))
    out = dict(tau=[], eig=[], imbalance=[], loss=[], pr=[])
    Y = z @ M.T
    for n in range(steps + 1):
        if n in rec_at:
            H = z @ E.T
            _, C = weighted_cov(H, p)
            lam_h = eigvals_psd(C)
            out["tau"].append(n * dt); out["eig"].append(lam_h.tolist()); out["pr"].append(pr(C))
            out["imbalance"].append(float(np.linalg.norm(E @ E.T - D.T @ D)))
            out["loss"].append(float(0.5 * p @ np.sum((H @ D.T - Y) ** 2, 1)))
        if n == steps:
            break
        R = (z @ E.T @ D.T - Y) * p[:, None]          # p_c (yhat_c - y_c)
        gD = R.T @ (z @ E.T) + lam * D                # sum_c p_c e_c h_c^T
        gE = D.T @ R.T @ z + lam * E                  # D^T sum_c p_c e_c z_c^T
        D, E = D - dt * gD, E - dt * gE
    return out


res = {}
TEACHERS = {"diag(1,0.25)": np.diag([1.0, 0.25]), "diag(1,1)": np.eye(2)}
a0, tau_end = 1e-3, 40.0
rng = np.random.default_rng(0)

# (a) aligned balanced init, Euler step refinement against the continuous-time prediction.
for tname, M in TEACHERS.items():
    s = np.linalg.svd(M, compute_uv=False)
    for d in [2, 8]:
        rows = []
        for dt in [0.1, 0.05, 0.025, 0.0125]:
            D, E = aligned_init(M, d, a0, rng=np.random.default_rng(1))
            o = train(D, E, M, dt, tau_end)
            pred = sigmoid_modes(s, a0, o["tau"])
            eig = np.array([e[:2] for e in o["eig"]])
            rows.append(dict(dt=dt, max_eig_err=float(np.abs(eig - pred).max()),
                             max_pr_err=float(np.abs(np.array(o["pr"]) - pr_of_modes(pred)).max()),
                             final_pr=o["pr"][-1], max_imbalance=max(o["imbalance"]), trace=o))
        res[f"aligned|{tname}|d{d}"] = dict(endpoint_pr_pred=balanced_endpoint_pr(M),
                                             runs=[{k: v for k, v in r.items() if k != "trace"} for r in rows],
                                             trace_finest=rows[-1]["trace"])

# (b) random small balanced init (not mode-aligned), (c) factor penalty, (d) imbalanced and generic inits.
for tname, M in TEACHERS.items():
    s = np.linalg.svd(M, compute_uv=False)
    for d in [2, 8]:
        for cond in ["random_balanced", "penalty_0.1", "imbalanced_E0.3_D0.001", "gaussian_0.01"]:
            finals = []
            for seed in range(3):
                g = np.random.default_rng(10 + seed)
                lam = 0.0
                if cond == "random_balanced":
                    D, E = random_balanced_init(2, 2, d, 0.03, g)
                elif cond == "penalty_0.1":
                    D, E = random_balanced_init(2, 2, d, 0.03, g); lam = 0.1
                elif cond == "imbalanced_E0.3_D0.001":
                    E = 0.3 * g.standard_normal((d, 2)) / np.sqrt(2); D = 1e-3 * g.standard_normal((2, d))
                else:
                    E = 0.01 * g.standard_normal((d, 2)); D = 0.01 * g.standard_normal((2, d))
                o = train(D, E, M, 0.0125, 60.0, lam=lam, n_rec=100)
                finals.append(dict(seed=seed, final_pr=o["pr"][-1], final_eig=o["eig"][-1][:3],
                                   imbalance0=o["imbalance"][0], imbalance_end=o["imbalance"][-1],
                                   final_loss=o["loss"][-1], trace=o if seed == 0 else None))
            s_eff = np.clip(s - lam, 0, None)
            pred = float(s_eff.sum() ** 2 / (s_eff**2).sum())
            res[f"{cond}|{tname}|d{d}"] = dict(endpoint_pr_pred=pred, endpoint_eigs_pred=s_eff.tolist(),
                                                runs=[{k: v for k, v in f.items() if k != "trace"} for f in finals],
                                                trace_seed0=finals[0]["trace"])

(OUT / "results.json").write_text(json.dumps(res))

# Figure: predicted vs measured PR trajectories.
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for i, tname in enumerate(TEACHERS):
    s = np.linalg.svd(TEACHERS[tname], compute_uv=False)
    tr = res[f"aligned|{tname}|d2"]["trace_finest"]
    tau = np.array(tr["tau"])
    ax[0, i].plot(tau, pr_of_modes(sigmoid_modes(s, a0, tau)), "k-", lw=3, alpha=.3, label="prediction (aligned balanced)")
    ax[0, i].plot(tau, tr["pr"], "C0--", label="measured, aligned, dt=0.0125")
    for cond, c in [("random_balanced", "C1"), ("imbalanced_E0.3_D0.001", "C3"), ("gaussian_0.01", "C2")]:
        t2 = res[f"{cond}|{tname}|d2"]["trace_seed0"]
        ax[0, i].plot(t2["tau"], t2["pr"], c, alpha=.8, label=cond)
    ax[0, i].axhline(balanced_endpoint_pr(TEACHERS[tname]), c="k", ls=":")
    ax[0, i].set_title(f"teacher {tname}, d=2: PR(C_h)"); ax[0, i].set_xlabel("tau"); ax[0, i].legend(fontsize=7)
    eig = np.array([e[:2] for e in tr["eig"]])
    ax[1, i].plot(tau, sigmoid_modes(s, a0, tau), "k-", lw=3, alpha=.3)
    ax[1, i].plot(tau, eig, "C0--"); ax[1, i].set_yscale("log"); ax[1, i].set_title("hidden covariance eigenvalues")
    ax[1, i].set_xlabel("tau")
fig.tight_layout(); fig.savefig(OUT / "pr_trajectories.png", dpi=130)

for k, v in res.items():
    print(k, "pred", round(v["endpoint_pr_pred"], 4), [{kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk in ("dt", "max_eig_err", "max_pr_err", "final_pr", "seed", "imbalance_end", "max_imbalance")} for r in v["runs"]])

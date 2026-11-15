"""Stage B: balanced linear learning reproduces the closed-form trajectories (derivations Package 2)."""
import numpy as np

from prt.linear import (aligned_init, balanced_endpoint_pr, gradient_flow, pr_of_modes, random_balanced_init,
                        sigmoid_modes, whiten)


def test_aligned_flow_matches_sigmoid_and_endpoint():
    M = np.diag([1.0, 0.25])
    D, E = aligned_init(M, d=5, a0=1e-3, rng=np.random.default_rng(0))
    _, _, rec = gradient_flow(D, E, M, dt=1e-2, steps=8000, record_every=10)  # tau = 80: weak mode saturates
    pred = sigmoid_modes([1.0, 0.25], 1e-3, rec["tau"])
    assert np.max(np.abs(rec["eig_h"][:, :2] - pred)) < 1e-8
    assert np.max(rec["imbalance"]) < 1e-12                     # exact balance is invariant
    assert abs(pr_of_modes(rec["eig_h"][-1, :2]) - balanced_endpoint_pr(M)) < 1e-6
    assert np.isclose(balanced_endpoint_pr(M), 1.25**2 / (1 + 0.0625))


def test_euler_converges_to_flow():
    M = np.diag([1.0, 0.25])
    D, E = aligned_init(M, d=2, a0=1e-2)
    errs = []
    for dt in [0.04, 0.02, 0.01]:
        _, _, rec = gradient_flow(D, E, M, dt=dt, steps=int(8 / dt), method="euler", record_every=int(8 / dt))
        errs.append(np.abs(rec["eig_h"][-1] - sigmoid_modes([1.0, 0.25], 1e-2, [8.0])[0]).max())
    assert errs[2] < errs[1] < errs[0] and errs[1] / errs[2] > 1.7  # first-order convergence


def test_weight_decay_and_unequal_rates():
    M = np.diag([1.0, 0.25])
    D, E = aligned_init(M, d=2, a0=1e-2)
    _, E1, _ = gradient_flow(D, E, M, dt=1e-2, steps=4000, lam=0.4, record_every=4000)
    assert np.allclose(np.sort(np.linalg.eigvalsh(E1 @ E1.T))[::-1], [0.6, 0.0], atol=1e-3)  # (s - lam)_+
    # Scaled-balanced init with rates (eta_D, eta_E): hidden spectrum scales uniformly, PR unchanged.
    eD, eE = 1.0, 4.0
    a0 = 1e-2
    E2 = np.diag(np.sqrt(a0 * np.sqrt(eE / eD)) * np.ones(2))
    D2 = np.diag(np.sqrt(a0 * np.sqrt(eD / eE)) * np.ones(2))
    _, Ef, rec = gradient_flow(D2, E2, M, dt=5e-3, steps=4000, eta_D=eD, eta_E=eE, record_every=4000)
    assert np.allclose(rec["eig_h"][-1], np.array([1.0, 0.25]) * np.sqrt(eE / eD), rtol=1e-4)


def test_random_balanced_init_is_balanced_and_whitening():
    rng = np.random.default_rng(4)
    D, E = random_balanced_init(2, 2, 6, 0.05, rng)
    assert np.linalg.norm(E @ E.T - D.T @ D) < 1e-14
    b = rng.standard_normal((5000, 3)) @ rng.standard_normal((3, 3))
    z, W, mu = whiten(b)
    assert np.allclose(np.cov(z.T, bias=True), np.eye(3), atol=1e-10)

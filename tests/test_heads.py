"""Package 5: analytic head gradients, onset coefficients and continuum solver (derivations §5.2-5.9, numerical check)."""
import numpy as np
import torch

from prt import heads as hd

POP = hd.Population.independent(0.7, 0.4, 0.8, 0.8, 3)


def test_onset_table_from_derivations():
    expected = {5: (2.044327685523, -0.008109837825), 9: (2.044328397154, -0.008109883412),
                17: (2.044328397168, -0.008109884767), 33: (2.044328397168, -0.008109884767)}
    for n, (pr0, prdd) in expected.items():
        o = hd.onset(*hd.quadrature_init(n), POP)
        assert abs(o["pr0"] - pr0) < 1e-11 and abs(o["prdd0"] - prdd) < 1e-11


def _torch_loss(theta, w, pop):
    """Independent autograd implementation of the same forward pass and population loss."""
    xs = torch.tensor(pop.xs); ts = torch.tensor(pop.ts, dtype=torch.float64); ds = torch.tensor(pop.ds)
    a, beta, kappa, g = theta[:, :2], theta[:, 2], theta[:, 3], theta[:, 4]
    al = torch.softmax(beta[:, None, None] * ts + kappa[:, None, None] * ds, -1)
    r = xs[:, -1] + torch.einsum("qms,msj->qmj", al, xs)
    psi = r + torch.tanh(g[:, None, None] * r)
    f = torch.einsum("q,qj,qmj->mj", torch.tensor(w), a, psi)
    return (torch.tensor(pop.p) * (torch.log(2 * torch.cosh(f)) - torch.tensor(pop.mu) * f).sum(1)).sum()


def test_velocity_equals_minus_H_times_autograd_gradient():
    rng = np.random.default_rng(0)
    th, w = hd.random_init(7, rng)
    th[:, :2] = rng.standard_normal((7, 2)); th[:, 4] += 0.3 * rng.standard_normal(7)
    t = torch.tensor(th, requires_grad=True)
    _torch_loss(t, w, POP).backward()
    assert np.abs(hd.velocity(th, w, POP) + 7 * t.grad.numpy()).max() < 1e-12
    assert abs(_torch_loss(torch.tensor(th), w, POP).item() - hd.loss(th, w, POP)) < 1e-13


def test_onset_matches_short_time_integration():
    """Symmetric difference [PR(tau) + PR(-tau) - 2 PR(0)] / tau^2 = PR''(0) + O(tau^2), as in the derivations' check.

    Negative time is only a local derivative check of the ODE, not a training run.
    """
    th, w = hd.quadrature_init(9)
    o = hd.onset(th, w, POP)
    rel = []
    for tau in [2e-3, 1e-3]:
        prs = [hd.pr_S(hd.integrate(th, w, POP, dt=sgn * tau / 4, steps=4, record_every=4, record=("S",))[1]["S"][-1])
               for sgn in (1, -1)]
        rel.append(abs((prs[0] + prs[1] - 2 * o["pr0"]) / tau**2 / o["prdd0"] - 1))
    assert rel[0] < 1e-3 and 3 < rel[0] / rel[1] < 5             # second-order differencing error


def test_loss_descends_along_flow():
    th, w = hd.quadrature_init(5)
    _, rec = hd.integrate(th, w, POP, dt=0.1, steps=50, record_every=5, record=("loss",))
    assert np.all(np.diff(rec["loss"]) < 0)

"""Checkpoint analysis for Stage D (experiment plan §2.2, §8.2, §9.2).

    python -m prt.analysis runs/x [--n_eval 65536] [--steps 0 1000 20000]

For each checkpoint and each captured site, at the final position only:
  * spectrum of C_h on all n_eval independent contexts (trace, PR, nPR, d95, normalized eigenvalues);
  * for each declared belief-coordinate set b in {m, u, m_aug, walsh}: forward regression h ≈ mu + E b fitted on the
    probe-train half and evaluated on the probe-test half, giving C_h = C_belief + C_res + C_cross there, PR and
    spectra of C_belief and C_res, the fraction of variance explained, and the trace-normalized metric G = E^T E;
  * reverse probes h -> b (held-out R^2 per coordinate), which test decodability only.
The sets last_token, m_token and u_token add a one-hot of the final token, a declared nuisance control: the
final-position residual contains that token's embedding, which alone spans up to rank V-1 = 3.
Coordinates are computed with the exact filter of the generator that produced the data; for the shuffled control
they come from the unshuffled generator applied to the same tokens (so they carry no predictive information).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .generators import belief_coordinates
from .spectra import summarize, weighted_cov
from .train import make_hmm
from .transformer import ModelConfig, Transformer

COORD_SETS = ["m", "u", "m_aug", "walsh", "last_token", "m_token", "u_token"]


def collect(model, toks, device, batch=8192):
    """Final-position activations at every site: {site: (N, d) float64}."""
    acts = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(toks), batch):
            cache = {}
            model(torch.as_tensor(toks[i:i + batch], device=device), cache)
            for k, v in cache.items():
                acts.setdefault(k, []).append(v[:, -1].double().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acts.items()}


def lstsq_affine(X, Y):
    """Fit Y ≈ c + X W by least squares (minimum-norm on rank-deficient X). Returns (c, W)."""
    Xa = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    return coef[0], coef[1:]


def spectrum_record(C, keep_eigs=True):
    s = summarize(C)
    out = {k: s[k] for k in ["trace", "pr", "npr", "d95"]}
    if keep_eigs:
        out["p"] = s["p"].tolist()
    return out


def decompose(H_tr, B_tr, H_te, B_te):
    """Forward regression from beliefs to activations with held-out covariance decomposition (plan §2.2)."""
    c, W = lstsq_affine(B_tr, H_tr)          # h ≈ c + b W, so E = W^T (d x r)
    Hhat = c + B_te @ W
    eps = H_te - Hhat
    _, C_h = weighted_cov(H_te)
    _, C_bel = weighted_cov(Hhat)
    _, C_res = weighted_cov(eps)
    hc, ec = Hhat - Hhat.mean(0), eps - eps.mean(0)
    X = hc.T @ ec / len(hc)
    C_cross = X + X.T
    E = W.T
    G = E.T @ E
    _, Sb = weighted_cov(B_tr)
    # Spectrum of Sigma_b^1/2 G Sigma_b^1/2 equals the spectrum of C_belief at the population optimum.
    lam, V = np.linalg.eigh(Sb)
    Sb_half = V @ np.diag(np.sqrt(np.clip(lam, 0, None))) @ V.T
    Gw = Sb_half @ G @ Sb_half
    trG, trGw = np.trace(G), np.trace(Gw)
    return dict(
        frac_explained=float(np.trace(C_bel) / np.trace(C_h)),
        # 1 - E||h - hhat||^2 / E||h - mean(h)||^2 on probe-test contexts (includes any held-out bias)
        r2_heldout=float(1 - np.mean(np.sum(eps**2, 1)) / np.trace(C_h)),
        cross_rel_fro=float(np.linalg.norm(C_cross) / np.linalg.norm(C_h)),
        C_belief=spectrum_record(C_bel), C_res=spectrum_record(C_res),
        G_norm_eigs=(np.linalg.eigvalsh(G)[::-1] / trG).tolist() if trG > 0 else None,
        Gw_norm_eigs=(np.linalg.eigvalsh(Gw)[::-1] / trGw).tolist() if trGw > 0 else None,
    )


def partialled_belief_block(H, B, k):
    """Belief block of a joint fit h ≈ c + E [b, onehot]: E_b = E[:, :k] with token identity partialled out.

    Returns the trace-normalized eigenvalues of G_b = E_b^T E_b, the PR of E_b Sigma_b E_b^T (Sigma_b = Cov(b)),
    and the trace of G_b. An isometric belief encoding has G_b ∝ I, so PR(E_b Sigma_b E_b^T) equals the generator-only
    isometric PR of the same coordinates. The one-hot is collinear with the intercept; lstsq returns the minimum-norm
    solution, which leaves E_b unaffected because b is not collinear with the one-hot.
    """
    _, W = lstsq_affine(B, H)
    Eb = W[:k].T                                   # (d, k)
    Gb = Eb.T @ Eb
    _, Sb = weighted_cov(B[:, :k])
    C = Eb @ Sb @ Eb.T
    tr = np.trace(Gb)
    return dict(G_norm_eigs=(np.linalg.eigvalsh(Gb)[::-1] / tr).tolist(), trace_G=float(tr),
                factor1_share=float(Gb[0, 0] / tr),           # G_b[0,0] / tr G_b: which factor dominates
                pr_C_belief=float(np.trace(C) ** 2 / np.sum(C * C)))


def reverse_probe(H_tr, B_tr, H_te, B_te):
    c, W = lstsq_affine(H_tr, B_tr)
    resid = B_te - (c + H_te @ W)
    var = B_te.var(0)
    return (1 - (resid**2).mean(0) / np.where(var > 0, var, np.nan)).tolist()


def analyze_checkpoint(model, hmm_data, hmm_coords, toks, device):
    acts = collect(model, toks[:, -model.c.ctx:], device)
    coords = belief_coordinates(hmm_coords, hmm_coords.filter(toks))
    # Token-identity nuisance control (plan §4.3, H4): one-hot of the final token, alone and alongside beliefs.
    onehot = np.eye(hmm_data.V)[toks[:, -1]]
    coords["last_token"] = onehot
    coords["m_token"] = np.hstack([coords["m"], onehot])
    if "u" in coords:
        coords["u_token"] = np.hstack([coords["u"], onehot])
    n = len(toks) // 2
    out = {}
    for site, H in acts.items():
        _, C = weighted_cov(H)
        rec = dict(C_h=spectrum_record(C))
        for name in COORD_SETS:
            if name not in coords:
                continue
            B = coords[name]
            rec[name] = decompose(H[:n], B[:n], H[n:], B[n:])
            rec[name]["reverse_r2"] = reverse_probe(H[:n], B[:n], H[n:], B[n:])
            if name in ("m_token", "u_token"):
                rec[name]["partialled"] = partialled_belief_block(H[:n], B[:n], k=2)
        out[site] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_eval", type=int, default=65536)
    ap.add_argument("--eval_seed", type=int, default=7)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. n262144 for a sample-size check")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    hmm_data = make_hmm(cfg)
    hmm_coords = make_hmm({**cfg, "shuffled": False})
    toks = hmm_data.sample(a.n_eval, cfg["ctx"], np.random.default_rng(a.eval_seed))
    model = Transformer(ModelConfig(**cfg["model"])).to(a.device)
    outdir = run / ("analysis" + (f"_{a.tag}" if a.tag else ""))
    outdir.mkdir(exist_ok=True)
    # Generator-only references on the same eval contexts (level-1 benchmarks, plan §3.2): v_a, factored and joint PR.
    co = belief_coordinates(hmm_coords, hmm_coords.filter(toks))
    _, Cm = weighted_cov(co["m"])
    _, Cw = weighted_cov(co["walsh"])
    v1, v2 = float(Cm[0, 0]), float(Cm[1, 1])
    gen_ref = dict(v1=v1, v2=v2, cov_m12=float(Cm[0, 1]), pr_factored=(v1 + v2) ** 2 / (v1**2 + v2**2),
                   pr_joint_walsh=float(np.trace(Cw) ** 2 / np.sum(Cw * Cw)))
    if "u" in co:  # isometric encoding of the readout coordinates u = atanh(rho q m) (plan §3.3)
        _, Cu = weighted_cov(co["u"])
        gen_ref["pr_u_isometric"] = float(np.trace(Cu) ** 2 / np.sum(Cu * Cu))
    ckpts = sorted((run / "ckpt").glob("step_*.pt"))
    if a.steps is not None:
        ckpts = [c for c in ckpts if int(c.stem.split("_")[1]) in set(a.steps)]
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=a.device))
        res = analyze_checkpoint(model, hmm_data, hmm_coords, toks, a.device)
        res["_meta"] = dict(step=int(ck.stem.split("_")[1]), n_eval=a.n_eval, eval_seed=a.eval_seed, generator=gen_ref)
        (outdir / f"{ck.stem}.json").write_text(json.dumps(res))
        print(ck.stem, {s: round(r["C_h"]["pr"], 3) for s, r in res.items() if s != "_meta"}, flush=True)


if __name__ == "__main__":
    main()

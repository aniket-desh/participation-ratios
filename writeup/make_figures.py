"""Figures for writeup/README.md, drawn from the committed result files of each experiment.

Style follows figures4papers (github.com/ChenLiu-1996/figures4papers, scientific-figure-making/design-theory.md):
sans font, top/right spines off, frameless legends, blue/green/red/neutral palette, tight_layout(pad=2), 300 dpi.

Run from the repository root:  uv run python writeup/make_figures.py
Outputs: writeup/figures/fig{1,2,3,4}_*.png and writeup/figures/fig1_reference_curves.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prt.generators import belief_coordinates, two_factor_hmm

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "writeup/figures"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "green_dark": "#3E8E41",
           "red_2": "#E9A6A1", "red_strong": "#B64342", "neutral": "#CFCECE", "grey": "#767676",
           "dark": "#272727", "teal": "#42949E", "violet": "#9A4D8E"}
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"], "font.size": 15,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 11.5, "svg.fonttype": "none",
    "xtick.major.width": 1.6, "ytick.major.width": 1.6, "lines.linewidth": 2.5,
})
HELD_OUT = (0.2, 0.35, 0.65, 0.9)
Q2_COLORS = {0.25: PALETTE["blue_main"], 0.5: PALETTE["green_dark"], 0.8: PALETTE["red_strong"]}


def save(fig, name):
    fig.tight_layout(pad=2)
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def label_panel(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")


# ---------------------------------------------------------------------------
# Figure 1: the locked prediction across reliabilities (Batch 4, transformer-confirm)
# ---------------------------------------------------------------------------
def reference_curves(qs, N=200_000, T=16):
    """Generator-only references at each q2 (Monte Carlo over T = 16 contexts, perm as in training).

    pr_m, pr_u: PR of the covariance of m and of u = atanh(rho q m), i.e. the belief PR under m- or u-isometry.
    share_u: factor-1 share of G_b(m) if the encoding were isometric in u. With h = E_u u + (token terms),
    regressing h on [m, onehot(x_T)] returns E_u B, B = coefficients of u on m in that same regression,
    so G_b(m) ∝ B^T B and the share is (B^T B)_11 / tr(B^T B).
    """
    pr_m, pr_u, share_u = [], [], []
    for q in qs:
        hmm = two_factor_hmm(0.8, 0.5, 0.8, q, perm=[2, 0, 3, 1])
        toks = hmm.sample(N, T, np.random.default_rng(0))
        c = belief_coordinates(hmm, hmm.filter(toks))
        for key, acc in (("m", pr_m), ("u", pr_u)):
            C = np.cov(c[key].T)
            acc.append(np.trace(C) ** 2 / np.sum(C * C))
        X = np.concatenate([c["m"], np.eye(4)[toks[:, -1]]], 1)  # the one-hot spans the intercept
        coef, *_ = np.linalg.lstsq(X, c["u"], rcond=None)
        B = coef[:2].T
        G = B.T @ B
        share_u.append(G[0, 0] / np.trace(G))
    return np.array(pr_m), np.array(pr_u), np.array(share_u)


def fig1():
    g = json.loads((EXP / "transformer-confirm/grade.json").read_text())["runs"]
    res = json.loads((EXP / "transformer-confirm/results.json").read_text())
    share = {0: {}, 20000: {}}
    for r in res:
        if r["step"] in share:
            share[r["step"]].setdefault(r["q2"], []).append(
                r["sites"]["L0.resid_post"]["m_token"]["partialled"]["factor1_share"])
    qs = np.round(np.arange(0.15, 0.951, 0.025), 3)
    pr_m, pr_u, share_u = reference_curves(qs)
    json.dump(dict(q2=qs.tolist(), pr_m_isometric=pr_m.round(4).tolist(), pr_u_isometric=pr_u.round(4).tolist(),
                   factor1_share_u_isometric=share_u.round(4).tolist()),
              open(OUT / "fig1_reference_curves.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6.0))
    a = ax[0]
    a.plot(qs, pr_m, color=PALETTE["dark"], lw=3, label="m-isometric (locked prediction)")
    a.plot(qs, pr_u, color=PALETTE["grey"], lw=2.2, ls=":", label="u-isometric")
    a.axhline(2, color=PALETTE["neutral"], lw=2.2, ls="--", label="whitened, PR = 2")
    for r in g.values():
        held = r["q2"] in HELD_OUT
        a.scatter(r["q2"] + np.random.default_rng(r["seed"]).uniform(-0.006, 0.006), r["pr_belief_m"], s=46,
                  color=PALETTE["red_strong"] if held else PALETTE["blue_main"], edgecolor="white", lw=0.6, zorder=3)
    a.scatter([], [], color=PALETTE["red_strong"], s=46, label="trained, held-out $q_2$")
    a.scatter([], [], color=PALETTE["blue_main"], s=46, label="trained, discovery $q_2$")
    a.set_xlabel("$q_2$, emission reliability of factor 2")
    a.set_ylabel("belief-component PR")
    a.set_ylim(0.95, 2.1)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10.5, columnspacing=1.0)
    label_panel(a, "a")

    b = ax[1]
    b.axhline(0.5, color=PALETTE["dark"], lw=3, label="m-isometric")
    b.plot(qs, share_u, color=PALETTE["grey"], lw=2.2, ls=":", label="u-isometric")
    for step, mk, alpha, lab in [(0, "X", 0.55, "untrained (step 0)"), (20000, "o", 1.0, "trained (step 20,000)")]:
        for q, v in sorted(share[step].items()):
            col = PALETTE["red_strong"] if q in HELD_OUT else PALETTE["blue_main"]
            b.scatter(q + np.linspace(-0.008, 0.008, len(v)), v, marker=mk, s=46 if mk == "o" else 40,
                      color=col, alpha=alpha, edgecolor="white", lw=0.6, zorder=3)
        b.scatter([], [], marker=mk, s=46, color=PALETTE["grey"], alpha=alpha, label=lab)
    b.set_xlabel("$q_2$, emission reliability of factor 2")
    b.set_ylabel("factor-1 share of $G_b$ in $m$")
    b.set_ylim(0, 1)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10.5, columnspacing=1.0)
    label_panel(b, "b")
    save(fig, "fig1_locked_prediction.png")


# ---------------------------------------------------------------------------
# Figure 2: the solvable models against their exact predictions
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))

    # (a) balanced linear learning, teacher diag(1, 0.25), aligned init a0 = 1e-3, finest Euler step
    tr = json.loads((EXP / "batch1-balanced-linear/results.json").read_text())["aligned|diag(1,0.25)|d2"]
    t = np.array(tr["trace_finest"]["tau"]); pr_meas = np.array(tr["trace_finest"]["pr"])
    s, a0 = np.array([1.0, 0.25]), 1e-3
    tt = np.linspace(0, t.max(), 400)
    a_i = s / (1 + (s / a0 - 1) * np.exp(-2 * s * tt[:, None]))
    pr_pred = a_i.sum(1) ** 2 / (a_i ** 2).sum(1)
    A = ax[0]
    A.plot(tt, pr_pred, color=PALETTE["neutral"], lw=7, label="closed form")
    A.plot(t, pr_meas, color=PALETTE["blue_main"], lw=2.2, ls="--", label="trained network")
    A.axhline(tr["endpoint_pr_pred"], color=PALETTE["dark"], lw=1.6, ls=":", label="minimum-norm endpoint")
    A.set_xlabel(r"training time $\tau$ = lr $\times$ steps"); A.set_ylabel("PR of hidden covariance")
    A.set_title("balanced linear network", fontsize=14)
    A.legend(loc="upper right", fontsize=10.5)
    label_panel(A, "a")

    # (b) learned positional decay beta: exact ODE vs SGD (3 seeds)
    sc = json.loads((EXP / "stagec-learned-filter/results.json").read_text())
    B = ax[1]
    cols = [PALETTE["teal"], PALETTE["blue_secondary"], PALETTE["violet"], PALETTE["red_strong"]]
    for (q1, d), c in zip(sc.items(), cols):
        B.plot(d["ode"]["tau"], d["ode"]["beta"], color=c, lw=2.6, label=f"$q_1$ = {q1}")
        for run in d["sgd"]:
            B.scatter(run["tau"], run["beta"], color=c, s=14, alpha=0.6, zorder=3)
    B.axhline(sc["0.5"]["beta_weak_signal"], color=PALETTE["grey"], lw=1.4, ls=":")
    B.text(395, sc["0.5"]["beta_weak_signal"] - 0.06, r"weak-signal $-\log\rho_1$", ha="right",
           fontsize=10.5, color=PALETTE["grey"])
    B.set_xlabel(r"training time $\tau$"); B.set_ylabel(r"learned decay $\beta$")
    B.set_title("positional attention: ODE (lines), SGD (dots)", fontsize=14)
    B.legend(loc="center right", bbox_to_anchor=(1.0, 0.66), fontsize=10.5)
    label_panel(B, "b")

    # (c) many-head continuum: RMS error of finite-H networks vs H
    f = json.loads((EXP / "heads-continuum/results/finite.json").read_text())
    Hs = [16, 64, 256, 1024]
    C = ax[2]
    for tau, ls in [(0, "-"), (100, "--")]:
        idx = f["tau"].index(float(tau))
        fro = [np.sqrt(np.mean([r["fro_err"][idx] ** 2 for r in f["runs"][str(H)]])) for H in Hs]
        prr = [np.sqrt(np.mean([(r["pr"][idx] - f["pr_ref"][idx]) ** 2 for r in f["runs"][str(H)]])) for H in Hs]
        C.plot(Hs, fro, "o", ls=ls, color=PALETTE["blue_main"], ms=7, label=rf"kernel, $\tau$ = {tau}")
        C.plot(Hs, prr, "s", ls=ls, color=PALETTE["red_strong"], ms=7, label=rf"PR, $\tau$ = {tau}")
    Hg = np.array([12, 1400])
    for y0 in (0.9, 0.06):
        C.plot(Hg, y0 * Hg ** -0.5, color=PALETTE["grey"], lw=1.2, ls=":")
    C.text(30, 0.9 * 30 ** -0.5 * 1.25, r"$\propto H^{-1/2}$", fontsize=11, color=PALETTE["grey"])
    C.set_xscale("log"); C.set_yscale("log")
    C.set_xlabel("number of heads $H$"); C.set_ylabel("RMS error vs continuum")
    C.set_title("many-head continuum vs finite networks", fontsize=14)
    C.set_ylim(2e-4, 0.5)
    C.legend(loc="lower left", fontsize=10, ncol=2, columnspacing=0.8)
    label_panel(C, "c")
    save(fig, "fig2_solvable_models.png")


# ---------------------------------------------------------------------------
# Figure 3: training-time behaviour to 50,000 steps (Batch 2b, transformer-longrun)
# ---------------------------------------------------------------------------
def fig3():
    runs = json.loads((EXP / "transformer-longrun/stability.json").read_text())["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))
    for key, a, ylab in [("top_m", ax[0], "top eigenvalue of $G_b$ in $m$"), ("pr_Ch", ax[1], "PR($C_h$), full residual")]:
        for q, col in Q2_COLORS.items():
            rs = [r for r in runs.values() if r["q2"] == q]
            steps = sorted(int(s) for s in rs[0][key])
            Y = np.array([[r[key][str(s)] for s in steps] for r in rs])
            for y in Y:
                a.plot(steps, y, color=col, lw=1.2, alpha=0.45)
            a.plot(steps, Y.mean(0), color=col, lw=3, label=f"$q_2$ = {q} (n = {len(rs)})")
        a.axvline(20000, color=PALETTE["grey"], lw=1.2, ls=":")
        a.set_xlabel("training step"); a.set_ylabel(ylab)
        a.set_xticks([0, 10000, 20000, 30000, 40000, 50000]); a.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])
    ax[0].axhspan(0.4, 0.6, color=PALETTE["neutral"], alpha=0.35, lw=0)
    ax[0].axhline(0.5, color=PALETTE["dark"], lw=1.2, ls="--")
    ax[0].set_ylim(0.45, 0.8)
    ax[0].text(49000, 0.465, "isotropic = 0.5; band [0.4, 0.6]", ha="right", fontsize=10.5, color=PALETTE["grey"])
    ax[1].legend(loc="upper right")
    label_panel(ax[0], "a"); label_panel(ax[1], "b")
    save(fig, "fig3_training_time.png")


# ---------------------------------------------------------------------------
# Figure 4: where in the block the geometry changes (Batches 3 + 3b, six seeds)
# ---------------------------------------------------------------------------
def fig4():
    res = json.loads((EXP / "transformer-ablations-seeds/results_combined.json").read_text())
    sites = ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"]
    archs = {"base": ("baseline", PALETTE["blue_main"]), "nomlp": ("attention-only", PALETTE["red_strong"]),
             "preln": ("pre-LN", PALETTE["green_dark"])}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for j, q in enumerate([0.25, 0.8]):
        for arch, (lab, col) in archs.items():
            rs = [r for r in res if r.get("arch") == arch and r["q2"] == q and r["step"] == 20000 and r["seed"] <= 5]
            xs, pr, sh = [], [], []
            for i, s in enumerate(sites):
                site = f"L0.{s}"
                if arch == "nomlp" and s in ("mlp_out",):
                    continue
                vals = [r["sites"][site] for r in rs if site in r["sites"]]
                if not vals:
                    continue
                xs.append(i)
                pr.append([v["C_h"]["pr"] for v in vals])
                gain = [v["m_token"]["r2_heldout"] - v["last_token"]["r2_heldout"] for v in vals]
                sh.append([v["m_token"]["partialled"]["factor1_share"] if gn >= 0.01 else np.nan
                           for v, gn in zip(vals, gain)])
            off = {"base": -0.12, "nomlp": 0.0, "preln": 0.12}[arch]
            for row, data in [(0, pr), (1, sh)]:
                D = np.array(data, dtype=float)
                m, sd = np.nanmean(D, 1), np.nanstd(D, 1, ddof=1)
                a = ax[row, j]
                a.errorbar(np.array(xs) + off, m, yerr=sd, color=col, marker="o", ms=7, lw=2.2, capsize=4,
                           label=f"{lab} (n = {D.shape[1]})")
        ax[0, j].set_title(f"$q_2$ = {q}", fontsize=15)
    ax[0, 0].set_ylabel("PR($C_h$)"); ax[1, 0].set_ylabel("factor-1 share of $G_b$ in $m$")
    for a in ax[1]:
        a.axhline(0.5, color=PALETTE["dark"], lw=1.2, ls="--")
        a.set_xticks(range(len(sites))); a.set_xticklabels(sites, rotation=20)
    ax[0, 1].legend(loc="upper left", fontsize=11)
    for a, s in zip(ax.flat, "abcd"):
        label_panel(a, s)
    save(fig, "fig4_block_sites.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()

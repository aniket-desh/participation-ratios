# Participation ratios of small transformers

This project asks whether the covariance spectrum of a trained transformer's residual stream, summarized by its participation ratio $\mathrm{PR}(C)=(\operatorname{tr}C)^2/\operatorname{tr}(C^2)$, can be predicted from the statistics of the sequence generator it was trained on together with a specified learning theory. Beliefs being decodable is not the target. The target is a prediction of a nonconstant spectrum or PR curve, made before the tested runs, that survives held-out generator conditions, seeds and widths. The primary testbed is a generator with two independent binary hidden factors and a four-token vocabulary, for which filtering, belief covariance and several learning baselines are exactly computable.

**Results so far** are synthesized in [`writeup/README.md`](writeup/README.md), which covers all nine experiments, the parts of the theory each one confirms or refutes, and the intuitions behind them.

## Documents

The three planning documents live in [`docs/`](docs/). They overlap by design: the roadmap states each calculation briefly, the derivations prove it step by step, and the experiment plan turns it into staged tests with gates. They are kept whole rather than merged so that each still reads on its own; the table below maps where each topic is treated.

| Topic | [Theory roadmap](docs/theory_roadmap.md) | [Full derivations](docs/derivations.md) | [Experiment plan](docs/experiment_plan.md) |
| --- | --- | --- | --- |
| Observable, population, PR, d95 | §2 | Conventions | §2.1 |
| Exact HMM filtering and belief covariance | §3.1–3.2 | §1.1–1.2 | §3.1, Stage A (§5) |
| Binary filter, factored vs joint (Walsh) spectra | §3.3 | §1.3–1.5 | §3.2, §3.4 table |
| Probability vs logit coordinates ($u=\operatorname{atanh}\rho q m$) | §5.3 | §2.7 | §3.3 |
| Weak-signal expansion | §3.4 | §1.4 | §7.1 |
| Belief → activation transport, metric $G=E^\top E$ | §4 | §1.6 | §2.2 |
| Balanced linear learning (Level C) | §5.1–5.2 | Package 2 | Stage B (§6) |
| Positional attention covariance and learning ODE (Level D) | §6 | Package 3 | Stage C (§7) |
| Finite nonlinear transformer gradients and covariance dynamics | — | Package 4 | Stage D (§8) |
| Context kernel ↔ PR identity | §7 | §5.7 | §9.3–9.4 |
| Many-head continuum, onset, finite-$H$ error | §9 (general Level E) | Package 5 | Stage E (§9) |
| Coupled (dependent) factors | — | §5.14 | §4.2 |
| Literature boundaries | §8, references | references | §1, references |

The derivations document contains the most detailed version of every overlapping calculation, and its notation is the reference when the documents differ. In particular, the derivations name the linear teacher $M$, whereas the roadmap and plan use $T$ for it locally.

## Code

`prt/` is a small package that keeps each equation recognizable. Every module names the section it implements.

| Module | Contents |
| --- | --- |
| [`prt/generators.py`](prt/generators.py) | Edge-emitting HMMs; exact enumeration via a prefix tree; sampling; matrix filter; scalar magnetization filter; the two-factor generator with token permutation and correlated-reset coupling; belief coordinate sets ($m$, $u$, $m$-augmented, Walsh, interaction); lagged token moments $\pi M_f P^{n-1} M_g\mathbf 1$; shuffled-token control |
| [`prt/spectra.py`](prt/spectra.py) | PR, nPR, $d_{95}$, normalized spectra, streaming moments, kernel ↔ covariance spectra, mean-free $\operatorname{tr}C$ and $\operatorname{tr}C^2$ estimators, $e_{\mathrm{PR}}$, $e_{\mathrm{spec}}$, principal angles |
| [`prt/linear.py`](prt/linear.py) | Whitening, closed-form balanced mode trajectories, minimum-norm endpoint, aligned and random balanced initializations, RK4/Euler integration of the exact two-layer ODE |
| [`prt/attention_filter.py`](prt/attention_filter.py) | Positional softmax statistic and its exact $\beta$ derivatives, imposed-filter covariance $\sum B_k\Gamma_{k\ell}B_\ell^\top$, weak-signal variances, the trainable $(w,\beta)$ ODE with exact context sums |
| [`prt/heads.py`](prt/heads.py) | The restricted many-head model of derivations §5.1 with analytic gradients, mean-field velocity, deterministic cubature continuum, finite-$H$ networks, centered kernel $S=P^{1/2}KP^{1/2}$, and the onset coefficients $A_j$, $B_u$, $\ddot K(0)$, $\ddot{\mathrm{PR}}(0)$ |
| [`prt/transformer.py`](prt/transformer.py) | The Stage D decoder-only transformer, with optional pre-LayerNorm and attention-only variants, capturing every residual site |
| [`prt/train.py`](prt/train.py) | Online next-token training with Bayes-referenced excess loss and checkpoints on the plan's log schedule |
| [`prt/analysis.py`](prt/analysis.py) | Per-checkpoint spectra at every site; held-out forward regression $h\approx c+Eb$ with $C_h=C_{\mathrm{belief}}+C_{\mathrm{res}}+C_{\mathrm{cross}}$; learned metric; reverse probes; token-identity controls |

`tests/` checks each piece against an exact reference: the plan's $T=6$ table, agreement of the two filters, Walsh eigenvalues, additive-logit next-token probabilities, prescribed encodings, kernel/covariance spectral equality, the closed-form linear trajectories, finite-difference attention derivatives, the $O(q^4)$ weak-signal error order, analytic head gradients against autograd, and the derivations' onset table ($\mathrm{PR}(0)=2.044328397168$, $\ddot{\mathrm{PR}}(0)=-0.008109884767$) to 12 digits.

```bash
uv sync
uv run pytest -q
```

## Experiments, in the plan's priority order

Each experiment has its own folder under [`experiments/`](experiments/) containing its script or definition, its results and its write-up. GPU runs go through `orch`. Their definitions are `EXPERIMENT.md`, and their raw outputs go to `runs/` on the network volume.

1. **Batch 0, generator and pipeline validation (Gate A)** — [`experiments/batch0-generator-validation`](experiments/batch0-generator-validation). CPU.
2. **Batch 1, balanced linear learning (Gate B)** — [`experiments/batch1-balanced-linear`](experiments/batch1-balanced-linear). CPU.
3. **Imposed-filter and weak-signal checks (Stage C §7.1)** — part of Batch 0.
4. **Package 5 many-head continuum: onset, weak-signal scaling, cubature convergence, finite-$H$ approach, generator sweep, controls** — [`experiments/heads-continuum`](experiments/heads-continuum). CPU.
5. **Batch 2, small-transformer discovery** ($L=1$, $d=64$, $H=2$; three reliabilities × three seeds, plus controls) — [`experiments/transformer-discovery`](experiments/transformer-discovery). GPU.
6. Later, conditional on the earlier results: Batch 3 architecture ablations (no-MLP, pre-LN), Batch 4 confirmatory sweep with the reserved held-out reliabilities $q_2\in\{0.35,0.65\}$, Batch 5 width pilot (only after the Bordelon–Chaudhry–Pehlevan parameterization is implemented and audited), Mess3, and coupled factors.

See [`HANDOFF.md`](HANDOFF.md) for the current status of each experiment and the decisions awaiting review.

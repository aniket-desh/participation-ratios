---
name: transformer-discovery
title: Batch 2: small-transformer discovery on the binary two-factor generator
status: draft
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 3
budget_usd: 8
disk_gb: 40
data:
secrets:
---

# Batch 2: small-transformer discovery on the binary two-factor generator

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

When a small nonlinear transformer (L=1, d=64, H=2, d_h=32, GELU MLP, no normalization) is trained online on the binary two-factor generator of the experiment plan (§3), how is its final-position residual covariance related to the generator's belief statistics? Specifically (plan §8.2, §11 Batch 2): (a) how much of the activation variance is explained by declared belief coordinates (b = m) versus readout coordinates (u = atanh(ρqm)) versus token identity; (b) whether the learned encoding metric G = EᵀE is close to isometric, i.e. whether PR(C_belief) equals the generator-only factored prediction (v1+v2)²/(v1²+v2²); (c) whether these are consistent across seeds at matched prediction quality. This is the first neural measurement in the project. It is worth doing now because Gates A and B are covered by the exact-reference tests in `tests/` and Batch 0/1 are running on the coordinator CPU, and the answer decides which theory level (metric prediction, readout coordinates, or nuisance content) the next work must address.

## Hypothesis

Discovery batch; hypotheses are labelled as in plan §8.2 and assessed, not assumed.
- H1 (isometric belief encoding): at the final checkpoint, `L0.resid_post`, PR(C_belief) with b = m is within 10% of the generator-only factored PR for every condition, and trace-normalized G has eigenvalues within 0.1 of 1/2. Against: e_PR > 0.1 in any condition consistently across all three seeds, or normalized-G eigenvalue gaps that vary systematically with q2.
- H1u (isometric readout-coordinate encoding): the same test with b = u against the generator-only PR of Cov(u), which differs sharply from the m prediction in the anisotropic conditions (Batch 0 exact T=6: 1.157 vs 1.606 at q2 = 0.25; 1.341 vs 1.855 at q2 = 0.8).
- H2 (readout coordinates): held-out forward-regression R² with b = u exceeds b = m by more than 0.01 and by more than twice the seed-to-seed SD in at least two of three conditions. Against: no such gap.
- H4 (nuisance content): the final token's one-hot explains a large share of `resid_post` variance; m+token explains substantially more than m alone. Reported descriptively (fractions with seed spread); no threshold is claimed as a test.
- Seed consistency: within each condition, the coefficient of variation of PR(C_h) across the three seeds is below 5%. If it is above 5% while final excess KL is matched (within 2x), that is evidence against a generator-only PR law (plan §8.2 last paragraph).

## Method

Code: `prt/` at the launch commit. Generator: two independent binary hidden factors (plan §3.1) with (ρ1,q1) = (0.8,0.5), ρ2 = 0.8, q2 ∈ {0.25, 0.5, 0.8}; four-token vocabulary with token labels permuted once by `perm_seed = 12345` (recorded in each run's `config.json`). Contexts are independent, length T = 16, started from the stationary prior.

Model (`prt/transformer.py`): decoder-only, 1 layer, d = 64, 2 heads of d_h = 32 with 1/√d_h scores, MLP width 256 with GELU, learned token and positional embeddings, untied linear readout, no dropout, no normalization. PyTorch default initialization, seed `1000·seed + 2`.

Training (`prt/train.py`): Adam, batch 256 fresh contexts per step (stream seed `1000·seed + 1`), 20,000 steps, cross-entropy on all 16 next-token targets (each context has 17 sampled tokens). Validation every 250 steps and at checkpoints on 8,192 fixed contexts (val seed 99): model CE, Bayes CE, mean KL(Bayes‖model) per position and at the final position, and the reference KL(Bayes‖uniform). Checkpoints at steps 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k (plan §10).

Learning rate: pilot on q2 = 0.5 with lr ∈ {1e-4, 3e-4, 1e-3}, pilot seed 100 (disjoint from main seeds). `select_lr.py` picks the largest lr that is stable (all validation losses finite; over the last quarter of evaluations the final-position KL never exceeds 2x its window minimum + 1e-3) and whose final mean KL is within 20% (+1e-4 nats) of the best pilot. That lr is frozen for all main and control runs.

Main sweep: q2 ∈ {0.25, 0.5, 0.8} × seeds {0, 1, 2} (nine runs).

Analysis (`prt/analysis.py`), every checkpoint, final position only, n_eval = 65,536 independent contexts (eval seed 7) at sites `L0.resid_pre` (embedding), `attn_out`, `resid_mid`, `mlp_out`, `resid_post`: spectrum of C_h (trace, PR, nPR, d95, normalized eigenvalues) on all contexts; for coordinate sets m, u, m_aug = (m1,m2,m1m2), walsh, last_token (one-hot), m_token, u_token: forward least-squares regression h ≈ c + E b fitted on the first 32,768 contexts and evaluated on the other 32,768, giving held-out R², explained-trace fraction, C_belief/C_res PR, the relative Frobenius size of the cross term C_cross, trace-normalized G = EᵀE and Σ_b^{1/2} G Σ_b^{1/2}; reverse probe h → b held-out R². Generator-only references (v1, v2, factored PR, joint Walsh PR, isometric-u PR) are computed on the same eval contexts. All coordinate sets were declared before any trained model was analyzed (plan §3.3).

Evaluation-sample check (plan §9.4): final checkpoint of each main run re-analyzed with n_eval = 16,384 and 262,144.

## Inputs

No datasets or model weights: all data are sampled online from the generator in `prt/generators.py`. `data` is empty.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
bash $EXP_DIR/run.sh
```

`run.sh` runs, in order: the three pilots (4 concurrent processes), `select_lr.py` → `$EXP_DIR/lr.json`, the nine main and three control runs at the selected lr (4 concurrent), `python -m prt.analysis <run> --n_eval 65536` for every run, the n_eval 16,384/262,144 check on the main runs' step 20,000, then `summarize.py` → `$EXP_DIR/results.json` and `plot.py` → `$EXP_DIR/figures/`. It skips runs whose final checkpoint already exists, so it can be rerun after an interruption.

## Expected outputs

- `runs/transformer-discovery/<run>/` (data dir, about 3–5 MB per run, ~70 MB total): `config.json`, `metrics.jsonl`, `ckpt/step_*.pt`, `analysis/step_*.json`, `analysis_n16384/`, `analysis_n262144/`.
- `$EXP_DIR/lr.json`, `$EXP_DIR/results.json` (a few MB), `$EXP_DIR/figures/{trajectories,variance_explained,metric}.png`, `$EXP_DIR/logs/*.log`.
- `$EXP_DIR/README.md`: the write-up, written on the coordinator after collection.

## Controls

- Bayes reference: excess loss is reported against the exact finite-context Bayes predictor, and normalized by KL(Bayes‖uniform) so conditions with different predictable signal are comparable.
- Step-0 checkpoints: every observable at initialization, so training-induced structure is separated from what random embeddings already give.
- Embedding site `L0.resid_pre`: the final token's embedding plus a constant position vector, which shows the token-identity component directly.
- Token-identity coordinate sets (last_token, m_token, u_token): nuisance control for H4 (plan §4.3).
- Shuffled-token control (`ctrl_shuffled_s0`): IID tokens with the generator's single-token frequencies, so no temporal prediction is possible; beliefs for analysis come from the unshuffled filter applied to the same tokens. Belief coordinates should explain almost nothing beyond token identity.
- Token-permutation control (`ctrl_perm777_q0.25_s0`, `ctrl_perm777_q0.8_s0`): a different token labelling; conclusions should not change.
- Held-out evaluation: regressions are fitted and evaluated on disjoint contexts, and C_cross is reported rather than assumed zero (plan §2.2).
- Evaluation-sample convergence at 16k/65k/262k contexts.

## Success and failure criteria

The run **succeeded** (produced an answer) if all nine main runs finish 20,000 steps with finite losses, final-position excess KL below 10% of KL(Bayes‖uniform) at the final position in every main run, and every analysis file exists. Then each hypothesis is graded as stated under Hypothesis, at step 20,000, site `L0.resid_post`, n_eval 65,536:
- H1 **supported** if e_PR(PR(C_belief, m), factored PR) < 0.1 and both normalized-G eigenvalues are within 0.1 of 0.5 in all nine main runs; **not supported** if e_PR > 0.1 in all three seeds of at least one condition; otherwise **inconclusive**.
- H1u graded by the same rule with b = u and the generator-only PR of Cov(u) (`pr_u_isometric`).
- H2 **supported** / **not supported** by the R² gap rule above.
- Seed consistency: CV(PR(C_h)) < 5% in each condition, or not.
- Measurement validity: PR(C_h) at n_eval 65,536 within 1% of the 262,144 value at the final checkpoint for every main run; otherwise the PR claims are marked as sample-limited.

The run **failed** if training diverges for every pilot lr, or main runs cannot reach the 10% excess-KL bar (then the results are reported as an under-trained discovery batch, labelled inconclusive, not as a test of the hypotheses).

All results are level-1 (specified-metric benchmark) or level-2 (conditional reconstruction) in the plan's §2.3 sense; none is an independent learning prediction.

## Time and cost

CPU smoke test of the whole driver on the coordinator (40 steps, n_eval 2,048) took 2.6 min. On the coordinator CPU a full 20,000-step run takes about 10 min; on a GPU with 4 concurrent processes, 15 runs are expected to take 20–40 min, analysis about 15–20 min, environment setup about 5 min: roughly 1 hour. Cap 3 h. Worst case at the cap: A40 $0.59/hr → $1.8; A100 PCIe fallback $2.09/hr → $6.3, under the $8 budget.

## If something fails

- A training process errors: read its log in `$EXP_DIR/logs/`, fix only a plumbing bug (paths, device), rerun `run.sh` (completed runs are skipped). Any change to `prt/` is committed on the coordinator and pushed to the worker with `orch push`, and noted under Amendments.
- All pilots unstable or none meets the rule: stop and record as failed; do not pick a learning rate by hand.
- Main runs miss the 10% excess-KL bar: do not extend training inside this experiment; record as inconclusive and propose an extension (plan §8.1 allows extending only after documenting why).
- Time running out: the analysis of the main runs takes priority over the controls and the sample-size check; publish what exists.
- Out of memory: reduce `PAR` to 2 and rerun.

## Questions asked before launch

None asked: the user was away and gave a standing go-ahead for the experiments in the plan within $150 total and $25 per worker. Decisions made in their absence: the lr selection rule above, the choice of controls (shuffled, one alternative permutation on two conditions), adding token-identity coordinate sets as a declared control, and running Batch 2 while Batch 1 runs on the coordinator CPU (Gate A/B exact checks already pass in `tests/`).

## Review

<fill>

## Amendments

None.

## Outcome

Not run yet.

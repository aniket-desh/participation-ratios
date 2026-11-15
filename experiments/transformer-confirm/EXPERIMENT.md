---
name: transformer-confirm
title: Batch 4: locked prediction of the belief-component spectrum on held-out reliabilities
status: draft
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 3
budget_usd: 8
disk_gb: 40
data:
secrets:
---

# Batch 4: locked prediction of the belief-component spectrum on held-out reliabilities

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

Does a prediction fixed before training, using only the generator, give the spectrum of the belief component of a trained transformer's final residual at generator conditions never used to form it? The prediction is that the token-partialled belief encoding is isometric in the probability coordinates m, so the PR of the belief component equals the generator-only curve (v1+v2)²/(v1²+v2²). That curve is nonconstant in q2: 1.620, 1.882, 2.000, 1.948 and 1.858 at q2 = 0.25, 0.35, 0.5, 0.65 and 0.8 (T = 16, Batch 0).

This is the plan's "first useful milestone" (§13): a generator sweep, a declared activation location, a prediction made before training, and held-out conditions (plan §10 and §11 Batch 4). In the plan's §2.3 terms it is a **calibrated prediction**, not a parameter-free one. The isotropy-in-m rule was found in Batch 2 (`experiments/transformer-discovery/`, `grade.json`): 6 of 6 anisotropic runs closer to isotropic in m than in u, and 8 of 9 runs in the isotropy band. That rule is now locked and tested on two reliabilities reserved since Batch 0, q2 = 0.35 and 0.65, and on new seeds at the three discovery reliabilities. No per-condition quantity is fitted.

What is predicted is the belief component. The full covariance PR(C_h) is not predicted: it also contains the final token's embedding, a rank-3 nuisance, and it is reported descriptively.

## Hypothesis

Locked in `predictions.json` before any Batch 4 run. All at site `L0.resid_post`, step 20,000, n_eval 65,536.
- **P1, geometry.** The token-partialled belief metric G_b (b = m) is near-isotropic: both normalized eigenvalues lie in [0.4, 0.6].
- **P2, PR curve.** PR(E_b Σ_m E_bᵀ) of the partialled belief block equals the generator-only isometric-in-m PR on the same eval contexts within 10% (e_PR < 0.1), at each of the five reliabilities. **P2-heldout** is the same rule restricted to q2 = 0.35 and 0.65 and is the headline test.
- **P3, m over u.** In the four anisotropic conditions the partialled metric is closer to isotropic in m than in u. The margin is small at the held-out conditions: if G_m were exactly isotropic, the predicted u top eigenvalue would be about 0.67 at q2 = 0.35 and 0.63 at 0.65, against about 0.80 and 0.73 at 0.25 and 0.8.

For context only (not part of the criteria): Batch 2's three-seed means were 1.553, 1.992 and 1.741 at q2 = 0.25, 0.5 and 0.8 against predictions of 1.619, 2.000 and 1.855, so the observed values sat 0–7% below the prediction. Its one outlier seed (q2 = 0.8, seed 2: top eigenvalue 0.615, e_PR 0.12) is the reason the rules below are per-condition majorities, not "every run".

## Method

Code: `prt/` at the launch commit, unchanged from Batches 2 and 3. Training and analysis as in Batch 2's Method (`experiments/transformer-discovery/EXPERIMENT.md`), except as stated here.

- Generator: two independent binary factors, (ρ1, q1) = (0.8, 0.5), ρ2 = 0.8, q2 ∈ {0.25, 0.35, 0.5, 0.65, 0.8}, token permutation seed 12345, T = 16, independent contexts from the stationary prior.
- Model: the Batch 2 baseline (L = 1, d = 64, 2 heads × 32, GELU MLP width 256, no normalization), PyTorch default init with seed 1000·seed + 2.
- Training: Adam with lr = 1e-3, Batch 2's selected lr (`experiments/transformer-discovery/lr.json`), fixed and not re-piloted. Batch 256 online contexts, 20,000 steps, checkpoints on the plan's schedule.
- Seeds 10–14 at every reliability: 25 runs, disjoint from Batch 2's seeds 0–2 and pilot seed 100. The plan asks for at least five seeds for confirmation (§10).
- Analysis (`prt.analysis`) at steps 0, 100, 1,000, 5,000 and 20,000, n_eval 65,536, eval seed 7, all sites and coordinate sets, including the token-partialled belief block. The generator-only reference PR is computed on the same eval contexts.
- Grading by `grade.py` against `predictions.json`. A **passing run** has a tail-mean final-position KL (evaluations at steps ≥ 15,000) below its own k = 4 short-window floor, and a belief-content gain Δ = [R²(m_token) − R²(last_token)] at step 20,000 minus at step 0 that exceeds 0.02. These are the Batch 2 rules as amended there.

## Inputs

None: data are generated online. `data` is empty.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
bash $EXP_DIR/run.sh
```

`run.sh` trains the 25 runs (4 concurrent; a crashed run does not stop the driver) and analyzes them, held-out conditions first. It then runs `summarize.py` → `results.json`, `grade.py` → `grade.json`, and `plot.py` → `figures/pr_curve.png`. Completed runs are skipped on a rerun.

## Expected outputs

- `runs/transformer-confirm/q{q2}_s{seed}/`: config, metrics, checkpoints and analysis, about 3–5 MB each.
- `$EXP_DIR/results.json`, `$EXP_DIR/grade.json`, `$EXP_DIR/figures/pr_curve.png`, `$EXP_DIR/logs/`, and `$EXP_DIR/README.md` (the write-up).

## Controls

- Step-0 analysis of every run. Belief content must grow beyond its initial value (Δ > 0.02), so the untrained near-isometry of random embeddings cannot pass P1.
- The token-identity control: the belief block is partialled against the final token's one-hot.
- Short-window floors (k = 1–4) as the prediction-quality bar.
- The u-coordinate alternative (P3), which predicts a different PR curve (1.16, 1.51, 2.00, 1.68 and 1.34 at T = 6), so the test can discriminate.
- New seeds at the discovery reliabilities, which separate replication from out-of-sample generalization.

## Success and failure criteria

A reliability counts toward a verdict only if at least 2 of its 5 runs pass. If any of the five reliabilities (for P2-heldout, either held-out reliability) has fewer than 2 passing runs, that verdict is **inconclusive**. Statistics below are over passing runs.
- **P1 supported** if, at every reliability, at least 80% of passing runs are in the [0.4, 0.6] band and the seed-mean top eigenvalue is ≤ 0.6. **Not supported** if any reliability has a seed-mean top eigenvalue above 0.6 or fewer than 50% of passing runs in the band. Otherwise inconclusive.
- **P2 supported** if, at every reliability, the e_PR of the seed-mean belief PR against the prediction is below 0.1 and at least 80% of passing runs have e_PR < 0.1. **Not supported** if any reliability has a seed-mean e_PR ≥ 0.1. Otherwise inconclusive. **P2-heldout** applies the same rule to q2 = 0.35 and 0.65 only.
- **P3 supported** if, at each of q2 = 0.25, 0.35, 0.65 and 0.8, at least 80% of passing runs are closer to isotropy in m than in u. **Not supported** if any of them has fewer than 50%. Otherwise inconclusive.
- **Reported, not graded:** whether the observed seed-mean ordering of the five reliabilities matches the predicted ordering; the seed-mean e_PR with its seed SD at each reliability; the full PR(C_h) against q2; belief-PR trajectories at steps 0, 100, 1,000, 5,000 and 20,000; the single-final-evaluation bar alongside the tail-mean bar.

The run **failed** if fewer than 20 of the 25 runs finish, or if analysis files are missing for either held-out reliability.

## Time and cost

Batch 2's worker took about 230 s per 20k-step run with 4 concurrent (from the Batch 3 reviewer's reading of its logs) and about 13.5 s per checkpoint of analysis. So: 25 runs ≈ 25 min of training, 125 checkpoint analyses ≈ 28 min, setup ≈ 5 min, about 1 h in total. The CPU smoke test of the whole driver (40 steps, n_eval 2,048) took 4.7 min. Cap 3 h. Worst case at the cap: $1.8 on an A40 ($0.59/hr), $6.3 on the A100 PCIe fallback, under the $8 budget. The night's committed worst case, Batches 2–4, is $26 of $150.

## If something fails

- A training process errors: read its log, fix only a plumbing bug, rerun `run.sh` (completed runs are skipped). Code changes are pushed with `orch push` and recorded under Amendments.
- Time running out: analysis of the held-out reliabilities comes first (as ordered in `run.sh`). If needed, the analysis steps can be cut to `ASTEPS="0 20000"` (record as an amendment).
- Out of memory: reduce `PAR` to 2.
- Do not change `predictions.json` or the criteria after any Batch 4 result has been seen.

## Questions asked before launch

None asked: the user was away and gave a standing go-ahead for the plan's experiments within $25 per worker and $150 total, which covers this definition (approved under it). Decisions made in their absence:
- Locking the m-isometry rule from Batch 2 as a calibrated prediction.
- Using Batch 2's selected lr without re-piloting.
- New seeds 10–14 at all five reliabilities.
- Per-condition majority rules (80% of passing runs) rather than "every run", because of Batch 2's one outlier seed.

## Review

<fill>

## Amendments

None.

## Outcome

Not run yet.

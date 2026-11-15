---
name: transformer-ablations
title: Batch 3: attention-only and pre-LayerNorm ablations of the small transformer
status: draft
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 4
budget_usd: 10
disk_gb: 40
data:
secrets:
---

# Batch 3: attention-only and pre-LayerNorm ablations of the small transformer

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

In the small transformer of the experiment plan (§8.1: L = 1, d = 64, 2 heads of 32, GELU MLP of width 256, learned token and position embeddings, untied readout), where in the block does the final-position representation acquire or change its belief geometry? Removing the MLP (attention only) or adding pre-LayerNorm changes the map from beliefs to activations. This experiment measures, site by site, how each change alters (i) the belief-specific content beyond token identity, (ii) the token-partialled belief metric and which factor dominates it, and (iii) PR of the full covariance. This is plan Batch 3 ("identify where covariance departures arise"), and it localizes which part of the block a later theory must close (plan H3, roadmap §6.2 step 5).

The conditions are fixed in advance at the two anisotropic reliabilities, q2 ∈ {0.25, 0.8}. There, Batch 0's exact calculation shows that the isometric-in-m and isometric-in-u predictions differ most (T = 6 PR 1.606 vs 1.157 and 1.855 vs 1.341), so a change of metric is most visible. The plan says to pick conditions after Batch 2. This choice is made beforehand so the two batches can run in parallel, and the reason is recorded under "Questions asked before launch".

## Hypothesis

Descriptive batch. Plan H3 (attention and MLP have distinct predicted geometries) cannot be tested as a prediction here, because no constrained-update calculation for this architecture exists yet. What can be assessed is whether the geometry is **architecture-dependent** at all:
- **A1, MLP dependence.** Removing the MLP changes the final residual's belief geometry: at `resid_post`, the seed-mean factor-1 share of the token-partialled belief metric (in m or in u) differs between attention-only and baseline by more than 0.1, or PR(C_h) differs by more than 5%, in either condition. Against: neither difference exceeds those sizes, or both lie within twice the larger seed SD.
- **A2, normalization dependence.** The same comparison for pre-LayerNorm vs baseline, and within the pre-LN model for `ln2` vs `resid_mid` (the normalized MLP input vs the raw residual it comes from).
- **A3, MLP-stage change within the baseline.** In the baseline model, `resid_post` differs from `resid_mid` in partialled factor-1 share by more than 0.1, which would mean the MLP reshapes the belief geometry that attention wrote.

## Method

Code: `prt/` at the launch commit, the same training and analysis as Batch 2 (`experiments/transformer-discovery/EXPERIMENT.md`, Method) except as stated here.

Generator: two independent binary factors, (ρ1, q1) = (0.8, 0.5), ρ2 = 0.8, q2 ∈ {0.25, 0.8}, token permutation seed 12345, T = 16, independent contexts from the stationary prior.

Architectures (all otherwise identical; PyTorch default init, seed 1000·seed + 2):
- `base`: the Batch 2 baseline (no normalization, with MLP).
- `nomlp`: attention only (`--no_mlp`): residual = embedding + attention output, then the readout.
- `preln`: pre-LayerNorm before attention and before the MLP (`--norm pre`), with the standard affine LayerNorm (ε = 1e-5). Its extra sites `L0.ln1` and `L0.ln2` are measured too.

Learning rate: each architecture gets its own pilot on q2 = 0.5 with lr ∈ {1e-4, 3e-4, 1e-3} (pilot seed 100) and the same declared selection rule as Batch 2 (`select_lr.py`: the largest stable lr whose final mean KL is within 20% + 1e-4 nats of that architecture's best pilot). The architectures are compared at their own selected lr, not a shared one, because a shared lr would favor whichever architecture it suits.

Main runs: 3 architectures × q2 ∈ {0.25, 0.8} × seeds {0, 1, 2} = 18 runs. Adam, batch 256, 20,000 steps, online data, checkpoints on the plan's log schedule. The baseline arm repeats Batch 2's configuration at those two conditions, so this experiment is self-contained. Its lr pilot is independent and may select a different lr from Batch 2's.

Analysis: `python -m prt.analysis` on every checkpoint of the 18 main runs, n_eval = 65,536 (eval seed 7), final position, all sites, with all coordinate sets including the token-partialled belief block (`m_token`, `u_token` → `partialled`: normalized E_bᵀE_b, `factor1_share`, PR(E_b Σ_b E_bᵀ)). The partialled metric is read only at sites where beliefs add at least 0.01 held-out R² beyond the final token (R²(m_token) − R²(last_token) ≥ 0.01); elsewhere it is reported as undefined, because the fit there is noise (at `resid_pre` the final-position activation is a pure function of the last token).

## Inputs

None: data are generated online by `prt/generators.py`. `data` is empty.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
bash $EXP_DIR/run.sh
```

`run.sh` runs the nine pilots (3 archs × 3 lrs; 4 concurrent), the per-architecture lr selection → `$EXP_DIR/lr_{base,nomlp,preln}.json`, the 18 main runs (4 concurrent), analysis of the main runs, `summarize.py` → `$EXP_DIR/results.json`, and `plot.py` → `$EXP_DIR/figures/sites_q{0.25,0.8}.png`. Completed runs are skipped on a rerun.

## Expected outputs

- `runs/transformer-ablations/<run>/`: config, metrics, checkpoints and analysis per run, about 3–5 MB each, about 120 MB total.
- `$EXP_DIR/lr_*.json`, `$EXP_DIR/results.json`, `$EXP_DIR/figures/`, `$EXP_DIR/logs/`.
- `$EXP_DIR/README.md`, the write-up.

## Controls

- Each architecture's own step-0 checkpoint, separating trained structure from random-embedding structure.
- A baseline arm trained in the same experiment with the same seeds, conditions and analysis.
- Per-architecture lr selection by a fixed rule, so no architecture is handicapped by a shared lr.
- Short-window floors (`kl_lastk_final`, k = 1–4) as the prediction-quality bar.
- The token-identity control and the R² gate on the partialled metric.

## Success and failure criteria

The run **succeeded** if all 18 main runs finish with finite losses and have analysis files.

Prediction-quality bar: a main run passes if its final-position KL(Bayes‖model) is below its own `kl_lastk_final[4]`. An architecture–condition cell with fewer than two passing seeds is reported, but excluded from the A1–A3 comparisons, which are then inconclusive for that cell. Whether `nomlp` can reach the bar at all is itself a reported result.

Grading, at step 20,000, seed means over passing runs, site `L0.resid_post` unless stated:
- **A1 supported** (MLP-dependent geometry) if, in at least one condition, |Δ factor-1 share| > 0.1 (m or u, where defined) or |ΔPR(C_h)|/PR > 5% between `nomlp` and `base`, with the difference also larger than twice the larger of the two cells' seed SDs. **Not supported** if, in both conditions, all of these differences are below those sizes. Otherwise **inconclusive**.
- **A2** graded the same way for `preln` vs `base`, and separately for `ln2` vs `resid_mid` within `preln`.
- **A3** graded the same way for `resid_post` vs `resid_mid` within `base`.
- Also reported: PR(C_h) and the belief-specific R² gain at every site for every run, the selected lr per architecture, and final excess KL against the k = 1–4 floors.

The run **failed** if every pilot of an architecture is unstable (then that architecture's arm is reported as failed and the others proceed), or if no main-run analysis exists.

## Time and cost

The CPU smoke test of the whole driver (20 steps, n_eval 2,048, 27 runs) took 5.5 min on the coordinator. Batch 2's 15 runs are expected to take about 1 hour on one GPU, so these 27 runs (9 pilots + 18 main) should take about 1.5–2 hours, including analysis and about 5 min of setup. Cap 4 h. Worst case at the cap: $2.4 on an A40 ($0.59/hr), $8.4 on the A100 PCIe fallback ($2.09/hr), under the $10 budget. Together with Batch 2 ($8 worst case), the night's worst-case committed spend is $18 of the $150 allowed.

## If something fails

- A training process errors: read its log in `$EXP_DIR/logs/`, fix only a plumbing bug, rerun `run.sh` (completed runs are skipped). Code changes go through the coordinator, are pushed with `orch push`, and are recorded under Amendments.
- An architecture's pilots are all unstable: record that arm as failed, continue with the others, and do not choose an lr by hand.
- Main runs miss the KL bar: report as is. Do not extend training inside this experiment.
- Time running out: analysis of the main runs at step 20,000 takes priority. If needed, rerun analysis with `--steps 0 20000` only (amendment).
- Out of memory: reduce `PAR` to 2.

## Questions asked before launch

None asked: the user was away and gave a standing go-ahead for the plan's experiments within $25 per worker and $150 total, which covers this definition. Decisions made in their absence: fixing the conditions at q2 ∈ {0.25, 0.8} before Batch 2 finished (reason above), selecting lr per architecture instead of reusing Batch 2's, including a baseline arm so the batch is self-contained, and grading architecture dependence descriptively because no constrained-update prediction exists yet.

## Review

<fill>

## Amendments

None.

## Outcome

Not run yet.

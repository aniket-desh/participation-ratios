---
name: transformer-ablations
title: Batch 3: attention-only and pre-LayerNorm ablations of the small transformer
status: done
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 4
budget_usd: 10
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-15T21:38:28Z
result: supported
finished_utc: 2026-11-15T23:20:37Z
writeup: experiments/transformer-ablations/README.md
---

# Batch 3: attention-only and pre-LayerNorm ablations of the small transformer

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

In the small transformer of the experiment plan (§8.1: L = 1, d = 64, 2 heads of 32, GELU MLP of width 256, learned token and position embeddings, untied readout), where in the block does the final-position representation acquire or change its belief geometry? Removing the MLP (attention only) or adding pre-LayerNorm changes the map from beliefs to activations. This experiment measures, site by site, how each change alters (i) the belief-specific content beyond token identity, (ii) the token-partialled belief metric and which factor dominates it, and (iii) PR of the full covariance. This is plan Batch 3 ("identify where covariance departures arise"), and it localizes which part of the block a later theory must close (plan H3, roadmap §6.2 step 5).

The conditions are fixed in advance at the two anisotropic reliabilities, q2 ∈ {0.25, 0.8}. There, Batch 0's exact calculation shows that the isometric-in-m and isometric-in-u predictions differ most (T = 6 PR 1.606 vs 1.157 and 1.855 vs 1.341), so a change of metric is most visible. The plan says to pick conditions after Batch 2. This choice is made beforehand so the two batches can run in parallel, and the reason is recorded under "Questions asked before launch".

## Hypothesis

Descriptive batch. Plan H3 (attention and MLP have distinct predicted geometries) cannot be tested as a prediction here, because no constrained-update calculation for this architecture exists yet. What can be assessed is whether the geometry is **architecture-dependent** at all:
Each comparison gets **two separate verdicts**, one for the token-partialled belief metric (factor-1 share of normalized E_bᵀE_b, in m and in u, read only where beliefs add ≥ 0.01 R² beyond the final token) and one for PR(C_h) of the full covariance. They are kept apart because the MLP can add belief-unrelated variance that moves PR(C_h) without changing the belief metric, which is what the question is about.
- **A1, MLP dependence.** Attention-only vs baseline, at `resid_post`.
- **A2, normalization dependence.** Pre-LN vs baseline at `resid_post`, and, within the pre-LN model, `ln2` vs `resid_mid` (the normalized MLP input vs the raw residual it comes from). This `--norm pre` normalizes the branch inputs only. There is no final LayerNorm before the readout, unlike standard pre-LN transformers.
- **A3, MLP-stage change within the baseline.** `resid_post` vs `resid_mid` in the baseline model, which asks whether the MLP reshapes the belief geometry that attention wrote.
The rules that turn these into verdicts are under "Success and failure criteria".

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

`run.sh` runs the nine pilots (3 archs × 3 lrs; 4 concurrent; a diverged pilot does not stop the driver), the per-architecture lr selection → `$EXP_DIR/lr_{base,nomlp,preln}.json` (`"lr": null` if no pilot is stable, and that arm is skipped), the 18 main runs (4 concurrent), analysis of every checkpoint of the main runs, analysis of the pilots at steps 0 and 20,000, `summarize.py` → `$EXP_DIR/results.json`, and `plot.py` → `$EXP_DIR/figures/sites_q{0.25,0.8}.png`. Completed runs are skipped on a rerun.

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

**Prediction-quality bar.** A main run passes if the mean of its final-position KL(Bayes‖model) over the last quarter of evaluations (steps 15,000–20,000, `kl_final_tailmean` in `results.json`) is below its own `kl_lastk_final["4"]`. A single final evaluation is not used: in Batch 2's live baseline runs at q2 = 0.8 the final-position KL ranged 0.0006–0.0017 over steps 15k–20k, against a k = 4 floor of 0.00132 (reviewer check). An architecture–condition cell with fewer than two passing seeds is reported, but its comparisons are inconclusive. Whether `nomlp` can reach the bar at all is itself a reported result.

**Matched prediction quality.** For every comparison, report the ratio of the two cells' seed-mean tail-mean excess KL. If the ratio exceeds 2, any difference found in that comparison is labelled "confounded with prediction quality" instead of "supported", because a worse predictor may simply encode differently.

**Verdict rule**, applied separately to (i) the partialled belief metric (m and u factor-1 share; the larger |Δ| of the two decides) and (ii) PR(C_h), at step 20,000, using seed means over passing runs. Δ is the difference between the two cells, and SD is the larger of the two cells' seed SDs:
- **Supported** if, in at least one condition, |Δ| exceeds its threshold (0.1 for factor-1 share, 5% relative for PR(C_h)), |Δ| > 2·SD, |Δ| is larger than the same difference at step 0 (where the step-0 value is undefined under the R² gate, as it is for the belief metric at initialization, this condition is met), and the excess-KL ratio is at most 2.
- **Not supported** if, in both conditions, |Δ| is below its threshold and 2·SD is also below the threshold, so a shift of the threshold size would have been detectable.
- **Inconclusive** otherwise, including whenever seed noise is too large to rule out a threshold-sized shift.
- For A1 and A3, same-seed runs share their initial embedding and attention weights (A1) or are the same model (A3). The per-seed paired differences are reported alongside the unpaired rule as a tighter descriptive check. Verdicts use the unpaired rule.

Also reported: every graded Δ at step 0 next to step 20,000; PR(C_h) and the belief-specific R² gain at every site for every run; the selected lr per architecture; final excess KL against the k = 1–4 floors. Where the selected lrs differ between compared architectures, also each architecture's within-pilot lr effect (pilots are analyzed at steps 0 and 20,000), so lr and architecture effects can be compared.

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

None asked: the user was away and gave a standing go-ahead for the plan's experiments within $25 per worker and $150 total, which covers this definition (approved under it). Decisions made in their absence: fixing the conditions at q2 ∈ {0.25, 0.8} before Batch 2 finished (reason above), selecting lr per architecture instead of reusing Batch 2's, including a baseline arm so the batch is self-contained, and grading architecture dependence descriptively because no constrained-update prediction exists yet.

## Review

First review (experiment-reviewer): no P0.
- P1, the bar relied on one noisy final KL reading, and prediction quality was unmatched across architectures. **Fixed:** the bar uses the tail mean over the last quarter of evaluations, and a comparison with an excess-KL ratio above 2 is labelled "confounded with prediction quality".
- P1, seed noise produced "not supported". **Fixed:** "not supported" now also requires 2·SD to be below the threshold. Everything else is inconclusive, with the same wording in the Hypothesis and criteria sections.
- P1, the headline could rest on PR(C_h) alone. **Fixed:** two separate verdicts per comparison (belief metric; full PR).
- P2, one crashing pilot would stop the driver. **Fixed:** `|| true` on the training stages, and `select_lr.py` writes `lr: null` when no pilot is stable or finished, so that arm is skipped.
- P2, lr was confounded with architecture. **Fixed:** pilots are analyzed at steps 0 and 20,000, and the within-pilot lr effect is reported where selected lrs differ.
- P2, step-0 differences were unused. **Fixed:** a supported Δ must exceed its step-0 value, and both are reported.
Re-check: all P1s and P2s confirmed fixed; one P2, the step-0 condition was undefined for the gated belief metric. **Fixed** with an explicit clause.
- Notes adopted: paired per-seed differences are reported, and the write-up must state that `--norm pre` has no final LayerNorm. Reviewer timing from Batch 2's live worker (220–240 s per 20k-step run with 4 concurrent; about 13.5 s analysis per checkpoint) puts this batch at about 1.5–2 h, inside the 4 h cap.

## Amendments

None.

## Outcome

**Supported**: the geometry is architecture-dependent, and the MLP stage is where it changes. Write-up: [README.md](README.md).
- **Run succeeded:** all 18 main runs finished with finite losses and analysis files; all 9 pilots were stable, and every architecture selected lr = 1e-3.
- **Prediction-quality bar:** passed by 3/3 seeds in every cell except attention-only at q2 = 0.8 (0/3; tail-mean KL 0.0025–0.0027 against a 4-token floor of 0.00132).
- **A1, attention-only vs baseline:** belief metric inconclusive (cell missing at q2 = 0.8); PR(C_h) supported (+0.66 at q2 = 0.25, excess-KL ratio 1.995, at the cap of 2).
- **A2, pre-LN vs baseline:** belief metric not supported (2·SD 0.0995 against 0.1); PR(C_h) supported at q2 = 0.8 only (−0.150, −6%). Within pre-LN, `ln2` vs `resid_mid`: belief not supported, PR inconclusive.
- **A3, baseline `resid_post` vs `resid_mid`:** belief metric supported at q2 = 0.8 (+0.129, 2·SD 0.100); PR(C_h) supported in both conditions (−1.74, −0.72).
- Five verdicts are near their limits; seeds 3–5 (about 1.5 h, under $1) would re-test them.
- **Cost:** one A40, 1 h 34 min, about $0.92 of the $10 budget, well inside the 4 h cap.

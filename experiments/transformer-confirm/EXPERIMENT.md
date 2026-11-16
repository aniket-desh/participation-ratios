---
name: transformer-confirm
title: Batch 4: locked prediction of the belief-component spectrum on held-out reliabilities
status: done
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 3
budget_usd: 8
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-15T22:48:14Z
result: not-supported
finished_utc: 2026-11-16T00:06:20Z
writeup: experiments/transformer-confirm/README.md
---

# Batch 4: locked prediction of the belief-component spectrum on held-out reliabilities

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

Does a prediction fixed before training, using only the generator, give the geometry and spectrum of the belief component of a trained transformer's final residual at generator conditions never used to form it? The locked rule, found in Batch 2 (`experiments/transformer-discovery/grade.json`: 6 of 6 anisotropic runs closer to isotropic in m than in u, 8 of 9 in the isotropy band), is that the token-partialled belief encoding is **isometric in the probability coordinates m**. The belief component's PR then follows the generator-only curve (v1+v2)²/(v1²+v2²), which is nonconstant in q2: 1.443, 1.620, 1.882, 2.000, 1.948, 1.858 and 1.791 at q2 = 0.2, 0.25, 0.35, 0.5, 0.65, 0.8 and 0.9 (T = 16).

In the plan's §2.3 terms this is a **calibrated prediction**: the rule comes from discovery, is locked here, and is tested on held-out reliabilities and new seeds, with nothing fitted per condition.

**What the PR test does and does not add.** The observed belief-component PR is computed as PR(E_b Σ_b E_bᵀ), with Σ_b the generator's own belief covariance on the eval contexts, and the prediction uses the same Σ_b. So the PR comparison measures nothing beyond the shape and orientation of the learned metric G_b. Its dependence on q2 is supplied by the generator, not learned. The PR test is informative only where competing metric hypotheses predict different PRs. At q2 = 0.35 and 0.65 almost any near-isotropic G passes, including a whitened encoding (constant PR 2) (reviewer check). This definition therefore adds held-out reliabilities 0.2 and 0.9, where the m-isometric curve (1.443, 1.791) is far from the whitened value 2 and from the u-isometric curve (1.070, 1.209). It also grades the alternatives by the same rule and counts a PR pass as discriminating only where they fail. The full covariance PR(C_h), which also contains the final token's embedding, is not predicted and is reported descriptively.

## Hypothesis

Locked in `predictions.json` before any Batch 4 run. All at site `L0.resid_post`, step 20,000, n_eval 65,536.
- **P1, geometry.** The token-partialled belief metric G_b (b = m) is near-isotropic: both normalized eigenvalues lie in [0.4, 0.6].
- **P1-learned.** Training moves G_b toward isotropy: the trained top eigenvalue is closer to 0.5 than the same run's step-0 value. This is needed because random maps are already often in the band (reviewer simulation: 72% of random 2-column maps in d = 64 have top eigenvalue ≤ 0.6; in Batch 2, 5 of 9 runs were in the band at step 0).
- **P2, PR curve.** The partialled belief-component PR matches the m-isometric prediction within 10%. P2 is graded identically for three alternatives: a whitened encoding (PR = 2), the u-isometric curve, and each run's own step-0 belief PR. **P2-discriminating** holds when the m-isometric prediction is supported and all three alternatives are not supported. The headline scope is the held-out reliabilities {0.2, 0.35, 0.65, 0.9}.
- **P3, m over u.** In the six anisotropic conditions the partialled metric is closer to isotropic in m than in u.

For context only: Batch 2's three-seed means were 1.553, 1.992 and 1.741 at q2 = 0.25, 0.5 and 0.8 against predictions of 1.619, 2.000 and 1.855, so observed values sat 0–7% below. Its one outlier seed (q2 = 0.8, seed 2: top eigenvalue 0.615, e_PR 0.12) is why the rules below pool runs across reliabilities instead of requiring every run.

## Method

Code: `prt/` at the launch commit, unchanged from Batches 2 and 3. Training and analysis as in Batch 2's Method (`experiments/transformer-discovery/EXPERIMENT.md`), except as stated here.

- Generator: two independent binary factors, (ρ1, q1) = (0.8, 0.5), ρ2 = 0.8, q2 ∈ {0.2, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9}, token permutation seed 12345, T = 16, independent contexts from the stationary prior. Held out (never trained on or analyzed before this batch): 0.2, 0.35, 0.65 and 0.9. Discovery reliabilities with new seeds: 0.25, 0.5 and 0.8.
- Model: the Batch 2 baseline (L = 1, d = 64, 2 heads × 32, GELU MLP width 256, no normalization), PyTorch default init with seed 1000·seed + 2.
- Training: Adam with lr = 1e-3, Batch 2's selected lr (`experiments/transformer-discovery/lr.json`), fixed and not re-piloted. Batch 256 online contexts, 20,000 steps, checkpoints on the plan's schedule.
- Seeds 10–14 at every reliability: 35 runs, disjoint from Batch 2's seeds 0–2 and pilot seed 100. The plan asks for at least five seeds for confirmation (§10).
- Analysis (`prt.analysis`) at steps 0, 100, 1,000, 5,000 and 20,000, n_eval 65,536, eval seed 7, all sites and coordinate sets, including the token-partialled belief block. The generator-only reference PR is computed on the same eval contexts.
- Grading by `grade.py` against `predictions.json`. A run without a step-20,000 analysis (for example a diverged run) counts as not finished and not passing. A **passing run** has a tail-mean final-position KL (evaluations at steps ≥ 15,000) below its own k = 4 short-window floor, and a belief-content gain Δ = [R²(m_token) − R²(last_token)] at step 20,000 minus at step 0 that exceeds 0.02. These are the Batch 2 rules as amended there.

## Inputs

None: data are generated online. `data` is empty.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
bash $EXP_DIR/run.sh
```

`run.sh` trains the 35 runs (4 concurrent; a crashed run does not stop the driver) and analyzes them, held-out reliabilities first. It then runs `summarize.py` → `results.json`, `grade.py` → `grade.json`, and `plot.py` → `figures/pr_curve.png` (with the m-isometric, u-isometric and whitened curves drawn). Completed runs are skipped on a rerun.

## Expected outputs

- `runs/transformer-confirm/q{q2}_s{seed}/`: config, metrics, checkpoints and analysis, about 3–5 MB each.
- `$EXP_DIR/results.json`, `$EXP_DIR/grade.json`, `$EXP_DIR/figures/pr_curve.png`, `$EXP_DIR/logs/`, and `$EXP_DIR/README.md` (the write-up).

## Controls

- Step-0 analysis of every run: the belief-content gate Δ > 0.02 requires training to add belief information beyond token identity, and P1-learned requires training to move the geometry toward isotropy (the Δ gate alone would not exclude the near-isometry of random maps).
- The token-identity control: the belief block is partialled against the final token's one-hot.
- Short-window floors (k = 1–4) as the prediction-quality bar.
- Alternatives graded by the same P2 rule: whitened (PR 2), u-isometric (T = 16: 1.070, 1.162, 1.517, 2.000, 1.677, 1.340 and 1.209 at the seven reliabilities), and each run's own step-0 PR.
- P1 is also computed on the step-0 analyses, as a reference that shows what an untrained network already gives.
- New seeds at the discovery reliabilities, which separate replication from out-of-sample generalization.

## Success and failure criteria

A reliability counts toward a verdict only if at least 2 of its 5 runs pass. If a reliability in a verdict's scope has fewer than 2 passing runs, that verdict is **inconclusive**. Pooled statistics are over all passing runs in scope. Seed means are per reliability.
- **P1 supported** if at least 80% of pooled passing runs are in the [0.4, 0.6] band and every reliability's seed-mean top eigenvalue is ≤ 0.6. **Not supported** if fewer than 50% are in band or any seed-mean top eigenvalue is above 0.6. Otherwise inconclusive. The same rule applied to step-0 analyses is reported as the untrained reference.
- **P1-learned supported** if at least 80% of pooled passing runs have a trained top eigenvalue closer to 0.5 than at step 0. **Not supported** below 50%. Otherwise inconclusive.
- **P2 rule** (applied to the prediction and to each alternative, over a scope of reliabilities): **supported** if every seed-mean e_PR < 0.1 and at least 80% of pooled passing runs have e_PR < 0.1. **Not supported** if any seed-mean e_PR ≥ 0.1. Otherwise inconclusive. **P2-discriminating (headline)** requires, on the held-out scope {0.2, 0.35, 0.65, 0.9}, that the m-isometric prediction is supported and the whitened, u-isometric and step-0 alternatives are all not supported. The all-reliability scope is reported too.
- **P3 supported** if at least 80% of pooled passing runs in the six anisotropic reliabilities are closer to isotropy in m than in u, and no reliability is below 50%. **Not supported** if the pooled share is below 50% or any reliability is below 50%. Otherwise inconclusive.
- **Reported, not graded:** P3's pooled share at step 0 (Batch 2 found the m-over-u ordering already weakly present before training), and the share of runs in which training widens the u-minus-m distance from isotropy; whether the observed ordering of the seven seed means matches the predicted ordering; seed-mean e_PR with its seed SD per reliability; full PR(C_h) against q2; belief-PR trajectories at steps 0, 100, 1,000, 5,000 and 20,000; the single-final-evaluation bar alongside the tail-mean bar.
- **Expected risk, stated in advance:** with Batch 2's outlier rate (1 in 9), the pooled rules pass about 95% of the time if the effect holds (reviewer estimate). A per-reliability "every run" rule would have passed only about 60% of the time.

The run **failed** if fewer than 30 of the 35 runs finish, or if any held-out reliability has no finished run (`grade.json` → `run_failed`).

## Time and cost

Batch 2's worker took 219–240 s per 20k-step run with 4 concurrent and about 13.5 s per checkpoint of analysis (reviewers' reading of its logs). So: 35 runs ≈ 35 min of training, 175 checkpoint analyses ≈ 40 min, setup ≈ 5 min, about 1.5 h in total. The CPU smoke test of the whole driver (40 steps, n_eval 2,048) took 4.7 min. Cap 3 h. Worst case at the cap: $1.8 on an A40 ($0.59/hr), $6.3 on the A100 PCIe fallback, under the $8 budget. The night's committed worst case, Batches 2–4, is $26 of $150; actual spend so far is under $2.

## If something fails

- A training process errors: read its log, fix only a plumbing bug, rerun `run.sh` (completed runs are skipped). Code changes are pushed with `orch push` and recorded under Amendments.
- A run diverges: do not edit `grade.py` or the criteria. It counts as not finished, and the failure rule above applies.
- Time running out: analysis of the held-out reliabilities comes first (as ordered in `run.sh`). If needed, the analysis steps can be cut to `ASTEPS="0 20000"` (record as an amendment).
- Out of memory: reduce `PAR` to 2.
- Do not change `predictions.json` or the criteria after any Batch 4 result has been seen.

## Questions asked before launch

None asked: the user was away and gave a standing go-ahead for the plan's experiments within $25 per worker and $150 total, which covers this definition (approved under it). Decisions made in their absence:
- Locking the m-isometry rule from Batch 2 as a calibrated prediction.
- Using Batch 2's selected lr without re-piloting.
- New seeds 10–14 at all seven reliabilities.
- Pooled majority rules instead of "every run", because of Batch 2's one outlier seed.
- Adding q2 = 0.2 and 0.9 as held-out reliabilities beyond the two reserved in Batch 0, so the PR test can discriminate (reviewer P0).

## Review

First review (experiment-reviewer):
- **P0**, at the reserved held-out reliabilities 0.35 and 0.65 the PR test cannot separate the m-isometric rule from alternatives, because it restates G_b isotropy (a whitened encoding passes). **Fixed:** added held-out q2 = 0.2 and 0.9, where alternatives differ by 28–35%; added alternative predictions graded by the same rule, with a "discriminating" verdict; stated in the Question what the PR test does not add.
- P1, the step-0 control did not test geometry. **Fixed:** added P1-learned (trained top eigenvalue closer to 0.5 than at step 0), and P1 is also reported at step 0.
- P1, `grade.py` crashed on unfinished runs. **Fixed:** unfinished runs are counted, `run_failed` is computed in the script, and both were tested on a partial result file.
- P1, about a 40% risk of "inconclusive" under per-reliability rules. **Fixed:** pooled rules, with the risk stated.
- P2, the T label on the u curve was wrong. **Fixed:** T = 16 values, recomputed.

Re-check: all four earlier findings confirmed fixed and no new P0 (including the P3 step-0 addition). The reviewer notes that over the held-out scope the whitened alternative's failure rests on q2 = 0.2 alone. Three stale lines were fixed (the Controls wording, the seed count, the q2 = 0.2 value).

## Amendments

None.

## Outcome

**Not supported** (2026-11-16). The locked m-isometric prediction holds at q2 = 0.2–0.65 but fails at held-out q2 = 0.9. Write-up: [README.md](README.md).
- Run validity: 35 of 35 runs finished and passed (KL tail mean below the k = 4 floor, Δ 0.16–0.41). `run_failed` is false.
- P1, geometry: not supported. 29/35 in band (83%), but the q2 = 0.9 seed-mean top eigenvalue is 0.668 > 0.6. The step-0 reference is also not supported (6/35 in band).
- P1-learned: supported, 34/35 (97%) moved toward isotropy.
- P2, m-isometric, held-out scope: not supported. Seed-mean e_PR is 0.008, 0.013 and 0.005 at 0.2, 0.35 and 0.65, and 0.222 at 0.9. All-reliability scope: not supported (q2 = 0.9 is the only reliability ≥ 0.1; 0.8 gives 0.066).
- Alternatives (whitened, u-isometric, own step 0): all not supported on both scopes.
- P2-discriminating (headline): not met.
- P3, m over u: supported, 29/30 (97%), q2 = 0.9 at 4/5.
- Cost: one A40, 1.16 h, about $0.68 of the $8 budget and 3 h cap. No amendments.

---
name: transformer-longrun
title: Batch 2b: does the m-isometric belief metric persist with longer training?
status: done
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 3
budget_usd: 8
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-15T22:52:09Z
result: supported
finished_utc: 2026-11-15T23:45:47Z
writeup: experiments/transformer-longrun/README.md
---

# Batch 2b: does the m-isometric belief metric persist with longer training?

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

Batch 2 (`experiments/transformer-discovery/README.md`) found that at 20,000 steps the token-partialled belief encoding of the small transformer is much nearer isometric in the probability coordinates m than in the readout coordinates u. But PR(C_h) fell between 10,000 and 20,000 steps in every main run, and the q2 = 0.8 metric was still moving, so the representation had not settled. Batch 2's supervisor recommended exactly this follow-on.

This experiment asks whether, with 2.5× more training (50,000 steps), the belief metric stays isotropic in m, drifts toward u (the coordinates the softmax readout needs), or drifts elsewhere. It also asks where the full covariance PR settles. It also adds three seeds at q2 = 0.8, the condition with Batch 2's one outlier. The answer decides whether Batch 4's locked 20,000-step prediction describes a stable property or a transient.

## Hypothesis

Two outcomes, assessed rather than assumed:
- **Stable m-isometry.** At 50,000 steps the partialled metric stays in the isotropy band and still favors m over u, with little change from 20,000 steps.
- **Drift.** The metric moves materially between 20,000 and 50,000 steps, for example toward u, which would mean the readout coordinates take over as training continues.

## Method

Code: `prt/` at the launch commit. Batch 2's baseline configuration exactly (see its Method), except for training length: L = 1, d = 64, 2 heads × 32, GELU MLP width 256, no normalization, Adam with lr = 1e-3 (Batch 2's selected lr), batch 256 online contexts, T = 16, generator (ρ1, q1) = (0.8, 0.5), ρ2 = 0.8, token permutation seed 12345.

- Runs: q2 ∈ {0.25, 0.5, 0.8} × seeds {0, 1, 2}, plus q2 = 0.8 × seeds {3, 4, 5}: 12 runs of **50,000 steps**. Seeds 0–2 use Batch 2's data and init seeds, so their first 20,000 steps repeat Batch 2's runs, up to GPU nondeterminism. That makes Batch 2's 20k analysis a direct reference.
- Checkpoints: the plan's log schedule plus 30,000, 40,000 and 50,000. The schedule entries above 20,000 were added to `prt/train.py`, and shorter runs are unaffected.
- Analysis (`prt.analysis`) at steps 0, 5,000, 10,000, 20,000, 30,000, 40,000 and 50,000, n_eval 65,536, eval seed 7, all sites and coordinate sets, including the token-partialled belief block.
- Summary (`stability.py`), per run at site `L0.resid_post`: trajectories of the partialled top eigenvalue in m and in u, belief PR divided by the m-isometric prediction, PR(C_h) and the belief-content gain; the tail-mean final-position KL at 15k–20k and over the last quarter; the change of each quantity from 20,000 to 50,000 steps.

## Inputs

None: data are generated online. `data` is empty.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
bash $EXP_DIR/run.sh
```

`run.sh` trains the 12 runs (4 concurrent; a crashed run does not stop the driver) and analyzes them, q2 = 0.8 first. It then runs `summarize.py` → `results.json` and `stability.py` → `stability.json` and `figures/stability.png`. Completed runs are skipped on a rerun.

## Expected outputs

- `runs/transformer-longrun/q{q2}_s{seed}/`: about 4–6 MB per run.
- `$EXP_DIR/results.json`, `$EXP_DIR/stability.json`, `$EXP_DIR/figures/stability.png`, `$EXP_DIR/logs/`, and `$EXP_DIR/README.md` (the write-up).

## Controls

- Batch 2's 20,000-step values (the same seeds 0–2), as a reproducibility check of the first 20,000 steps.
- Step-0 analyses.
- The token-identity partialling and the belief-content gain.
- Short-window KL floors. A run whose last-quarter tail-mean final-position KL is above its k = 4 floor is flagged, and drift in it is reported separately.

## Success and failure criteria

The run **succeeded** if at least 10 of the 12 runs reach 50,000 steps with analysis at 20,000 and 50,000.

Grading at site `L0.resid_post`, over finished runs whose last-quarter tail-mean KL is below their k = 4 floor:
- **Stable m-isometry** if, at 50,000 steps, at least 80% of runs have a partialled top eigenvalue (m) in [0.4, 0.6]; at least 80% of anisotropic runs are closer to isotropy in m than in u; and the median |change of the top eigenvalue (m) from 20k to 50k| is ≤ 0.05.
- **Drift** if the median |change of the top eigenvalue (m) from 20k to 50k| exceeds 0.05, or fewer than 50% of runs are in band at 50k. Report the direction: toward u if, in the anisotropic runs, the u-minus-m distance from isotropy shrinks in at least 80% of them.
- **Inconclusive** otherwise.
- **Reported, not graded:** whether PR(C_h) settles, measured as the median relative change from 40k to 50k; the reproducibility of seeds 0–2 against Batch 2 at 20k (the difference of the partialled top eigenvalue and of PR(C_h)); the q2 = 0.8 seeds 3–5 against Batch 2's outlier; and the belief PR divided by the m-isometric prediction over time.

The run **failed** if fewer than 10 runs reach 50,000 steps.

## Time and cost

Batch 2's worker took about 230 s per 20,000-step run with 4 concurrent, so 50,000 steps take about 9.5 min per run: 12 runs in about 30 min. Analysis: 84 checkpoints × about 13.5 s ≈ 19 min. Setup about 5 min. Total about 1 h. Cap 3 h. Worst case at the cap: $1.8 on an A40 and $6.3 on the A100 PCIe fallback, under the $8 budget. Committed worst case for the night, including Batches 2–4, is $34 of $150, and actual spend is still under $3.

## If something fails

- A training process errors: read its log and fix only a plumbing bug, then rerun `run.sh`. Completed runs are skipped.
- Time running out: the analysis at steps 20,000 and 50,000 comes first. If needed, rerun the analysis with `ASTEPS="0 20000 50000"` and record it as an amendment.
- Out of memory: reduce `PAR` to 2.

## Questions asked before launch

None asked: the user was away and gave a standing go-ahead for the plan's experiments within $25 per worker and $150 total, which covers this definition (approved under it). This follow-on was recommended in Batch 2's `SUPERVISOR.md`. Decisions made in the user's absence: 50,000 steps, which is 2.5× Batch 2 and stays inside one cheap worker-hour; reusing seeds 0–2 so Batch 2's 20k values serve as a reference; and three extra seeds at q2 = 0.8.

## Review

skipped: this reruns the approved Batch 2 design (same model, data, lr, analysis and partialled metric) with only a longer training horizon and three extra seeds. Its grading thresholds reuse Batch 2's reviewed band and majority rules. The driver was smoke-tested end to end at 40 steps, and `stability.py` was exercised on Batch 2's real results.

## Amendments

None.

## Outcome

**Supported: stable m-isometry** (pre-registered rules, site `L0.resid_post`). Write-up: [README.md](README.md); grading: [grade.json](grade.json).

- Run succeeded: met, 12/12 runs reached 50,000 steps with analysis at 20k and 50k.
- KL filter: 12/12 below the k = 4 floor (last-quarter tail mean 22–29% of the floor at q2 = 0.25/0.5, 52–63% at q2 = 0.8); none flagged.
- Stable m-isometry: met. 11/12 in band at 50k (92%; q2 = 0.8 seed 0 at 0.612), m closer than u in 9/9 anisotropic runs, median |Δ top eig (m)| 20k→50k = 0.019.
- Drift: not met. The u-minus-m gap shrank in 4/9 anisotropic runs (all at q2 = 0.8), below the 80% needed for "toward u".
- PR(C_h) settling (reported): not settled; median relative change 40k→50k −2.7%, 11/12 fell.
- Reproducibility vs Batch 2 at 20k (reported): seeds 0–2 agree to within 1.1e-4 (top eig) and 0.01% (PR(C_h)).
- q2 = 0.8 seeds 3–5 (reported): 0.504, 0.545, 0.550 at 20k, all in band; Batch 2's outlier (seed 2) returned to the band by 40k, while seed 0 left it at 50k. Batch 2's every-run H1 rule fails at 50k on two runs.
- Belief PR / m-isometric prediction (reported): 0.992–1.001 at q2 = 0.25/0.5 and 0.880–0.978 at q2 = 0.8 at 50k.
- Cost and runtime: one A40, 47 min, about $0.46, against $8 and 3 h.

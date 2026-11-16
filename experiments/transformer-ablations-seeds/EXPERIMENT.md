---
name: transformer-ablations-seeds
title: Batch 3b: seeds 3-5 for the architecture ablations, graded on six seeds
status: done
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe
count: 1
hours: 3
budget_usd: 8
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-15T23:26:11Z
result: supported
finished_utc: 2026-11-16T00:54:30Z
writeup: experiments/transformer-ablations-seeds/README.md
---

# Batch 3b: seeds 3-5 for the architecture ablations, graded on six seeds

> The definition of one experiment on a GPU worker. Whoever drives the worker follows it as written and changes it only under "Amendments".

## Question

Batch 3 (`experiments/transformer-ablations/README.md`) found that the one-layer transformer's final-position geometry changes at the MLP stage. Five of its verdicts sat near their pre-registered limits:
- A3 on the belief metric (+0.129 against a 0.1 threshold, 2·SD = 0.100).
- A2 on PR(C_h) at q2 = 0.8 (−6% against 5%).
- A1's prediction-quality match (excess-KL ratio 1.995 against a cap of 2).
- The two belief-metric nulls (2·SD = 0.0995 against 0.1).

Its supervisor recommended re-testing them with seeds 3–5. Do Batch 3's verdicts hold when graded on six seeds instead of three, and do seeds 3–5 alone reproduce them?

## Hypothesis

Batch 3's verdicts are robust: each one, re-graded on seeds 0–5 pooled, keeps its direction. A verdict that flips between supported, not supported and inconclusive is reported as fragile.

## Method

This is an exact rerun of Batch 3's approved design (`experiments/transformer-ablations/EXPERIMENT.md`, Method) with three new seeds:
- Architectures base, nomlp and preln, each at the learning rate Batch 3's declared rule selected (1e-3 for all three; `lr_*.json` copied here). The pilots are not rerun.
- q2 ∈ {0.25, 0.8}, seeds {3, 4, 5}: 18 runs of 20,000 steps.
- The same model, generator, training, analysis (every checkpoint, n_eval 65,536) and partialled metric.

Grading uses Batch 3's own `grade.py`, unchanged except that the seed list is a parameter. With seeds 0–2 it reproduces Batch 3's `grades.json` exactly; this was checked before launch. It is applied twice:
- to Batch 3's committed `results.json` combined with these runs (seeds 0–5 pooled), which is the headline;
- to these runs alone (seeds 3–5), as an independent replication.

## Inputs

None: data are generated online. Batch 3's committed `experiments/transformer-ablations/results.json` is read for the pooled grading.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
bash $EXP_DIR/run.sh
```

`run.sh` trains the 18 runs (4 concurrent; a crashed run does not stop the driver), analyzes every checkpoint, then runs `summarize.py` → `results.json`. It combines that with Batch 3's results into `results_combined.json`, and grades seeds 0–5 → `grades_seeds0-5.{json,txt}` and seeds 3–5 → `grades_seeds3-5.{json,txt}`. Completed runs are skipped on a rerun.

## Expected outputs

- `runs/transformer-ablations-seeds/<arch>_q<q2>_s<seed>/`, about 3–5 MB each.
- `$EXP_DIR/results.json`, `results_combined.json`, `grades_seeds0-5.*`, `grades_seeds3-5.*`, `logs/`, and `README.md` (the write-up).

## Controls

As in Batch 3: step-0 analyses, the token-identity partialling and its R² gate, the short-window KL floors, the matched-quality label, and per-architecture learning rates from Batch 3's pilots.

## Success and failure criteria

The run **succeeded** if at least 16 of the 18 runs finish with analysis at step 20,000. The verdict rules are exactly Batch 3's (its "Success and failure criteria" and `grade.py` docstring), applied to seeds 0–5 pooled. The headline result is a table listing, for each of Batch 3's verdicts (A1, A2, A2-within-preln and A3, each with a belief-metric and a PR(C_h) verdict), Batch 3's three-seed verdict, the six-seed verdict and the seeds-3–5 verdict. A verdict is **robust** if the six-seed verdict equals Batch 3's and **fragile** otherwise. Whether attention-only passes the prediction-quality bar at q2 = 0.8 with the new seeds is reported.

The run **failed** if fewer than 16 runs finish.

## Time and cost

Batch 3 took 1 h 34 min for 27 runs plus analysis on one A40. These 18 runs and their analysis take about 1 h. Cap 3 h. Worst case: $1.8 on an A40 and $6.3 on the A100 PCIe fallback, under the $8 budget. Actual spend for the night so far is about $3.

## If something fails

- A training process errors: read its log and fix only a plumbing bug, then rerun `run.sh`. Completed runs are skipped.
- Time running out: analysis at step 20,000 is what grading needs. If needed, rerun the analysis with `--steps 0 20000` and record it as an amendment.
- Do not change `grade.py`'s rules.

## Questions asked before launch

None asked: the user was away and gave a standing go-ahead for the plan's experiments within $25 per worker and $150 total, which covers this definition (approved under it). The follow-on was recommended in Batch 3's `SUPERVISOR.md`.

## Review

skipped: this is a rerun of the reviewed and approved Batch 3 design with new seeds. Its grading is Batch 3's own unchanged rules, and the parameterized grader was verified to reproduce Batch 3's `grades.json` exactly with seeds 0–2. The driver was smoke-tested end to end at 20 steps.

## Amendments

None.

## Outcome

**Supported**: every one of Batch 3's verdicts holds on six seeds. Write-up: [README.md](README.md).
- Run success: 18 of 18 runs finished with analysis at step 20,000 (criterion: at least 16).
- Six-seed verdicts: all 8 equal Batch 3's (A1, A2, A2-within-pre-LN and A3, each on belief and PR), so all 8 are robust and none is fragile.
- Seeds 3–5 alone: 7 of 8 agree with Batch 3. Within-pre-LN PR is supported on seeds 3–5 against inconclusive on seeds 0–2 and on seeds 0–5.
- Attention-only at q2 = 0.8: 0 of 3 new seeds pass the prediction-quality bar (0 of 6 overall), so A1 stays undefined there.
- Cost and time: one A40, 1 h 21 min, about $0.80, against a 3 h cap and an $8 budget.

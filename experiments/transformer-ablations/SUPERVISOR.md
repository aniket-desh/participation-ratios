# Supervisor record: transformer-ablations

This experiment ran unattended from launch to write-up in tmux session `exp-transformer-ablations` on 2026-11-15, following [EXPERIMENT.md](EXPERIMENT.md) as approved. The result is **supported**: the MLP stage is where the final-position geometry changes. The PR part of that result is robust. The belief-metric parts, and two of the PR verdicts, sit near their limits. The findings are in [README.md](README.md), and this file covers what was done and decided.

## What ran

1. **Launch.** `orch experiment launch transformer-ablations` created `q349cwi9uik3x3`, an NVIDIA A40 in EU-SE-1 at $0.59/h. The A100 fallback was out of stock outside US-NC-1, and the A40 was first in the list anyway. The worker was ready about a minute after creation.
2. **The one command.** `orch exec q349cwi9uik3x3 --wait --name run -- 'set -euo pipefail; bash $EXP_DIR/run.sh'` ran from 21:39 to about 23:11 UTC and exited 0. It ran 9 pilots (all stable), lr selection (1e-3 for all three architectures), 18 main runs (about 3.5 min each, 4 concurrent), analysis of 15 checkpoints per main run (about 3 min per run, run sequentially, which took most of the time), pilot analysis, `summarize.py` and `plot.py`.
3. **Finish.** `orch finish` published 756 data files and 60 code and write-up files, merged the experiment folder into `main` (commit `750f7c6`), and deleted the worker. `orch ls` afterwards showed no worker of this experiment. The two pods still running then, `transformer-longrun` and `transformer-confirm`, belong to other supervisors.
4. **Write-up.** I wrote the README, ran the fact-checker on it, applied its corrections, ran `orch experiment done --result supported`, filled in the Outcome, and committed the folder (`0d9ca18`).

Nothing failed, so the definition's "If something fails" was never invoked, and there are **no amendments**.

## Result against each criterion

| Criterion | Result |
|---|---|
| Run succeeded (18 main runs finite, with analysis) | met: 18/18, 288 analysis records |
| Prediction-quality bar | 3/3 seeds pass in every cell except attention-only at q2 = 0.8 (0/3), so that cell's comparisons are inconclusive |
| A1 attention-only vs baseline | belief metric inconclusive; PR(C_h) supported (excess-KL ratio 1.995, against a cap of 2) |
| A2 pre-LN vs baseline | belief metric not supported (2·SD 0.0995 against 0.1); PR(C_h) supported at q2 = 0.8 only |
| A2 within pre-LN, `ln2` vs `resid_mid` | belief metric not supported; PR(C_h) inconclusive |
| A3 baseline `resid_post` vs `resid_mid` | belief metric supported at q2 = 0.8 (narrowly); PR(C_h) supported in both conditions |
| Failure conditions (an arm with all pilots unstable, or no analysis) | not triggered |

## Cost

One A40 for 1 h 34 min, about **$0.92** against a $10 budget and a 4 h cap (from the experiment log). One fact-checker agent run and a CPU smoke run on the coordinator cost no pod time.

## Decisions I made

- **I wrote `grade.py` and `tables.py`** to apply the definition's verdict rule and to produce the tables it asks for, since the definition's `summarize.py` collects the numbers but doesn't grade them. `grade.py` was written and tested on a 20-step CPU smoke run before the GPU results came back. The smoke outputs are in the session scratch directory, not in the repository.
- **Interpretations of the verdict rule where the definition is silent.** All four are also listed in the README under "Judgement calls", and none changes a verdict under the obvious alternative.
  - The PR "5% relative" threshold is taken relative to the second-named (reference) cell.
  - Sample SD (ddof = 1) is used over the three seeds.
  - The excess-KL ratio is computed over passing runs.
  - The m-or-u coordinate with the larger |Δ| is graded with its own SD and step-0 value.
- **The u coordinate is gated on the m_token R² gain, as the definition literally says**, rather than on its own gain as `plot.py` does. I checked that the two never disagree on any gate decision.
- **Overall result recorded as "supported".** The hypothesis is that the geometry is architecture-dependent, and A3 is supported on both metrics, while A1 and A2 are supported on PR. The recorded note and the Outcome both say which verdicts are borderline.
- **I added two exploratory analyses to the README, labelled as found after looking.** One splits the trace across the MLP stage. The other reports the within-pilot lr effect on PR, which the definition requires only where the selected lrs differ (they didn't).
- **The fact-checker corrected 12 of about 95 claims, mostly framing, and I applied all of them.** One correction asked me to cut the smoke-test claim because the checker couldn't find the outputs. I kept the claim and said where the outputs are instead, because I did run that test.

## What you need to do

- **Decide whether to run seeds 3–5** (18 more runs, about 1.5 h on an A40, under $1). Five verdicts are near their limits, and more seeds would show whether they hold: A3 belief, A2 PR, the A1 quality match, and the two belief nulls.
- **Decide whether the attention-only model should get longer training** in a separate experiment. This one forbids extending, and at q2 = 0.8 it ends at about twice the 4-token KL floor.
- **Add this experiment to `HANDOFF.md`.** I didn't edit the experiment log, `HANDOFF.md` or any file outside this folder, as instructed. The experiment log entry already has the run and cost, and `orch experiment done` recorded the result.
- **Nothing is left running** from this experiment.

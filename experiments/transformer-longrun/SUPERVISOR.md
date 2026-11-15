# Supervisor report: transformer-longrun

The experiment ran end to end inside its limits, and the result is **supported**: under the pre-registered rules the token-partialled belief metric is stable and m-isometric at 50,000 steps. The margin is thin at q2 = 0.8. Two of its six seeds drifted up to 0.60–0.61 over the extra 30,000 steps, one of them ending just outside the band, and grading q2 = 0.8 alone passes the in-band rule with no run to spare. PR(C_h) has not settled. The full write-up is [README.md](README.md), and the per-criterion verdict is in the "Outcome" section of [EXPERIMENT.md](EXPERIMENT.md).

## What ran

- **Launch:** the worker launched at 22:52 UTC on 2026-11-15 on one NVIDIA A40 in CA-MTL-1 at $0.59/hr (pod `82bc0u8tp1xycw`, code at commit `bc504c4`). The A40 was first in the definition's GPU list.
- **Driver:** the definition's one command, `bash $EXP_DIR/run.sh`, ran as a single `orch exec --wait` job and exited 0 after 45 min. All 12 runs trained to 50,000 steps (about 7.5 min each, four at a time), and all 84 checkpoints were analyzed (7 per run), followed by `summarize.py` and `stability.py`. The waiting `orch exec` logged one SSH timeout partway through. The process stayed alive and the job, which runs detached on the pod, wasn't affected, so I did nothing about it.
- **Collection:** `orch finish` published and merged the folder (commit `95dd5cf`, made by orch), brought back 324 data files under `runs/transformer-longrun/`, and deleted the worker (HTTP 204). Afterwards `orch ls` showed no worker of this experiment. The two pods still listed then, `aniket-pr-transformer-confirm` and `aniket-pr-transformer-ablations-seeds`, belong to other supervisors.
- **Grading and write-up:** on the coordinator I ran `grade.py` (criteria → `grade.json`) and wrote `README.md`. The fact-checker checked it, and its corrections were applied (commit `617bc9a`).

## Result against each criterion

| Criterion | Result |
|---|---|
| Run succeeded (≥ 10 of 12 runs to 50k with analysis at 20k and 50k) | Met, 12/12 |
| KL filter (last-quarter tail mean below the k = 4 floor) | 12/12 pass; none flagged |
| Stable m-isometry | Met: 11/12 in band (92%), m closer than u in 9/9, median \|Δ\| 0.019 |
| Drift | Not met; the u-minus-m gap shrank in 4/9 (needs 80% for "toward u") |
| PR(C_h) settling (reported) | Not settled: median −2.7% from 40k to 50k, 11/12 fell |
| Seeds 0–2 against Batch 2 at 20k (reported) | Agree to within 1.1e-4 |
| q2 = 0.8 seeds 3–5 (reported) | 0.504–0.550 at 20k, all in band. Batch 2's outlier came back into the band, and seed 0 left it at 50k |
| Belief PR / m-isometric prediction (reported) | 0.99–1.00 at q2 = 0.25/0.5, and 0.88–0.98 at q2 = 0.8 |

## Cost

One A40 for 47 min, about $0.46, against the $8 budget and 3 h cap. The fact-checker agent was the only other paid call.

## Decisions I made

1. **I wrote the grading code before seeing results.** The definition gives the rules in words, so I wrote `grade.py` to implement them at 22:54 UTC, while the first training wave was running and before any 20k checkpoint existed. I smoke-tested it on Batch 2's results with synthetic 50k entries; those scratch files weren't kept. It grades with a strict "below the floor" and counts a run as in band when its top eigenvalue lies in [0.4, 0.6]. That is the same as requiring both eigenvalues in band, since they sum to 1.
2. **I graded with the definition's pooled rules and reported a q2 = 0.8-only grading separately,** marked in the write-up as not pre-registered and computed after looking. It gives the same verdict (5/6 in band, 6/6 m-closer, median |Δ| 0.029), with no run to spare on the in-band rule.
3. **I recorded the result as "supported".** The definition's outcome labels are stable, drift and inconclusive, while `orch experiment done` takes supported, not-supported, inconclusive or failed. The "Stable m-isometry" hypothesis met all three of its rules, so I mapped it to "supported".
4. **I stated in the write-up that Batch 2's stricter every-run H1 rule still fails at 50k** (q2 = 0.8 seeds 0 and 1, against one seed at 20k). It isn't one of this experiment's criteria, but the fact-checker pointed out that a reader linking this result to Batch 4's prediction needs it.
5. **I applied all 13 fact-checker corrections,** mostly scope wording. The one that changed the reading softened "What this means" from "not a transient" to "persists from 20,000 to 50,000 steps under the pre-registered criteria".
6. **No amendments.** Nothing failed, so the definition's "If something fails" never applied, and the full analysis schedule fit well inside the cap.

## What you need to do

- **Decide whether the q2 = 0.8 drift needs a follow-on before relying on "isometric in m" for every seed.** Seeds 0 and 1 rose in every 10,000-step interval from 20k to 50k, but jumps between neighboring checkpoints at q2 = 0.8 reach 0.057, so three intervals can't separate a trend from a wander. Analyzing checkpoints every 1,000 steps from 20k to 50k, or training to 100k, would settle it in about an hour on one A40, for under $1.
- **Any PR(C_h) law should name its training step.** The median PR(C_h) still falls 2–4% per 10,000 steps at 50k.
- **Commit the experiment log when convenient.** I left `EXPERIMENTS.md` to the planning session as instructed. orch has written this run's entry, but the file is untracked in git.
- **Nothing was pushed,** and no worker of this experiment is left running.

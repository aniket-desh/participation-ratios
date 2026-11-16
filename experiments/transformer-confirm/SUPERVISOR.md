# Supervisor record: transformer-confirm (Batch 4)

The locked prediction failed its headline test. The m-isometric belief-PR curve matches the trained transformers within 2% (seed means) from q2 = 0.2 to 0.65, including held-out q2 = 0.2, where every alternative fails. At held-out q2 = 0.9 it misses by e_PR = 0.22, with all five seeds above 0.1, so P2 and P2-discriminating are not met. The run itself was clean: 35 of 35 runs finished and passed, nothing was amended, and it cost about $0.68. The result is recorded as `not-supported`, and the write-up is [README.md](README.md).

## What ran

- The worker was `ic0lhov0n3hdoq` (`aniket-pr-transformer-confirm`), one NVIDIA A40 in EU-SE-1 at $0.59/h. It launched from approved commit 723c40b with an empty working-tree diff. It was created at 22:48:36 UTC and deleted at 23:58:07 UTC by `orch finish`.
- The one command from the definition, `bash $EXP_DIR/run.sh`, ran as job `20261115T224922-run` from 22:49:29 to 23:56:30 and exited with code 0. Training took 24 min (35 runs, 4 at a time) and analysis 43 min (175 checkpoints). Then `summarize.py`, `grade.py` and `plot.py` ran on the worker.
- The results were published and merged: 770 data files under `runs/transformer-confirm/`, and the experiment folder merged into `experiments/transformer-confirm/`.

## Result against each criterion

| criterion | verdict | key number |
|---|---|---|
| run validity (≥ 30 of 35 finished, every held-out q2 present) | met | 35/35 finished, 35/35 passing |
| P1, geometry | not supported | q2 = 0.9 seed-mean top eigenvalue 0.668 > 0.6 (pooled in band 29/35) |
| P1-learned | supported | 34/35 moved toward isotropy |
| P2, m-isometric, held-out scope | not supported | seed-mean e_PR 0.008 / 0.013 / 0.005 / **0.222** at 0.2 / 0.35 / 0.65 / 0.9 |
| P2, m-isometric, all reliabilities | not supported | q2 = 0.9 is the only reliability with seed-mean e_PR ≥ 0.1 (0.8: 0.066) |
| P2 alternatives (whitened, u-isometric, own step 0) | all not supported | the whitened alternative passes at 0.35 and 0.65 (anticipated) and fails at 0.2 and 0.9 |
| P2-discriminating (headline) | not met | it fails because the m-isometric prediction fails at 0.9 |
| P3, m over u | supported | 29/30, q2 = 0.9 at 4/5 |

## Cost

One A40 for 1.16 h comes to about $0.68 (orch estimate at $0.59/h), against an $8 budget and a 3 h cap. One fact-checker agent run was the only other paid work.

## Decisions I made

- **No amendments.** Nothing failed, so no clause of "If something fails" was used, and `predictions.json`, `grade.py` and the criteria stayed as committed.
- **I added `extras.py` after grading** to compute the "reported, not graded" items the definition lists but `grade.py` doesn't produce. Those are the single-final-evaluation bar, per-reliability e_PR with seed SD, trajectories at the five checkpoints and the step-0 P1 detail. It also produces one figure, `figures/trajectories.png`. It reads only `results.json` and `grade.json`, and it changes no verdict. Two of its quantities weren't named in the definition: the factor-1 share and the position of the observed PR between the u-isometric and m-isometric predictions. The write-up labels both exploratory.
- **I recorded the result as `not-supported`, not `inconclusive` or `failed`.** The rule is explicit (any seed-mean e_PR ≥ 0.1 in scope means not supported), and the run produced a full answer. I also reported, marked as post hoc, that leaving out q2 = 0.9 would pass P2. I didn't treat that as a verdict, because q2 = 0.9 is one of the two reliabilities added to make the test discriminating.
- **I applied the fact-checker's six corrections.** One error was a miscount of the 50,000-step follow-on's q2 = 0.8 seeds (four of six moved away from isotropy, not two). The other five were scope and completeness fixes. None changed a verdict. Before applying the Batch 2b numbers, I checked them against `experiments/transformer-longrun/grade.json` and its README.
- **I left one cross-experiment link in the write-up.** "Not yet checked" cites Batch 2b's 50,000-step result at q2 = 0.8, because it bears on whether the q2 = 0.9 miss is undertraining. I didn't touch that experiment's files.

## What the user must do

- **Update the experiment log and any shared notes.** I left `EXPERIMENTS.md` and the HANDOFF notes alone, as instructed. The planning session should add Batch 4's entry from the Outcome section of [EXPERIMENT.md](EXPERIMENT.md).
- **Decide on the follow-on.** The cheapest informative next step is to train the five q2 = 0.9 seeds to 50,000 steps and add q2 = 0.85 and 0.95. That would cost under $1 and about 30–60 min on one A40, and it would show whether the high-reliability departure from m-isometry is stable and grows with q2 or is a stage of training. Batch 2b's finding that q2 = 0.8 seeds drift *away* from isotropy past 20,000 steps suggests that the departure is real.
- **Nothing to clean up.** No worker of this experiment is left (`orch ls` at the end shows only another supervisor's pod), and nothing was pushed.

# Supervisor report: transformer-discovery

The experiment ran end to end inside its limits and came out **inconclusive**. H1u (isometric in u) is not supported, m is favored over u in all six anisotropic runs, and H1 (isometric in m) misses "supported" by one seed at q2 = 0.8. That verdict depends on amendment A1, which the planning session sent mid-run. Under the original prediction-quality bar, H1 would read supported. The full write-up is [README.md](README.md), and the per-criterion verdict is in the "Outcome" section of [EXPERIMENT.md](EXPERIMENT.md).

## What ran

- **Launch:** the worker was launched at 21:13 UTC on 2026-11-15: one NVIDIA A40 in EU-SE-1 at $0.59/hr (pod `33iq91taufktt0`, code at commit `f8511e4`). The A100 fallback was out of stock, and the A40 was first in the definition's list anyway.
- **Driver:** the definition's one command, `bash $EXP_DIR/run.sh`, ran as a single `orch exec --wait` job and exited 0 after 1 h 16 min.
- **Training:** three lr pilots ran, then the declared rule selected lr = 1e-3. The nine main runs and three controls followed, each 20,000 steps at about 3.5–4 min per run with four at a time.
- **Analysis:** all 15 checkpoints of all 15 runs were analyzed at n_eval 65,536, and the main runs' final checkpoints were re-analyzed at 16,384 and 262,144. The job ended with `results.json` and three figures.
- **Collection:** `orch finish` published and merged the folder (commit `5ab347a`, which orch made), brought back 498 data files under `runs/transformer-discovery/`, and deleted the worker. `orch ls` afterwards showed no worker of this experiment. The one remaining pod, `aniket-pr-transformer-ablations`, belongs to another supervisor.
- **Grading:** on the coordinator I wrote `grade.py` (criteria → `grade.json`) and `plot_grade.py` (`figures/partialled_metric.png`), then the write-up. The fact-checker checked it, and its 13 corrections were applied.

## Result against each criterion

| Criterion | Result |
|---|---|
| Run succeeded (9 main runs to 20k, finite, all analysis files) | Met |
| Prediction-quality bar, tail mean (A1) | 9/9 pass. Single-final rule: 8/9, and q2 = 0.8 seed 2 misses by 0.4% |
| Belief-content precondition, Δ > 0.02 | 9/9 (Δ 0.24–0.45; untrained 0.002–0.005) |
| H1, isometric in m | Inconclusive. 8/9 in band; q2 = 0.8 seed 2 has top eig 0.615 and e_PR 0.141. Supported under the original bar |
| H1u, isometric in u | Not supported. u top eig 0.68–0.80 in all 6 anisotropic runs |
| m vs u | m favored, 6/6. Already weakly true at step 0 |
| Seed consistency, CV(PR(C_h)) < 5% | Met: 4.7%, 1.7%, 2.7%; tail-KL ratio ≤ 1.21 |
| Measurement validity, within 1% at 262k | Met, within 0.25% |
| Shuffled control | Δ_shuf = −0.001, no sign of memory or artifact |
| Permutation control | Passes for both relabelled runs |

## Cost

One A40 for 1 h 19 min, about $0.78, against the $8 budget and 3 h cap. The fact-checker agent was the only paid call beyond the worker.

## Decisions I made

1. **Relaunch after the session restart.** The first attempt was killed (exit 137) before anything launched. On restart I confirmed with `orch ls` and `orch experiment ls` that no worker existed and the definition was still `approved`, then launched once.
2. **Every `orch` call ran with `ORCH_CONFIG` set to this project's `orch.toml`,** per the definition's review note. I confirmed it with `orch config path`.
3. **I didn't parallelize the analysis.** It ran serially at about 3.5 min per run, and the projected finish of about 1.4 h was well inside the cap, so I left `run.sh` unchanged.
4. **I accepted amendment A1 from the planning session** (`pr-work`). It moved the KL bar to the mean of steps 15k–20k and recorded both verdicts. Before recording it, I checked its factual claim against the worker's logs, and it held. I wrote the amendment's provenance into it: it arrived after training, and I had already seen that it would admit q2 = 0.8 seed 2. I graded with the amended rule because the definition says changes recorded under Amendments govern, and the tail mean is the less noisy measure. Afterwards it turned out to decide H1. The write-up gives both verdicts prominently, so you can apply either rule.
5. **Where the definition was silent, I filled in the grading rules myself.** The seed-consistency CV uses the sample standard deviation (the population form doesn't change the verdict). The m-vs-u count uses all six anisotropic runs, which are also all passing runs. A run is in band only if both eigenvalues lie in [0.4, 0.6], and the bar is a strict "below the floor". "Not supported" through the factor-1-share trend requires both q2 = 0.25 and q2 = 0.8 to have at least two passing seeds.
6. **I added one figure and two scripts in the folder,** because the existing figures show the raw-fit metric, not the graded partialled one. `plot.py`, `summarize.py` and `run.sh` were not changed.
7. **I applied all 13 fact-checker corrections.** The two that changed the reading softened the headline from "isometric in m" to "much nearer isometric in m than in u", and added that m-over-u and the H1u rejection also hold weakly at step 0. Both therefore rely on the belief-content precondition to count as learned.

## What you need to do

- **Decide which prediction-quality rule you accept for H1.** With amendment A1 (as graded) H1 is inconclusive. With the original rule it is supported on eight runs, and the q2 = 0.8 condition then rests on two seeds. Nothing else in the result depends on this choice.
- **Consider a short follow-on before treating "isometric in m" as a law.** It would cover more seeds at q2 = 0.8 and longer training. PR(C_h) fell between 10k and 20k in every main run, and the q2 = 0.8 metric is still moving. Training the nine main runs to 50k steps is about 30 min on one A40 plus analysis, roughly $0.5–1. The definition doesn't allow extending inside this experiment.
- **Update the experiment log.** I left `EXPERIMENTS.md` to the planning session as instructed. It is currently untracked in git. orch wrote this run's entry, but the result field there may still say "not recorded yet".
- **Nothing was pushed,** and no worker of this experiment is left running.

# Supervisor record: transformer-ablations-seeds (Batch 3b)

An unattended Claude session in tmux on the coordinator supervised this experiment from launch to write-up, between 23:26 on 2026-11-15 and about 01:00 UTC on 2026-11-16. It followed the approved definition in [EXPERIMENT.md](EXPERIMENT.md) without amendments.

## What ran

`orch experiment launch transformer-ablations-seeds` took the definition's first GPU choice that was in stock and could stay under budget. The A100 PCIe fallback was out of stock outside US-NC-1, and the launcher picked an NVIDIA A40 in CA-MTL-1 at $0.59/h (worker `ukxzx46svb5s9j`, job `20261115-232636-transformer-ablations-seeds-ukxzx46svb5s9j`, code at commit `4f590ba`). The definition's single command, `bash $EXP_DIR/run.sh`, ran as one `orch exec --wait` step from 23:27:52 and exited 0 at about 00:46. Training the 18 runs took 14 minutes, four at a time, and analyzing all 15 checkpoints of each run took about 65 minutes, one run at a time. The driver then wrote `results.json` (270 records), `results_combined.json` (Batch 3's 288 plus these 270) and both gradings. `orch finish` published the results, merged the folder into `main` as commit `852341f` and deleted the worker at 00:47:35. `orch ls` afterwards reported no running workers for this project.

## Result against each criterion

- **Run success (at least 16 of 18 runs analyzed at step 20,000):** met, 18 of 18.
- **Robustness (six-seed verdict equals Batch 3's):** all 8 verdicts are robust and none is fragile. On seeds 0–5 the verdicts are: A1 belief inconclusive, A1 PR supported, A2 belief not supported, A2 PR supported, A2-within-pre-LN belief not supported, A2-within-pre-LN PR inconclusive, A3 belief supported, A3 PR supported.
- **Replication on seeds 3–5 alone:** 7 of 8 agree. Within-pre-LN PR is supported on the new seeds (Δ −0.414 at q2 = 0.25, 2·SD 0.283), against inconclusive on seeds 0–2 and pooled.
- **Attention-only at q2 = 0.8:** fails the prediction-quality bar in all 3 new seeds (tail KL 0.00251–0.00263 against a floor of 0.00132).
- **Recorded result:** `supported` (the hypothesis was that the verdicts are robust). Details and the near-limit margins are in [README.md](README.md).

## Cost

The worker ran 1 h 21 min at $0.59/h, about $0.80, against the definition's $8 budget and 3 h cap. The fact-checker agent and this session's own API use are not included.

## Decisions made

- **No amendments.** Nothing failed, so the definition's "If something fails" never applied, and the analysis ran at every checkpoint as specified rather than falling back to `--steps 0 20000`.
- **Two SSH timeouts to the worker** during progress checks (around 00:31 and 00:46) were treated as transient, because the next attempt connected and the driver was still progressing. Neither affected the job.
- **Result label `supported`,** because the definition's hypothesis is that Batch 3's verdicts keep their direction on six seeds, and all eight did.
- **Added one figure and its script** ([`plot.py`](plot.py), [`figures/deltas.png`](figures/deltas.png)) that show each comparison's Δ in units of its threshold under the three seed sets. The definition doesn't ask for it; it is descriptive and changes no verdict.
- **Checks beyond the definition:** I reran `grade.py` to confirm it reproduces Batch 3's `grades.json` on seeds 0–2 and this run's six-seed file byte for byte. I also checked the partialling invariant (agreement to 2e-9), the step-0 gate, the u-gate convention (one disagreement, at an ungraded site) and that all losses are finite.
- **Fact-check.** The fact-checker agent corrected six of about 100 claims in the write-up, none of which changed a verdict. All six were applied, and the write-up lists them under "How it was checked".
- **Left untouched:** the shared `EXPERIMENTS.md` (untracked in the working tree) and every file outside this folder, as instructed.

## What the user must do

- Nothing is required. No worker is left running and nothing was pushed.
- The project's experiment log and `HANDOFF.md` don't record this result yet; the planning session owns them.
- One optional follow-up: whether LayerNorm before the MLP lowers PR at q2 = 0.25 (supported on seeds 3–5, inconclusive pooled) would take about three more seeds, roughly $0.80 and 80 minutes on an A40.

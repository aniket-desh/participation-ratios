# Debrief: first overnight session on participation-ratios (2026-11-15 to 16)

## Outcome

The three planning documents are organized under `docs/` with a README that maps where each topic is treated. A tested codebase (`prt/`) covers every calculation in the plan up to the small transformer. Four CPU studies and five GPU experiments ran to completion for $3.64 of GPU spend. The main scientific result is a locked prediction, made before training, that works at moderate reliability and fails at the high end. In a one-layer transformer trained on the two-factor binary generator, the belief encoding (with final-token identity partialled out) is close to isometric in the posterior-probability coordinates m, so the belief component's PR follows a curve computed from the generator alone. That holds within 2% at q2 = 0.2–0.65, including held-out 0.2, where the three alternatives all fail. It misses by 22% at held-out q2 = 0.9, where the encoding moves part of the way toward the readout coordinates u = atanh(ρqm).

## Left for you

1. **Decide which prediction-quality bar you accept for Batch 2's H1.** Under the tail-mean bar, which I amended in mid-run before grading, H1 (isometric in m) is inconclusive. Under the original single-final-evaluation bar it is supported on eight runs. Batch 2b (50k steps) and Batch 4 make this mostly moot, but the verdict on record depends on it. Details: `experiments/transformer-discovery/README.md`, "Judgement calls".
2. **Review the docs decision.** I did not merge or delete overlapping text between the roadmap, derivations and plan. I moved them with `git mv` and added a concordance to `README.md` instead. If you wanted a real merge, that is still to do.
3. **Fix or check the tmux environment problem.** The tmux server's global environment carries coupled-tensor-networks' `ORCH_*` values, so every `orch session start` and `orch experiment supervise` session opens configured for that project. I worked around it per session (below), but the toolkit's `session-tmux` should pin each session's project. Also check how the global environment got those values.
4. **Choose the next theory step.** The failure at high reliability and the q2 = 0.8 drift (Batch 2b) both point at roadmap §5.3: a Fisher/Hessian-weighted metric from cross-entropy, which would interpolate between m (weak signal) and u (strong signal). The other open directions are listed under "Not yet checked".
5. **Stop this planning session when you've read this:** `orch session stop pr-work`. It is idle and costs nothing, but it stays up until stopped.

## What was done and found

**Docs and onboarding.** The documents are now `docs/theory_roadmap.md`, `docs/derivations.md` and `docs/experiment_plan.md`, and only their two cross-links were edited. `orch.toml` reuses the existing coordinator and volumes with prefix `aniket-pr`. No infrastructure was created.

**Codebase** (`prt/`, 26 tests passing). It covers:
- exact HMM filtering and enumeration;
- the two-factor generator, including its coupled and shuffled variants;
- spectral observables;
- the balanced linear benchmark;
- restricted positional attention with its learning ODE;
- the Package 5 many-head model with its continuum solver, onset coefficients and §5.14 interaction extension;
- the Stage D transformer, online training and per-checkpoint analysis, including a token-partialled belief metric added on review.

Every module is checked against an exact reference: the plan's T = 6 table and the derivations' onset table (to 12 digits) are reproduced, and analytic gradients match autograd to 1e-12.

**CPU studies** (write-ups fact-checked):
- **Gate A** ([batch0](experiments/batch0-generator-validation/README.md)) passes: two independent filters agree on all 4,096 contexts, and the plan's table is reproduced to six digits.
- **Gate B** ([batch1](experiments/batch1-balanced-linear/README.md)) passes: the closed-form balanced trajectory, including its nonmonotone PR path, is reproduced with first-order Euler convergence.
- **Stage C** ([learned filter](experiments/stagec-learned-filter/README.md)): the exact gradient-flow ODE predicts the attention decay that SGD learns. The 2–15% excess at low reliability shrinks when the learning rate is halved, so it comes from SGD's discretization.
- **Package 5** ([continuum](experiments/heads-continuum/README.md)):
  - finite networks approach the computed continuum with error close to $H^{-1/2}$, and attention learning drives the PR change;
  - the initialization fluctuation formula matches 2,000 draws per H to within 2% for H ≥ 64;
  - on coupled generators the interaction logit learns real interaction content at ε = 0.2, but it raises PR even when the generator is independent.

**GPU experiments.** All definitions were reviewed, and each ran under its own tmux supervisor:

| Experiment | Result | Cost |
| --- | --- | --- |
| [Batch 2, discovery](experiments/transformer-discovery/README.md) | Inconclusive. The partialled belief metric is much nearer isotropic in m than in u (6/6 anisotropic runs); H1u is not supported, and H1 misses by one seed | $0.78 |
| [Batch 2b, 50k steps](experiments/transformer-longrun/README.md) | Stable m-isometry: 11/12 runs in band, median drift 0.019. PR(C_h) has not settled | $0.46 |
| [Batch 3, ablations](experiments/transformer-ablations/README.md) | Supported: the MLP stage is where the geometry changes (PR(C_h) 3.78 → 2.04 at q2 = 0.25). Attention-only misses the quality bar at q2 = 0.8 | $0.92 |
| [Batch 3b, seeds 3–5](experiments/transformer-ablations-seeds/README.md) | All 8 Batch 3 verdicts hold on six seeds | $0.80 |
| [Batch 4, locked prediction](experiments/transformer-confirm/README.md) | Not supported: holds within 2% at q2 = 0.2–0.65, misses by 22% at held-out 0.9. P1-learned (34/35) and m-over-u (29/30) are supported | $0.68 |

## Judgement calls

Open to change:
- **The docs were kept whole** (see "Left for you").
- **Batch 2's KL bar was amended mid-run.** The Batch 3 reviewer showed that the single-final-evaluation bar was decided by noise at q2 = 0.8. The amendment reached Batch 2's supervisor after training had finished but before grading, and it decided H1's verdict. Both verdicts are reported.
- **Batch 3's conditions were fixed before Batch 2 finished** (q2 ∈ {0.25, 0.8}, the reliabilities where m and u separate most), so the two ran in parallel. The plan says to choose them afterwards.
- **Batch 4's m-isometry rule came from Batch 2.** It is a calibrated prediction, not a parameter-free one. On review I also added held-out q2 = 0.2 and 0.9 beyond the two values reserved in Batch 0, because at 0.35 and 0.65 the PR test cannot tell the rule from a whitened encoding.
- **Batch 2b and Batch 3b skipped review.** Each is a rerun of a reviewed design with a longer horizon or new seeds, and Batch 3b's grader reproduces Batch 3's grades exactly.
- **Not done:**
  - Mess3: its exact matrices could not be fetched and verified overnight.
  - The width pilot (Batch 5): it needs the Bordelon–Chaudhry–Pehlevan parameterization implemented and audited, and Gate D/E are not reached.
  - The §5.12 trained-time fluctuation theory.

Routine: each GPU definition's reviewer findings were all fixed before launch (each has a "Review" section). Stage C and the continuum studies ran on the coordinator CPU without definitions, because they are not GPU work.

## How it was checked

- **Tests.** 26 tests against exact references or analytical limits (`uv run pytest -q`).
- **Experiment reviews.** Each new GPU definition was reviewed, and the reviews changed the designs substantially:
  - Batch 2: the token-partialled metric replaced a raw metric that untrained models already passed, and a u-versus-m R² test that could not fail was dropped.
  - Batch 3: the tail-mean KL bar, a matched-quality label, and separate verdicts for the belief metric and the full PR.
  - Batch 4: the held-out q2 = 0.2 and 0.9 conditions, alternatives graded by the same rule, a learned-isotropy test, and pooled rules.
- **Fact-checks.** The four CPU write-ups were fact-checked: 14 of about 128 claims were corrected and every headline held. Each GPU supervisor also fact-checked its own write-up.
- **Fluctuation check.** A first fluctuation comparison flagged a 1.55× variance excess (p = 0.011). 2,000 draws per H showed it was sampling noise, and the write-up says so.

## Not yet checked

- **Fisher-weighted metric theory for the high-reliability regime** (roadmap §5.3): the obvious next step after Batch 4. It is a theory task, not a run.
- **Dense q2 = 0.8 checkpoints or 100k-step training,** to tell drift from wander (Batch 2b's supervisor): about 1 h of A40 time, under $1.
- **Whether PR(C_h) settles.** It was still falling 2–4% per 10k steps at 50k, so any full-covariance PR law must name its training step.
- **Attention-only with longer training,** since it misses the quality bar at q2 = 0.8: under $1.
- **Finite-H continuum comparison past τ = 100:** about 1 h of CPU.
- **Mess3, the width pilot, and the unwhitened linear benchmark** (see "Judgement calls").

## Where things landed

- **participation-ratios:** branch `main`, 51 commits from `57f76ea` (the original docs) to the latest, all pushed to `github.com/aniket-desh/participation-ratios`. Nothing is uncommitted. The run data (`runs/`, 967 MB of checkpoints and analysis JSON) is git-ignored and lives on the network volume. The supervisors' merges came back through `orch finish` as ordinary commits on `main`.
- **coupled-tensor-networks:** no lasting change. My first `orch experiment new`, under its inherited environment, created `studies/ttn/transformer-discovery/EXPERIMENT.md` and `.orch/collected/claims/studies__ttn__transformer-discovery`. I deleted both within minutes; they were my own files. Its modified `orch.toml`, its modified `EXPERIMENTS.md` (its own experiments) and its untracked `.vscode/` were not touched.

## Side effects

- **Spend.** $3.64 of GPU across five A40 workers, against $150 allowed and $25 per worker; the largest single experiment cost $0.92. Claude sessions and subagents (reviewers, fact-checkers, five supervisors) are on top of that and were not metered here.
- **Pods.** None remain (`orch ls`). No volumes or coordinators were created or changed, and none of coupled-tensor-networks' pods was touched.
- **Files in this project's git-ignored `.orch/`:**
  - `.orch/pr_env.sh`, this project's `ORCH_*` exports;
  - a `source` line at the top of each `.orch/logs/sessions/exp-*/runner.sh` that pins the session to this project;
  - `.orch/pods/tt9hswhnr5wjg4/meta.json`, copied from coupled-tensor-networks so `orch doctor` recognizes the coordinator.

  All three can be deleted safely.
- **tmux.** Sessions `exp-transformer-*` are stopped and `pr-work` (this session) is still running. Scratch files are under `/tmp/claude-0/-workspace-participation-ratios/`.
- **Nothing persistent was written.** No memory files, settings or skills were created or changed.

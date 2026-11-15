# Handoff: overnight session of 2026-11-15

Written for Aniket, who was away. This file is updated as the work goes; the newest status is at the top of each section.

## Status at a glance

| Item | Status | Where |
| --- | --- | --- |
| Docs organized | done | `docs/`, `README.md` |
| Onboarding (`orch.toml`, reusing the coordinator and volumes) | done | `orch.toml` |
| Codebase `prt/` with exact-reference tests (25 passing) | done | `prt/`, `tests/` |
| Batch 0 / Gate A (generator, filter, covariance pipeline, imposed filter, weak signal, coupled generator) | done, passes | `experiments/batch0-generator-validation/README.md` |
| Batch 1 / Gate B (balanced linear learning) | done, passes | `experiments/batch1-balanced-linear/README.md` |
| Stage C learned positional filter (exact ODE vs SGD) | done; the ODE predicts SGD's β and PR | `experiments/stagec-learned-filter/README.md` |
| Package 5 many-head continuum study | running on coordinator CPU | `experiments/heads-continuum/` |
| Batch 2 small-transformer discovery (GPU) | reviewed twice, approved, running under supervisor `exp-transformer-discovery` on an A40 ($0.59/hr, 3 h cap, $8 budget) | `experiments/transformer-discovery/` |

## Decisions made in your absence (please review)

- **Docs were not deduplicated by deletion.** The roadmap, the derivations and the plan overlap, but each argues a different level of detail, and deleting text from research documents overnight seemed worse than keeping them whole. Instead the README has a section concordance across the three documents, and it names the derivations as the notation reference. The files were moved with `git mv` (history kept) and renamed to `docs/theory_roadmap.md`, `docs/derivations.md` and `docs/experiment_plan.md`. The only edits are the two cross-links.
- **Token-identity coordinate sets were added to the analysis** (`last_token`, `m_token`, `u_token`) as a declared control. The final-position residual contains the last token's embedding, which by itself spans rank 3.
- **The held-out reliabilities are locked at $q_2\in\{0.35,0.65\}$** in Batch 0, before any model was analyzed.
- **Batch 2 started while Batch 1 was still running.** Gate A and Gate B checks were already passing in `tests/`, and Batch 2 is a discovery batch, not a claim about trained-metric prediction.
- **I amended Batch 2's prediction-quality bar mid-run (before grading).** It now uses the tail mean of final-position KL over steps 15k–20k instead of the single last evaluation. The Batch 3 reviewer found that, in Batch 2's live logs, the single-evaluation KL at q2 = 0.8 fluctuates around the k = 4 floor (0.0006–0.0017 against 0.00132). The supervisor was asked to record this under Amendments and to report both verdicts.
- **Batch 3 (architecture ablations) runs in parallel with Batch 2,** with its conditions fixed in advance at q2 ∈ {0.25, 0.8} rather than chosen after Batch 2. It has its own baseline arm and its own per-architecture lr pilots, so it is self-contained.
- The rest are listed in each experiment's definition under "Questions asked before launch".

## Things you should know

- **The tmux session inherited coupled-tensor-networks' `ORCH_*` environment** (including `ORCH_CONFIG=/workspace/coupled-tensor-networks/orch.toml`). The first `orch experiment new` therefore created `studies/ttn/transformer-discovery/EXPERIMENT.md` and a name-claim file inside coupled-tensor-networks. I deleted both. They were my own creations from that command, and nothing else in that repo was touched (its pre-existing modified `orch.toml` and untracked `.vscode/` were left as they were). Every `orch` call since then sets `ORCH_CONFIG` to this project's file explicitly. It's worth checking how the tmux server's global environment got that value.

## Spend

| Worker | GPU | Hours | Cost | Status |
| --- | --- | --- | --- | --- |
| — | — | — | — | — |

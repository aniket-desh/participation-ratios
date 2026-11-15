# Handoff: overnight session of 2026-11-15

Written for Aniket, who was away. This file is updated as the work goes; the newest status is at the top of each section.

## Status at a glance

| Item | Status | Where |
| --- | --- | --- |
| Docs organized | done | `docs/`, `README.md` |
| Onboarding (`orch.toml`, reusing the coordinator and volumes) | done | `orch.toml` |
| Codebase `prt/` with exact-reference tests (26 passing) | done | `prt/`, `tests/` |
| Fact-check of the four CPU write-ups | done; 14 of about 128 claims corrected, all headlines held | the write-ups' "How it was checked" |
| Batch 0 / Gate A (generator, filter, covariance pipeline, imposed filter, weak signal, coupled generator) | done, passes | `experiments/batch0-generator-validation/README.md` |
| Batch 1 / Gate B (balanced linear learning) | done, passes | `experiments/batch1-balanced-linear/README.md` |
| Stage C learned positional filter (exact ODE vs SGD) | done; the ODE predicts SGD's β and PR | `experiments/stagec-learned-filter/README.md` |
| Package 5 many-head continuum study | done. The continuum predicts finite networks with H^{-1/2} error, attention learning drives the PR change, the initialization fluctuation formula matches 2,000 draws per H to within 2% (H ≥ 64), and on coupled generators the interaction logit learns real interaction content at ε = 0.2 but raises PR even when the generator is independent | `experiments/heads-continuum/README.md` |
| Package 5.14 interaction extension (dependent generators) | code and autograd test done; no runs yet | `prt/heads.py` |
| Batch 3 architecture ablations (GPU) | **done, supported**: the MLP stage is where the final-position geometry changes (PR(C_h) 3.78 → 2.04 at q2 = 0.25). Attention-only misses the KL bar at q2 = 0.8. Five verdicts sit near their limits. Cost $0.92 | `experiments/transformer-ablations/README.md`, `SUPERVISOR.md` |
| Batch 3b, seeds 3–5 for the ablations, graded with Batch 3's rules on six seeds (GPU) | approved (review skipped: a rerun with new seeds; the grader reproduces Batch 3 exactly), supervisor `exp-transformer-ablations-seeds` | `experiments/transformer-ablations-seeds/` |
| Batch 2 small-transformer discovery (GPU) | **done, inconclusive**: the token-partialled belief metric is much nearer isometric in m than in u (6/6 anisotropic runs), so H1u is not supported. H1 misses "supported" by one q2 = 0.8 seed (top eig 0.615), and depends on which KL bar you accept (see below). Cost $0.78 | `experiments/transformer-discovery/README.md`, `SUPERVISOR.md` |
| Batch 4 confirmatory sweep with locked m-isometry prediction, held-out q2 ∈ {0.2, 0.35, 0.65, 0.9}, 5 seeds × 7 reliabilities (GPU) | reviewed twice, approved, supervisor `exp-transformer-confirm` | `experiments/transformer-confirm/` |
| Batch 2b longer training (50k steps) to test whether the m-isometry persists (GPU) | **done, stable m-isometry**: 11/12 runs in band at 50k, m closer than u in 9/9 anisotropic runs, median drift of the top eigenvalue 0.019 from 20k to 50k. q2 = 0.8 seeds wander near the band edge. PR(C_h) still falling (−2.7% over the last 10k). Seeds 0–2 reproduce Batch 2 at 20k to 1e-4. Cost $0.46 | `experiments/transformer-longrun/README.md`, `SUPERVISOR.md` |

## Decisions made in your absence (please review)

- **Docs were not deduplicated by deletion.** The roadmap, the derivations and the plan overlap, but each argues a different level of detail, and deleting text from research documents overnight seemed worse than keeping them whole. Instead the README has a section concordance across the three documents, and it names the derivations as the notation reference. The files were moved with `git mv` (history kept) and renamed to `docs/theory_roadmap.md`, `docs/derivations.md` and `docs/experiment_plan.md`. The only edits are the two cross-links.
- **Token-identity coordinate sets were added to the analysis** (`last_token`, `m_token`, `u_token`) as a declared control. The final-position residual contains the last token's embedding, which by itself spans rank 3.
- **The held-out reliabilities are locked at $q_2\in\{0.35,0.65\}$** in Batch 0, before any model was analyzed.
- **Batch 2 started while Batch 1 was still running.** Gate A and Gate B checks were already passing in `tests/`, and Batch 2 is a discovery batch, not a claim about trained-metric prediction.
- **I amended Batch 2's prediction-quality bar mid-run (before grading).** It now uses the tail mean of final-position KL over steps 15k–20k instead of the single last evaluation. The Batch 3 reviewer found that, in Batch 2's live logs, the single-evaluation KL at q2 = 0.8 fluctuates around the k = 4 floor (0.0006–0.0017 against 0.00132). The supervisor was asked to record this under Amendments and to report both verdicts.
- **Batch 3 (architecture ablations) runs in parallel with Batch 2,** with its conditions fixed in advance at q2 ∈ {0.25, 0.8} rather than chosen after Batch 2. It has its own baseline arm and its own per-architecture lr pilots, so it is self-contained.
- The rest are listed in each experiment's definition under "Questions asked before launch".

## Decisions waiting for you

- **Batch 2, H1's verdict depends on the KL bar.** Under the amended tail-mean bar, H1 (isometric in m) is inconclusive. Under the original single-final-evaluation bar, it is supported on eight runs. Both verdicts are in the Batch 2 write-up. The amendment arrived after training had finished but before grading. Its stated reason (single-evaluation noise around the floor) holds against the logs.
- **Batch 4's PR test mostly restates the geometry test.** The belief-component PR is computed with the generator's own belief covariance, so it adds evidence only at reliabilities where alternatives (whitened, u-isometric) predict different values. The reviewer caught this, and Batch 4 now includes q2 = 0.2 and 0.9 and grades the alternatives with the same rule. Read its "discriminating" verdict, not the raw P2 pass.

## Things you should know

- **The tmux session inherited coupled-tensor-networks' `ORCH_*` environment** (including `ORCH_CONFIG=/workspace/coupled-tensor-networks/orch.toml`). The first `orch experiment new` therefore created `studies/ttn/transformer-discovery/EXPERIMENT.md` and a name-claim file inside coupled-tensor-networks. I deleted both. They were my own creations from that command, and nothing else in that repo was touched (its pre-existing modified `orch.toml` and untracked `.vscode/` were left as they were). Every `orch` call since then sets `ORCH_CONFIG` to this project's file explicitly. It's worth checking how the tmux server's global environment got that value.

## Spend

| Worker | GPU | Hours | Cost | Status |
| --- | --- | --- | --- | --- |
| — | — | — | — | — |

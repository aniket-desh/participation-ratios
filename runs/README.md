# Raw per-run outputs

These are the raw outputs of the GPU transformer experiments, copied from the network volume. Each top-level folder matches an experiment under [`../experiments/`](../experiments/), whose `results.json`, grades and figures were computed from these files. Model checkpoints (`ckpt/step_*.pt`, about 315 MB) are not in git, so they stay on the network volume.

Each run folder is named `<variant>_q<q2>_s<seed>`. The variant prefix is missing for the base architecture. Folders whose names contain `pilot_lr` or start with `_sel` hold the learning-rate selection runs. Each run folder contains:

- `config.json` holds the generator parameters ($\rho_a$, $q_a$, token permutation), the architecture, the optimizer settings, the seeds and the final excess-loss diagnostics.
- `metrics.jsonl` has one line per evaluation step, with cross-entropy, the Bayes-optimal cross-entropy and excess KL, both averaged over positions and at the final position.
- `analysis/step_NNNNNN.json` has, for each checkpoint, one record per residual site (`L0.resid_pre`, `L0.attn_out`, `L0.resid_mid`, `L0.mlp_out`, `L0.resid_post`), computed at the final position over `n_eval` held-out contexts. `C_h` gives the activation spectrum (trace, PR, nPR, $d_{95}$, normalized eigenvalues `p`). Each belief-coordinate set (`m`, `u`, `m_aug`, `walsh`, and the token-controlled `last_token`, `m_token`, `u_token`) gives the forward-regression split $C_h = C_{\mathrm{belief}} + C_{\mathrm{res}} + C_{\mathrm{cross}}$, the learned metric eigenvalues and reverse-probe $R^2$. `m_token` and `u_token` also give the token-partialled belief metric. `_meta.generator` stores the exact generator reference values ($v_1$, $v_2$, factored, Walsh and $u$-isometric PR).

The fields are defined in the docstring of [`prt/analysis.py`](../prt/analysis.py), which produced them.

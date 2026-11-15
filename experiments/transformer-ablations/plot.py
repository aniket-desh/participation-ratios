"""Batch 3 figures from results.json: per-site observables at the final checkpoint, by architecture and condition."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

recs = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
main = [r for r in recs if "_q" in r["run"] and r["analysis"] == "analysis"]
last = max(r["step"] for r in main)
fin = [r for r in main if r["step"] == last]
ARCH = ["base", "nomlp", "preln"]; Q2 = sorted({r["q2"] for r in fin}); col = dict(zip(ARCH, ["C0", "C1", "C2"]))
def _gated(s, cs):
    """Partialled metric only where beliefs add >= 0.01 held-out R^2 beyond token identity; otherwise it is noise."""
    gain = s[cs]["r2_heldout"] - s["last_token"]["r2_heldout"]
    return s[cs]["partialled"]["factor1_share"] if gain >= 0.01 else np.nan


panels = [("PR(C_h)", lambda s: s["C_h"]["pr"]),
          ("R2(m_token) - R2(last_token)", lambda s: s["m_token"]["r2_heldout"] - s["last_token"]["r2_heldout"]),
          ("partialled G_b(m) factor-1 share*", lambda s: _gated(s, "m_token")),
          ("partialled G_b(u) factor-1 share*", lambda s: _gated(s, "u_token"))]
for q in Q2:
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for a in ARCH:
        rs = [r for r in fin if r["arch"] == a and r["q2"] == q]
        if not rs:
            continue
        sites = [f"L0.{k}" for k in ["resid_pre", "ln1", "attn_out", "resid_mid", "ln2", "mlp_out", "resid_post"]]
        for i, (title, fn) in enumerate(panels):
            for r in rs:
                xs = list(range(len(sites)))
                ax[i].plot(xs, [fn(r["sites"][s]) if s in r["sites"] else np.nan for s in sites], "o-", c=col[a], alpha=.6,
                           label=a if r is rs[0] else None)
            ax[i].set_xticks(range(len(sites))); ax[i].set_xticklabels([s.split(".")[1] for s in sites], rotation=45)
            ax[i].set_title(title)
    ax[0].legend(); fig.suptitle(f"q2 = {q}, step {last}, final position, 3 seeds per architecture "
                 "(*shown only where beliefs add >= 0.01 R^2 beyond the final token)")
    fig.tight_layout(); fig.savefig(out / f"sites_q{q}.png", dpi=130)
print("figures ->", out)

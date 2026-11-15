"""Declared LR selection (experiment plan §8.1): the largest pilot LR that is reliably stable with good validation learning.

Stable: every validation loss finite, and over the last quarter of evaluations the final-position KL never exceeds
2x its minimum over that window plus 1e-3 nats (no late divergence or spikes).
Good learning: final mean KL(Bayes||model) within 20% (+1e-4 nats) of the best pilot.
"""
import json
import math
import sys
from pathlib import Path

rows = {}
for d in sorted(Path(sys.argv[1]).glob("pilot_lr*")):
    if not (d / "metrics.jsonl").exists():
        continue
    m = [json.loads(l) for l in open(d / "metrics.jsonl")]
    lr = json.loads((d / "config.json").read_text())["lr"]
    finite = all(math.isfinite(r["ce"]) for r in m) and m[-1]["step"] == json.loads((d / "config.json").read_text())["steps"]
    tail = [r["kl_final"] for r in m[-max(1, len(m) // 4):]]
    stable = finite and max(tail) <= 2 * min(tail) + 1e-3
    rows[lr] = dict(final_kl=m[-1]["kl"], final_kl_last_pos=m[-1]["kl_final"], stable=stable)
stable = [r["final_kl"] for r in rows.values() if r["stable"]]
if not stable:  # every pilot unstable or missing: this architecture's arm is reported as failed
    print(json.dumps(dict(lr=None, pilots={str(k): v for k, v in rows.items()}), indent=1)); sys.exit(0)
best = min(stable)
ok = [lr for lr, r in rows.items() if r["stable"] and r["final_kl"] <= 1.2 * best + 1e-4]
print(json.dumps(dict(lr=max(ok), pilots={str(k): v for k, v in rows.items()}), indent=1))

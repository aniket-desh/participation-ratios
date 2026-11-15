"""Collect per-run, per-checkpoint, per-site observables into one small JSON (no activations, spectra truncated to 16)."""
import json
import sys
from pathlib import Path

R, out = Path(sys.argv[1]), Path(sys.argv[2])
recs = []
for run in sorted(p for p in R.iterdir() if (p / "config.json").exists()):
    cfg = json.loads((run / "config.json").read_text())
    metrics = {r["step"]: r for r in map(json.loads, open(run / "metrics.jsonl"))}
    for adir in sorted(run.glob("analysis*")):
        for f in sorted(adir.glob("step_*.json")):
            a = json.loads(f.read_text())
            meta = a.pop("_meta")
            sites = {}
            for site, r in a.items():
                s = dict(C_h={k: r["C_h"][k] for k in ["trace", "pr", "npr", "d95"]} | {"p16": r["C_h"]["p"][:16]})
                for cs, d in r.items():
                    if cs == "C_h":
                        continue
                    s[cs] = {k: d[k] for k in ["frac_explained", "r2_heldout", "cross_rel_fro", "G_norm_eigs",
                                                "Gw_norm_eigs", "reverse_r2"]}
                    s[cs]["pr_C_belief"] = d["C_belief"]["pr"]
                    s[cs]["pr_C_res"] = d["C_res"]["pr"]
                    if "partialled" in d:
                        s[cs]["partialled"] = d["partialled"]
                sites[site] = s
            m = metrics.get(meta["step"], {})
            recs.append(dict(run=run.name, analysis=adir.name, step=meta["step"], n_eval=meta["n_eval"],
                             q2=cfg["q2"], seed=cfg["seed"], lr=cfg["lr"], shuffled=cfg.get("shuffled", False),
                             perm_seed=cfg["perm_seed"], kl=m.get("kl"), kl_final=m.get("kl_final"),
                             kl_uniform_final=cfg.get("kl_uniform_final"), kl_lastk_final=cfg.get("kl_lastk_final"), generator=meta["generator"], sites=sites))
out.write_text(json.dumps(recs))
print(f"{len(recs)} records -> {out}")

"""Online next-token training of the Stage D transformer on a two-factor generator (experiment plan §8.1, §10).

    python -m prt.train --out runs/x --q2 0.25 --seed 0 [--lr 1e-3 --steps 20000 ...]

Each step draws fresh independent contexts of T+1 tokens from the generator's stationary prior; the model sees
T tokens and is trained with cross-entropy on all T next-token targets. Validation reports model cross-entropy,
the finite-context Bayes cross-entropy and their difference (mean conditional KL, nats/token), both averaged over
positions and at the final position. Checkpoints follow the plan's log schedule.

Seeds are separated: generator token permutation (perm_seed), training stream (seed*1000+1),
initialization (seed*1000+2), validation contexts (fixed val_seed shared across runs).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .generators import shuffled_control, two_factor_hmm
from .transformer import ModelConfig, Transformer

SCHEDULE = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def make_hmm(cfg: dict):
    perm = np.random.default_rng(cfg["perm_seed"]).permutation(4) if cfg["perm_seed"] >= 0 else np.arange(4)
    hmm = two_factor_hmm(cfg["rho1"], cfg["q1"], cfg["rho2"], cfg["q2"], eps=cfg["eps"], c=cfg["c_corr"], perm=perm)
    return shuffled_control(hmm) if cfg.get("shuffled") else hmm


def evaluate(model, toks_np, bayes, device, batch=4096):
    """Model CE, Bayes CE and KL(Bayes || model) on fixed contexts, per position then averaged."""
    model.eval()
    ce_pos, kl_pos = [], []
    with torch.no_grad():
        for i in range(0, len(toks_np), batch):
            x = torch.as_tensor(toks_np[i:i + batch], device=device)
            logq = F.log_softmax(model(x[:, :-1]).double(), -1)
            eta = torch.as_tensor(bayes[i:i + batch], device=device)
            ce_pos.append(-(logq.gather(-1, x[:, 1:, None])[..., 0]).cpu())
            kl_pos.append((eta * (torch.log(eta.clamp_min(1e-300)) - logq)).sum(-1).cpu())
    ce, kl = torch.cat(ce_pos).mean(0), torch.cat(kl_pos).mean(0)
    model.train()
    return dict(ce=ce.mean().item(), kl=kl.mean().item(), kl_final=kl[-1].item(), ce_final=ce[-1].item())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho1", type=float, default=0.8); ap.add_argument("--q1", type=float, default=0.5)
    ap.add_argument("--rho2", type=float, default=0.8); ap.add_argument("--q2", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.0); ap.add_argument("--c_corr", type=float, default=0.8)
    ap.add_argument("--perm_seed", type=int, default=12345, help="-1 = identity token labels")
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--d_head", type=int, default=32); ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--norm", default="none", choices=["none", "pre"]); ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val_seed", type=int, default=99)
    ap.add_argument("--n_val", type=int, default=8192); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = vars(a)
    out = Path(a.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)

    hmm = make_hmm(cfg)
    cfg["perm"] = hmm.meta.get("perm")
    mc = ModelConfig(vocab=4, ctx=a.ctx, d=a.d, n_heads=a.n_heads, d_head=a.d_head, n_layers=a.n_layers,
                     norm=a.norm, mlp=not a.no_mlp)
    torch.manual_seed(a.seed * 1000 + 2)
    model = Transformer(mc).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    cfg["model"] = model.config_dict()
    cfg["n_params"] = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    rng_train = np.random.default_rng(a.seed * 1000 + 1)
    val_toks = hmm.sample(a.n_val, a.ctx + 1, np.random.default_rng(a.val_seed))
    val_bayes = hmm.bayes_predictive(val_toks[:, :-1])
    bayes_ce = float(-np.mean(np.log(np.take_along_axis(val_bayes, val_toks[:, 1:, None], -1))))
    schedule = sorted({s for s in SCHEDULE if s <= a.steps} | {a.steps})

    log = open(out / "metrics.jsonl", "w")
    t0 = time.time()
    for step in range(a.steps + 1):
        if step in schedule:
            torch.save(model.state_dict(), out / "ckpt" / f"step_{step:06d}.pt")
        if step % a.eval_every == 0 or step in schedule:
            ev = evaluate(model, val_toks, val_bayes, a.device)
            rec = dict(step=step, tokens=step * a.batch * a.ctx, bayes_ce_sampled=bayes_ce, time=time.time() - t0, **ev)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if not math.isfinite(ev["ce"]):
                raise SystemExit(f"non-finite validation loss at step {step}")
        if step == a.steps:
            break
        x = torch.as_tensor(hmm.sample(a.batch, a.ctx + 1, rng_train), device=a.device)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 4), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    log.close()
    print(json.dumps(dict(out=str(out), final=ev, seconds=time.time() - t0)))


if __name__ == "__main__":
    main()

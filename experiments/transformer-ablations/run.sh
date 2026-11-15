#!/usr/bin/env bash
# Batch 3 driver (experiment plan §8.1 ablations, §11 Batch 3). Run from the project root: bash $EXP_DIR/run.sh
# Three architectures (base = Batch 2 baseline, nomlp = attention only, preln = pre-LayerNorm with MLP), each with
# its own declared lr pilot on q2 = 0.5, then q2 in {0.25, 0.8} x seeds {0,1,2}; analysis; summary; figures.
set -euo pipefail
EXP_DIR=${EXP_DIR:-experiments/transformer-ablations}
R=${R:-runs/transformer-ablations}
STEPS=${STEPS:-20000}
PAR=${PAR:-4}
NEVAL=${NEVAL:-65536}
EXTRA=${EXTRA:-}
mkdir -p "$R" "$EXP_DIR/logs"
train() { name=$1; shift; [ -f "$R/$name/ckpt/step_$(printf %06d "$STEPS").pt" ] && return 0
          uv run python -m prt.train --out "$R/$name" --steps "$STEPS" $EXTRA "$@" > "$EXP_DIR/logs/train_$name.log" 2>&1; }
export -f train; export R STEPS EXP_DIR EXTRA
declare -A FLAGS=([base]="" [nomlp]="--no_mlp" [preln]="--norm pre")

for arch in base nomlp preln; do
  for lr in 1e-4 3e-4 1e-3; do echo "${arch}_pilot_lr${lr} --q2 0.5 --seed 100 --lr $lr ${FLAGS[$arch]}"; done
done | sed 's/[[:space:]]*$//' | xargs -P "$PAR" -L1 bash -c 'train "$@"' _ || true   # a diverged pilot is judged by select_lr.py

for arch in base nomlp preln; do
  mkdir -p "$R/_sel_$arch"; for d in "$R/${arch}_pilot_lr"*; do ln -sfn "$(realpath "$d")" "$R/_sel_$arch/pilot_lr${d##*_lr}"; done
  uv run python "$EXP_DIR/select_lr.py" "$R/_sel_$arch" > "$EXP_DIR/lr_$arch.json"
done
{
  for arch in base nomlp preln; do
    LR=$(python3 -c "import json;print(json.load(open('$EXP_DIR/lr_$arch.json'))['lr'])")
    [ "$LR" = "None" ] && { echo "arch $arch: no stable pilot, arm skipped" >&2; continue; }
    for q2 in 0.25 0.8; do for s in 0 1 2; do echo "${arch}_q${q2}_s${s} --q2 $q2 --seed $s --lr $LR ${FLAGS[$arch]}"; done; done
  done
} | sed 's/[[:space:]]*$//' | xargs -P "$PAR" -L1 bash -c 'train "$@"' _ || true   # analyze whatever finished

for d in "$R"/base_q*/ "$R"/nomlp_q*/ "$R"/preln_q*/; do
  [ -d "$d" ] || continue; n=$(basename "$d"); [ -f "$d/analysis/step_$(printf %06d "$STEPS").json" ] && continue
  uv run python -m prt.analysis "$d" --n_eval "$NEVAL" > "$EXP_DIR/logs/analysis_$n.log" 2>&1
done
# pilots at their final checkpoint only, to measure the lr effect within each architecture (reviewer P2)
for d in "$R"/*_pilot_lr*/; do
  [ -d "$d" ] || continue; n=$(basename "$d"); [ -f "$d/analysis/step_$(printf %06d "$STEPS").json" ] && continue
  [ -f "$d/ckpt/step_$(printf %06d "$STEPS").pt" ] || continue
  uv run python -m prt.analysis "$d" --n_eval "$NEVAL" --steps 0 "$STEPS" > "$EXP_DIR/logs/analysis_$n.log" 2>&1
done
uv run python "$EXP_DIR/summarize.py" "$R" "$EXP_DIR/results.json"
uv run python "$EXP_DIR/plot.py" "$EXP_DIR/results.json" "$EXP_DIR/figures"
echo done

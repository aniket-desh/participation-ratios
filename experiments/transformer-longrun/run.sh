#!/usr/bin/env bash
# Batch 2b driver. Run from the project root: bash $EXP_DIR/run.sh
# Batch 2's baseline configuration and lr (1e-3), trained to 50,000 steps: q2 in {0.25, 0.5, 0.8} x seeds {0,1,2},
# plus q2 = 0.8 seeds {3,4,5}. Analysis at steps 0, 5000, 10000, 20000, 30000, 40000, 50000; stability summary.
set -euo pipefail
EXP_DIR=${EXP_DIR:-experiments/transformer-longrun}
R=${R:-runs/transformer-longrun}
STEPS=${STEPS:-50000}
PAR=${PAR:-4}
NEVAL=${NEVAL:-65536}
ASTEPS=${ASTEPS:-0 5000 10000 20000 30000 40000 50000}
EXTRA=${EXTRA:-}
mkdir -p "$R" "$EXP_DIR/logs"
train() { name=$1; shift; [ -f "$R/$name/ckpt/step_$(printf %06d "$STEPS").pt" ] && return 0
          uv run python -m prt.train --out "$R/$name" --steps "$STEPS" $EXTRA "$@" > "$EXP_DIR/logs/train_$name.log" 2>&1; }
export -f train; export R STEPS EXP_DIR EXTRA
{ for q2 in 0.25 0.5 0.8; do for s in 0 1 2; do echo "q${q2}_s${s} --q2 $q2 --seed $s --lr 1e-3"; done; done
  for s in 3 4 5; do echo "q0.8_s${s} --q2 0.8 --seed $s --lr 1e-3"; done; } | xargs -P "$PAR" -L1 bash -c 'train "$@"' _ || true
for d in "$R"/q0.8_*/ "$R"/q0.25_*/ "$R"/q0.5_*/; do
  [ -d "$d" ] || continue; n=$(basename "$d"); [ -f "$d/analysis/step_$(printf %06d "$STEPS").json" ] && continue
  uv run python -m prt.analysis "$d" --n_eval "$NEVAL" --steps $ASTEPS > "$EXP_DIR/logs/analysis_$n.log" 2>&1
done
uv run python "$EXP_DIR/summarize.py" "$R" "$EXP_DIR/results.json"
uv run python "$EXP_DIR/stability.py" "$EXP_DIR/results.json" "$R" "$EXP_DIR/stability.json" "$EXP_DIR/figures" "$STEPS"
echo done

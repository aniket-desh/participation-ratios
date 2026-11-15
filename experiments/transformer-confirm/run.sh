#!/usr/bin/env bash
# Batch 4 driver (experiment plan §11 Batch 4). Run from the project root: bash $EXP_DIR/run.sh
# Baseline architecture at Batch 2's selected lr (1e-3), q2 in {0.2, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9} x seeds 10-14
# (disjoint from Batch 2's seeds 0-2 and pilot seed 100); analysis at steps 0, 100, 1000, 5000, 20000; grading.
set -euo pipefail
EXP_DIR=${EXP_DIR:-experiments/transformer-confirm}
R=${R:-runs/transformer-confirm}
STEPS=${STEPS:-20000}
PAR=${PAR:-4}
NEVAL=${NEVAL:-65536}
ASTEPS=${ASTEPS:-0 100 1000 5000 20000}
EXTRA=${EXTRA:-}
mkdir -p "$R" "$EXP_DIR/logs"
train() { name=$1; shift; [ -f "$R/$name/ckpt/step_$(printf %06d "$STEPS").pt" ] && return 0
          uv run python -m prt.train --out "$R/$name" --steps "$STEPS" $EXTRA "$@" > "$EXP_DIR/logs/train_$name.log" 2>&1; }
export -f train; export R STEPS EXP_DIR EXTRA
for q2 in 0.2 0.35 0.65 0.9 0.25 0.5 0.8; do for s in 10 11 12 13 14; do echo "q${q2}_s${s} --q2 $q2 --seed $s --lr 1e-3"; done; done \
  | xargs -P "$PAR" -L1 bash -c 'train "$@"' _ || true
for d in "$R"/q0.2_*/ "$R"/q0.35_*/ "$R"/q0.65_*/ "$R"/q0.9_*/ "$R"/q0.25_*/ "$R"/q0.5_*/ "$R"/q0.8_*/; do   # held-out first
  [ -d "$d" ] || continue; n=$(basename "$d"); [ -f "$d/analysis/step_$(printf %06d "$STEPS").json" ] && continue
  uv run python -m prt.analysis "$d" --n_eval "$NEVAL" --steps $ASTEPS > "$EXP_DIR/logs/analysis_$n.log" 2>&1
done
uv run python "$EXP_DIR/summarize.py" "$R" "$EXP_DIR/results.json"
uv run python "$EXP_DIR/grade.py" "$EXP_DIR/results.json" "$R" "$EXP_DIR/predictions.json" "$EXP_DIR/grade.json" "$STEPS"
uv run python "$EXP_DIR/plot.py" "$EXP_DIR/results.json" "$EXP_DIR/grade.json" "$EXP_DIR/figures"
echo done

#!/usr/bin/env bash
# Batch 3b driver. Run from the project root: bash $EXP_DIR/run.sh
# Batch 3's three architectures at their selected lr (1e-3 for all), q2 in {0.25, 0.8}, seeds {3,4,5}; analysis of all
# checkpoints; then grading with Batch 3's rules on seeds 0-5 pooled (Batch 3's committed results.json + these runs)
# and on seeds 3-5 alone.
set -euo pipefail
EXP_DIR=${EXP_DIR:-experiments/transformer-ablations-seeds}
R=${R:-runs/transformer-ablations-seeds}
STEPS=${STEPS:-20000}
PAR=${PAR:-4}
NEVAL=${NEVAL:-65536}
B3=${B3:-experiments/transformer-ablations/results.json}
EXTRA=${EXTRA:-}
mkdir -p "$R" "$EXP_DIR/logs"
train() { name=$1; shift; [ -f "$R/$name/ckpt/step_$(printf %06d "$STEPS").pt" ] && return 0
          uv run python -m prt.train --out "$R/$name" --steps "$STEPS" $EXTRA "$@" > "$EXP_DIR/logs/train_$name.log" 2>&1; }
export -f train; export R STEPS EXP_DIR EXTRA
declare -A FLAGS=([base]="" [nomlp]="--no_mlp" [preln]="--norm pre")
for arch in base nomlp preln; do for q2 in 0.25 0.8; do for s in 3 4 5; do
  echo "${arch}_q${q2}_s${s} --q2 $q2 --seed $s --lr 1e-3 ${FLAGS[$arch]}"; done; done; done \
  | sed 's/[[:space:]]*$//' | xargs -P "$PAR" -L1 bash -c 'train "$@"' _ || true
for d in "$R"/*_q*/; do
  [ -d "$d" ] || continue; n=$(basename "$d"); [ -f "$d/analysis/step_$(printf %06d "$STEPS").json" ] && continue
  uv run python -m prt.analysis "$d" --n_eval "$NEVAL" > "$EXP_DIR/logs/analysis_$n.log" 2>&1
done
uv run python "$EXP_DIR/summarize.py" "$R" "$EXP_DIR/results.json"
python3 - "$B3" "$EXP_DIR/results.json" "$EXP_DIR/results_combined.json" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
json.dump(a + b, open(sys.argv[3], "w")); print(len(a), "+", len(b), "records combined")
PY
uv run python "$EXP_DIR/grade.py" "$EXP_DIR/results_combined.json" "$EXP_DIR/grades_seeds0-5.json" "0,1,2,3,4,5" > "$EXP_DIR/grades_seeds0-5.txt"
uv run python "$EXP_DIR/grade.py" "$EXP_DIR/results.json" "$EXP_DIR/grades_seeds3-5.json" "3,4,5" > "$EXP_DIR/grades_seeds3-5.txt"
echo done

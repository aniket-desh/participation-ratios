#!/usr/bin/env bash
# Batch 2 driver (experiment plan §8, §11). Run from the project root on the worker: bash $EXP_DIR/run.sh
# Phases: (1) LR pilot on the symmetric generator, (2) declared LR selection, (3) nine main runs + three controls,
# (4) checkpoint analysis at n_eval=65536, (5) sample-size check at the final checkpoint, (6) summary and figures.
set -euo pipefail
EXP_DIR=${EXP_DIR:-experiments/transformer-discovery}
R=${R:-runs/transformer-discovery}
NEVAL=${NEVAL:-65536}
CONV=${CONV:-16384 262144}
STEPS=${STEPS:-20000}
PAR=${PAR:-4}
EXTRA=${EXTRA:-}               # extra train flags, smoke tests only                       # concurrent training processes on the one GPU
mkdir -p "$R" "$EXP_DIR/logs"
train() { name=$1; shift; [ -f "$R/$name/ckpt/step_$(printf %06d "$STEPS").pt" ] && return 0
          uv run python -m prt.train --out "$R/$name" --steps "$STEPS" $EXTRA "$@" > "$EXP_DIR/logs/train_$name.log" 2>&1; }
export -f train; export R STEPS EXP_DIR EXTRA

# (1) pilot: q2 = 0.5, pilot seed 100 (disjoint from the main seeds 0-2)
printf '%s\n' 1e-4 3e-4 1e-3 | xargs -P "$PAR" -I{} bash -c 'train pilot_lr{} --q2 0.5 --seed 100 --lr {}'

# (2) declared selection rule, written to lr.json
uv run python "$EXP_DIR/select_lr.py" "$R" > "$EXP_DIR/lr.json"
LR=$(python3 -c "import json;print(json.load(open('$EXP_DIR/lr.json'))['lr'])")
echo "selected lr=$LR"

# (3) main sweep q2 in {0.25, 0.5, 0.8} x seeds {0,1,2}; controls at seed 0
{
  for q2 in 0.25 0.5 0.8; do for s in 0 1 2; do echo "main_q${q2}_s${s} --q2 $q2 --seed $s --lr $LR"; done; done
  echo "ctrl_shuffled_s0 --q2 0.25 --shuffled --seed 0 --lr $LR"
  echo "ctrl_perm777_q0.25_s0 --q2 0.25 --perm_seed 777 --seed 0 --lr $LR"
  echo "ctrl_perm777_q0.8_s0 --q2 0.8 --perm_seed 777 --seed 0 --lr $LR"
} | xargs -P "$PAR" -L1 bash -c 'train "$@"' _

# (4) analysis of every checkpoint of every run (main, controls, pilots)
for d in "$R"/main_*/ "$R"/ctrl_*/ "$R"/pilot_*/; do     # main runs first (priority if time runs short)
  [ -d "$d" ] || continue; n=$(basename "$d"); [ -f "$d/analysis/step_$(printf %06d "$STEPS").json" ] && continue
  uv run python -m prt.analysis "$d" --n_eval "$NEVAL" > "$EXP_DIR/logs/analysis_$n.log" 2>&1
done

# (5) evaluation-sample convergence at the final checkpoint of the main runs (plan §9.4)
for d in "$R"/main_*/; do for n in $CONV; do
  [ -f "$d/analysis_n$n/step_$(printf %06d "$STEPS").json" ] && continue
  uv run python -m prt.analysis "$d" --n_eval $n --tag n$n --steps "$STEPS" >> "$EXP_DIR/logs/analysis_conv.log" 2>&1
done; done

# (6) small summary + figures into the experiment folder
uv run python "$EXP_DIR/summarize.py" "$R" "$EXP_DIR/results.json"
uv run python "$EXP_DIR/plot.py" "$EXP_DIR/results.json" "$EXP_DIR/figures"
echo done

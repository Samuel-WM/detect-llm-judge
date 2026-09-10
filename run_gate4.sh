#!/bin/sh
# Round 5 Gate 4: the three stochastic real reward-model arms, trained and scored serially.
set -e
PY=.venv/Scripts/python.exe
for ARM in RM_OA-deberta RM_gpt2-helpful RM_gpt2-harmless; do
  echo "=========== $ARM train  $(date) ==========="
  $PY -m src.train.round5_attribution --train --arm "$ARM"
  echo "=========== $ARM score  $(date) ==========="
  $PY -m src.train.round5_attribution --score --arm "$ARM"
done
echo "=========== GATE 4 RUNS COMPLETE  $(date) ==========="

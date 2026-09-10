#!/bin/sh
# Round 6 tier arm: train then score, serially, under the GPU lock.
set -e
ARM="$1"
PY=.venv/Scripts/python.exe
echo "=========== $ARM train  $(date) ==========="
$PY -m src.train.round5_attribution --train --arm "$ARM"
echo "=========== $ARM score  $(date) ==========="
$PY -m src.train.round5_attribution --score --arm "$ARM"
echo "=========== $ARM COMPLETE  $(date) ==========="

#!/usr/bin/env bash
set -euo pipefail
cd /home/leo/code/TADA_SIGMOD
export TA_OPENAI_API_KEY=dummy
python3 bench/run_cuad_tada_rq1.py \
  --clause "License Grant" \
  --rows-per-class 60 \
  --seeds 1,2,3 \
  --model gpt-5.5 \
  --out-root bench/results/RQ1/cuad_tada_gpt55 \
  > bench/results/RQ1/cuad_x3.log 2>&1
echo "ALL CUAD RUNS DONE"

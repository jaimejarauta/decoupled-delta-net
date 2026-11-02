#!/bin/sh
# usage: run_wave.sh PROBLEM...   (2 LLM runs + 3 classical seeds per problem, 1500 s)
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1; PY=../.venv-nova/bin/python
for p in "$@"; do
  for r in a b; do $PY evolve.py $p ${p}_$r 36 1500 runs/main/evo_${p}_$r.json > runs/main/evo_${p}_$r.log 2>&1 & done
  for s in 0 1 2; do $PY baseline.py $p $s 1500 runs/main/base_${p}_$s.json > runs/main/base_${p}_$s.log 2>&1 & done
done
wait

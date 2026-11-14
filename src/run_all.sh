#!/usr/bin/env bash
# 본문·보강 분석을 순서대로 돌린다. 저장소 루트에서 실행. 수집(data/)이 먼저 끝나 있어야 한다.
set -u
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
for n in 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 52 43; do
  echo "=== final$n $(date +%H:%M:%S)"
  $PY src/final$n.py > results/final$n.stdout 2>&1 && echo "ok" || echo "FAIL final$n"
done
echo "=== done $(date +%H:%M:%S)"

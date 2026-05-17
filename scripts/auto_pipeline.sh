#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/project2
source /mnt/d/project2/.venv/bin/activate
mkdir -p logs

TARGET=130
INTERVAL=120

echo "[auto] waiting for candidates to reach ${TARGET} files..."
while true; do
  c=$(python3 -c 'from pathlib import Path; print(len(list(Path("llm_candidates").glob("*.json"))))')
  echo "[auto] candidates_count=${c}"
  if [ "$c" -ge "$TARGET" ]; then
    echo "[auto] candidates ready, starting make run"
    make run
    break
  fi
  sleep "$INTERVAL"
done


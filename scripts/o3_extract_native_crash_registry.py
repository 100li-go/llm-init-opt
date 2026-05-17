"""Extract CUTEst native crash registry from logs and runs outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd

from src.native_crash import (
    build_crash_registry,
    parse_log_crash_events,
    parse_runs_crash_events,
    save_registry_json,
)


def _parse_codes(raw: str) -> Set[int]:
    out = set()
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.add(int(chunk))
    if not out:
        out = {-6, -11}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract native crash registry (-6/-11) from logs and runs")
    p.add_argument("--runs", default="results/runs.parquet", help="runs path (.parquet or .csv)")
    p.add_argument("--logs-dir", default="logs", help="directory that contains *.log files")
    p.add_argument("--codes", default="-6,-11", help="comma-separated exit codes to track")
    p.add_argument("--output", default="results/native_crash_registry.json", help="output json path")
    return p.parse_args()


def _load_runs_df(runs_path: Path):
    if not runs_path.exists() or not runs_path.is_file():
        return None
    suffix = runs_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(runs_path)
    if suffix == ".parquet":
        return pd.read_parquet(runs_path)
    raise ValueError(f"Unsupported runs file format: {runs_path}")


def main() -> None:
    args = parse_args()
    target_codes = _parse_codes(args.codes)
    logs_dir = Path(args.logs_dir)
    runs_path = Path(args.runs)
    out_path = Path(args.output)

    log_paths = sorted(logs_dir.glob("*.log")) if logs_dir.exists() else []
    events = parse_log_crash_events(log_paths, target_codes)

    try:
        runs_df = _load_runs_df(runs_path)
    except Exception as e:
        runs_df = None
        print(f"[WARN] Skip runs parsing: {type(e).__name__}: {e}")

    if runs_df is not None:
        events.extend(parse_runs_crash_events(runs_df, target_codes))

    registry = build_crash_registry(events, target_codes)
    save_registry_json(registry, out_path)

    print(f"[DONE] events={registry['event_count']}, problems={registry['problem_count']}, output={out_path}")
    if registry["problem_count"]:
        print("[TOP] first 10 problems by event_count")
        for item in registry["problems"][:10]:
            print(
                f"  {item['problem']}: events={item['event_count']}, "
                f"codes={item['crash_codes']}, sources={','.join(item['sources'])}"
            )


if __name__ == "__main__":
    main()


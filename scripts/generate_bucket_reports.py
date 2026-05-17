"""Generate route-key bucket reports and Wilcoxon tests from runs data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.config import Config
from src.aggregator import build_problem_arm_summary, build_bucket_report
from src.analysis.stats import wilcoxon_by_bucket
from src.analysis.buckets import BUCKETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate bucket-level reports from runs.parquet")
    p.add_argument("--config", default=None, help="Optional config yaml path")
    p.add_argument("--runs", default=None, help="Optional runs parquet path")
    p.add_argument("--out", default=None, help="Optional output directory")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace, cfg):
    runs_path = Path(args.runs) if args.runs else Path(cfg.results_dir) / "runs.parquet"
    multi_cfg = cfg.llm.get("multi_output", {}) if isinstance(cfg.llm, dict) else {}
    enabled = bool(multi_cfg.get("enabled", False))
    suffix = str(multi_cfg.get("output_suffix", "_llmK5"))
    if enabled and args.runs is None:
        runs_path = Path(str(runs_path.parent).rstrip("/\\") + suffix) / runs_path.name

    if args.out:
        out_dir = Path(args.out)
    else:
        reports_dir = cfg.paths.get("reports_dir") if isinstance(cfg.paths, dict) else None
        out_dir = Path(reports_dir) / "buckets" if reports_dir else Path("results/reports/buckets")
        if enabled:
            out_dir = Path(str(out_dir.parent).rstrip("/\\") + suffix) / out_dir.name

    return runs_path, out_dir


def main() -> int:
    args = _parse_args()
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    cfg = Config.from_yaml(config_path)
    runs_path, out_dir = _resolve_paths(args, cfg)

    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs file: {runs_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.read_parquet(runs_path)

    problem_arm = build_problem_arm_summary(runs_df)
    valid_buckets = {b["name"] for b in BUCKETS}
    problem_arm = problem_arm[problem_arm["bucket"].isin(valid_buckets)].copy()

    bucket_report = build_bucket_report(problem_arm)
    present_sources = set(problem_arm.get("init_source", pd.Series(dtype=str)).astype(str).unique())
    llm_left = "llm_post" if "llm_post" in present_sources else ("llm" if "llm" in present_sources else None)
    random_baselines = [src for src in ["random_raw", "random_post", "random"] if src in present_sources]
    wilcoxon_outputs = []
    if llm_left is not None:
        for baseline in random_baselines:
            wil = wilcoxon_by_bucket(problem_arm, left=llm_left, right=baseline)
            wilcoxon_outputs.append((f"wilcoxon_{llm_left}_vs_{baseline}.parquet", wil))
    wil_lcx = wilcoxon_by_bucket(problem_arm, left=(llm_left or "llm"), right="cutest")

    f_problem_arm = out_dir / "problem_arm_summary.parquet"
    f_bucket = out_dir / "bucket_report.parquet"
    f_w_lcx = out_dir / f"wilcoxon_{(llm_left or 'llm')}_vs_cutest.parquet"

    problem_arm.to_parquet(f_problem_arm, index=False)
    bucket_report.to_parquet(f_bucket, index=False)
    wil_lcx.to_parquet(f_w_lcx, index=False)
    generated_wil_files = []
    for fname, df_w in wilcoxon_outputs:
        f = out_dir / fname
        df_w.to_parquet(f, index=False)
        generated_wil_files.append(f)

    print("Bucket problem counts:")
    counts = problem_arm.groupby("bucket")["problem"].nunique().sort_index()
    for b, n in counts.items():
        print(f"  {b}: {int(n)}")

    print("Generated files:")
    for f in [f_problem_arm, f_bucket, f_w_lcx, *generated_wil_files]:
        print(f"  {f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Step 4: 聚合分析，输出汇总表 + 图表"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import CFG
from src.aggregator import (
    build_summary,
    build_route_key_stats,
    build_route_key_problem_stats,
    build_problem_arm_summary_v2,
    build_route_key_problem_stats_v2,
)
from src.analysis.stats import compare_success_rate, compare_nfev, compare_all_arms
from src.analysis.performance_profile import performance_profile
from src.analysis.report import generate_report


def _resolve_mode_dirs(cfg):
    multi_cfg = cfg.llm.get("multi_output", {}) if isinstance(cfg.llm, dict) else {}
    enabled = bool(multi_cfg.get("enabled", False))
    suffix = str(multi_cfg.get("output_suffix", "_llmK5"))

    results_dir = cfg.results_dir
    figures_dir = Path(cfg.paths["figures_dir"])
    if enabled:
        results_dir = Path(str(results_dir).rstrip("/\\") + suffix)
        figures_dir = Path(str(figures_dir).rstrip("/\\") + suffix)
    return enabled, suffix, results_dir, figures_dir

if __name__ == "__main__":
    enabled, suffix, results_dir, figures_dir_cfg = _resolve_mode_dirs(CFG)
    runs_path = results_dir / "runs.parquet"
    if not runs_path.exists():
        raise FileNotFoundError(
            f"Missing runs file: {runs_path}. "
            "Run experiments first: PYTHONPATH=/mnt/d/project2 python scripts/o3_run_experiments.py"
        )
    runs = pd.read_parquet(runs_path)
    summary = build_summary(runs)
    summary_v2 = build_problem_arm_summary_v2(runs, cfg=CFG)
    route_stats = build_route_key_stats(runs)
    route_problem_stats = build_route_key_problem_stats(runs)
    route_problem_stats_v2 = build_route_key_problem_stats_v2(summary_v2)
    stats_all_arms = compare_all_arms(summary_v2)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] multi_output_enabled={enabled}, suffix={suffix}, results_dir={results_dir}")
    summary.to_parquet(results_dir / "summary.parquet", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    summary_v2.to_parquet(results_dir / "summary_problem_arm_v2.parquet", index=False)
    summary_v2.to_csv(results_dir / "summary_problem_arm_v2.csv", index=False)
    route_stats.to_parquet(results_dir / "route_key_stats.parquet", index=False)
    route_stats.to_csv(results_dir / "route_key_stats.csv", index=False)
    route_problem_stats.to_parquet(results_dir / "route_key_problem_stats.parquet", index=False)
    route_problem_stats.to_csv(results_dir / "route_key_problem_stats.csv", index=False)
    route_problem_stats_v2.to_parquet(results_dir / "route_key_problem_stats_v2.parquet", index=False)
    route_problem_stats_v2.to_csv(results_dir / "route_key_problem_stats_v2.csv", index=False)
    stats_all_arms.to_parquet(results_dir / "stats_all_arms.parquet", index=False)
    stats_all_arms.to_csv(results_dir / "stats_all_arms.csv", index=False)

    print("=== RQ1: Success Rate ===")
    print(compare_success_rate(summary))
    print("=== RQ2: nfev ===")
    print(compare_nfev(summary))
    print("=== All arm comparisons (problem-level, capped metrics included) ===")
    print(stats_all_arms.to_string(index=False))

    fig_dir = figures_dir_cfg
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Performance Profile (nfev)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for has_bounds, label, ax in zip([False, True], ["Unbounded", "Bounded"], axes):
        performance_profile(
            summary[summary["has_bounds"] == has_bounds],
            metric="best_nfev", ax=ax, tau_max=10.0
        )
        ax.set_title(f"{label} — nfev")
    fig.tight_layout()
    fig.savefig(fig_dir / "perf_profile_nfev.pdf", dpi=150)

    # Performance Profile (time)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for has_bounds, label, ax in zip([False, True], ["Unbounded", "Bounded"], axes2):
        performance_profile(
            summary[summary["has_bounds"] == has_bounds],
            metric="best_time", ax=ax, tau_max=10.0
        )
        ax.set_title(f"{label} — time")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "perf_profile_time.pdf", dpi=150)
    print("Figures saved.")

    # Generate text report
    report_path = results_dir / "report.txt"
    report_kwargs = {
        "route_problem_stats": route_problem_stats,
        "stats_all_arms": stats_all_arms,
        "out_path": report_path,
    }
    report = generate_report(summary, **report_kwargs)
    print(report)
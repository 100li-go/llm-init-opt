"""Step 4: 聚合分析，输出汇总表 + 图表"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import CFG
from src.aggregator import build_summary
from src.analysis.stats import compare_success_rate, compare_nfev
from src.analysis.performance_profile import performance_profile

if __name__ == "__main__":
    runs = pd.read_parquet(CFG.results_dir / "runs.parquet")
    summary = build_summary(runs)
    summary.to_parquet(CFG.results_dir / "summary.parquet", index=False)
    summary.to_csv(CFG.results_dir / "summary.csv", index=False)

    print("=== RQ1: Success Rate ===")
    print(compare_success_rate(summary))
    print("=== RQ2: nfev ===")
    print(compare_nfev(summary))

    fig_dir = Path(CFG.paths["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Performance Profile (nfev)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cat, ax in zip(["A", "B"], axes):
        performance_profile(
            summary[summary["category"] == cat],
            metric="best_nfev", ax=ax, tau_max=10.0
        )
        ax.set_title(f"Category {cat} — nfev")
    fig.tight_layout()
    fig.savefig(fig_dir / "perf_profile_nfev.pdf", dpi=150)

    # Performance Profile (time)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for cat, ax in zip(["A", "B"], axes2):
        performance_profile(
            summary[summary["category"] == cat],
            metric="best_time", ax=ax, tau_max=10.0
        )
        ax.set_title(f"Category {cat} — time")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "perf_profile_time.pdf", dpi=150)
    print("Figures saved.")
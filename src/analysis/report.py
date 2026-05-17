"""
职责：读取 summary.parquet 和统计检验结果，打印/保存文字报告。
"""
import pandas as pd
from pathlib import Path
from typing import Optional
from src.analysis.stats import compare_success_rate, compare_nfev


def generate_report(
    summary: pd.DataFrame,
    route_problem_stats: Optional[pd.DataFrame] = None,
    stats_all_arms: Optional[pd.DataFrame] = None,
    out_path: Path = None,
) -> str:
    source_set = set(summary["init_source"].astype(str))
    present_sources = [s for s in ["cutest", "random_raw", "random_post", "random", "llm_raw", "llm_post", "llm"] if s in source_set]
    llm_col = "llm_post" if "llm_post" in source_set else ("llm" if "llm" in source_set else None)

    lines = []
    lines.append("=" * 60)
    lines.append("阶段一结论报告")
    lines.append("=" * 60)

    # 总体成功题数
    for src in present_sources:
        s = summary[summary["init_source"] == src]
        n_any = int(s["success_any"].sum())
        lines.append(f"[{src}] 至少成功一次的题数: {n_any}/{len(s)}")

    lines.append("")

    # RQ1
    try:
        rq1 = compare_success_rate(summary)
        lines.append(
            f"RQ1 成功率检验 (Wilcoxon): stat={rq1['wilcoxon_stat']:.4f}, "
            f"p={rq1['p_value']:.4f}, n={rq1['n']}"
        )
    except Exception as e:
        lines.append(f"RQ1 检验失败: {e}")

    # RQ2
    try:
        rq2 = compare_nfev(summary)
        lines.append(
            f"RQ2 nfev检验 (Wilcoxon): stat={rq2['wilcoxon_stat']:.4f}, "
            f"p={rq2['p_value']:.4f}, median_ratio={rq2['median_ratio']:.4f}, n={rq2['n']}"
        )
    except Exception as e:
        lines.append(f"RQ2 检验失败: {e}")

    # 按是否有 bounds 分析
    lines.append("")
    for has_bounds, label in [(False, "Unbounded"), (True, "Bounded")]:
        sub = summary[summary["has_bounds"] == has_bounds]
        lines.append(f"{label}:")
        for src in present_sources:
            s = sub[sub["init_source"] == src]
            med = s["best_nfev"].median()
            lines.append(
                f"  [{src}] success_any={int(s['success_any'].sum())}, "
                f"median_best_nfev={med:.1f}"
            )

    if route_problem_stats is not None and len(route_problem_stats):
        lines.append("")
        lines.append("Route-level (per-problem weighted) success_any_rate deltas from LLM")
        wide = route_problem_stats.pivot_table(
            index=["route_key", "constraint_tag", "objective_tag"],
            columns="init_source",
            values="problem_success_any_rate",
            aggfunc="first",
        ).reset_index()
        for baseline in ["random_raw", "random_post", "random"]:
            if llm_col in wide.columns and baseline in wide.columns:
                delta_col = f"delta_{llm_col}_minus_{baseline}"
                wide[delta_col] = wide[llm_col] - wide[baseline]
                wide_sorted = wide.sort_values(delta_col, ascending=False)
                lines.append(f"  {llm_col} - {baseline}:")
                for _, row in wide_sorted.iterrows():
                    lines.append(
                        f"    {row['route_key']}: {row[delta_col]:+.3f} "
                        f"({llm_col}={row[llm_col]:.3f}, {baseline}={row[baseline]:.3f})"
                    )

    if stats_all_arms is not None and len(stats_all_arms):
        lines.append("")
        lines.append("All-arm pairwise stats (problem-level)")
        for _, row in stats_all_arms.iterrows():
            lines.append(
                f"  {row['metric']} | {row['compare_pair']}: p={row['p_value']:.4g}, "
                f"paired_n={int(row['paired_n'])}/{int(row['pair_total_n'])}, "
                f"median_diff={row['median_diff']:.4g}"
            )

    report = "\n".join(lines)
    if out_path:
        out_path.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_path}")
    return report

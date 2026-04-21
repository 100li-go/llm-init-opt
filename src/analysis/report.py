"""
职责：读取 summary.parquet 和统计检验结果，打印/保存文字报告。
"""
import pandas as pd
from pathlib import Path
from src.analysis.stats import compare_success_rate, compare_nfev


def generate_report(summary: pd.DataFrame, out_path: Path = None) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("阶段一结论报告")
    lines.append("=" * 60)

    # 总体成功题数
    for src in ["cutest", "random", "llm"]:
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

    # 分类分析
    lines.append("")
    for cat in ["A", "B"]:
        sub = summary[summary["category"] == cat]
        lines.append(f"Category {cat}:")
        for src in ["cutest", "random", "llm"]:
            s = sub[sub["init_source"] == src]
            med = s["best_nfev"].median()
            lines.append(
                f"  [{src}] success_any={int(s['success_any'].sum())}, "
                f"median_best_nfev={med:.1f}"
            )

    report = "\n".join(lines)
    if out_path:
        out_path.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_path}")
    return report

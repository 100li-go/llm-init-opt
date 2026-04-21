"""
职责：绘制 Dolan-Moré Performance Profile。
支持 nfev 和 time_sec 两套指标，支持多 solver（LLM/Random/CUTEst）。
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def performance_profile(summary: pd.DataFrame, metric: str = "best_nfev",
                        sources: list = None, ax=None, tau_max: float = 10.0):
    """
    metric: 列名，如 best_nfev 或 best_time
    sources: init_source 列表，如 ['cutest','random','llm']
    """
    if sources is None:
        sources = ["cutest", "random", "llm"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    # pivot: rows=problem, cols=source
    pivot = summary.pivot_table(index="problem", columns="init_source",
                                values=metric, aggfunc="first")
    pivot = pivot[sources].dropna()

    best = pivot.min(axis=1)            # 每题最佳solver的值
    taus = np.logspace(0, np.log10(tau_max), 500)

    for src in sources:
        ratios = (pivot[src] / best).values
        rho = [(ratios <= t).mean() for t in taus]
        ax.semilogx(taus, rho, label=src, linewidth=1.8)

    ax.set_xlabel(f"Performance ratio τ  (metric: {metric})")
    ax.set_ylabel("Fraction of problems ρ(τ)")
    ax.set_title(f"Performance Profile — {metric}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(1, tau_max)
    ax.set_ylim(0, 1.05)
    return ax
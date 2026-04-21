"""
职责：读取 runs.parquet，输出每道题 × 每个 init_source 的汇总指标表。

关键接口：
  build_summary(runs_df) -> pd.DataFrame
  columns: problem, category, init_source,
           success_rate, success_any,
           best_nfev, median_nfev,
           best_time, median_time,
           best_f_final, fallback_rate
"""
import pandas as pd
import numpy as np

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (problem, cat, src), g in df.groupby(["problem", "category", "init_source"]):
        succ = g[g["success"] & g["is_f_final_finite"]]
        n_total = len(g)
        n_succ = len(succ)
        rows.append({
            "problem": problem,
            "category": cat,
            "init_source": src,
            "n_runs": n_total,
            "success_rate": n_succ / n_total if n_total > 0 else 0.0,
            "success_any": n_succ > 0,
            "best_nfev": succ["nfev"].min() if n_succ > 0 else np.nan,
            "median_nfev": succ["nfev"].median() if n_succ > 0 else np.nan,
            "best_time": succ["time_sec"].min() if n_succ > 0 else np.nan,
            "median_time": succ["time_sec"].median() if n_succ > 0 else np.nan,
            "best_f_final": succ["f_final"].min() if n_succ > 0 else np.nan,
            "fallback_rate": g["fallback"].mean(),
        })
    return pd.DataFrame(rows)
"""
职责：读取 runs.parquet，输出每道题 × 每个 init_source 的汇总指标表。

关键接口：
  build_summary(runs_df) -> pd.DataFrame
  columns: problem, has_bounds, init_source,
           success_rate, success_any,
           best_nfev, median_nfev,
           best_time, median_time,
           best_f_final, fallback_rate
"""
import pandas as pd
import numpy as np
from src.analysis.buckets import assign_bucket


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "analysis") and isinstance(cfg.analysis, dict):
        return cfg.analysis.get(key, default)
    if hasattr(cfg, "llm") and isinstance(cfg.llm, dict):
        return cfg.llm.get(key, default)
    return default


def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (problem, has_bounds, src), g in df.groupby(["problem", "has_bounds", "init_source"]):
        succ = g[g["success"] & g["is_f_final_finite"]]
        n_total = len(g)
        n_succ = len(succ)
        rows.append({
            "problem": problem,
            "has_bounds": bool(has_bounds),
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


def build_route_key_stats(df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    group_cols = ["route_key", "constraint_tag", "objective_tag", "init_source"]
    for keys, g in df.groupby(group_cols):
        route_key, ctag, otag, src = keys
        succ = g[g["success"] & g["is_f_final_finite"]]
        n_total = len(g)
        n_succ = len(succ)
        rows.append(
            {
                "route_key": route_key,
                "constraint_tag": ctag,
                "objective_tag": otag,
                "init_source": src,
                "n_runs": n_total,
                "success_rate": n_succ / n_total if n_total > 0 else 0.0,
                "primary_hit_rate": float(g["primary_hit"].mean()) if n_total > 0 else 0.0,
                "backup_trigger_rate": float(g["backup_triggered"].mean()) if n_total > 0 else 0.0,
                "median_nfev": succ["nfev"].median() if n_succ > 0 else np.nan,
                "median_time": succ["time_sec"].median() if n_succ > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_route_key_problem_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Route-level stats with equal per-problem weighting.

    This avoids run-count bias when arms have different replicate counts
    (e.g., random K=10 while cutest/llm usually K=1).
    """
    per_problem_rows = []
    problem_cols = [
        "problem",
        "route_key",
        "constraint_tag",
        "objective_tag",
        "init_source",
    ]

    for keys, g in df.groupby(problem_cols):
        problem, route_key, ctag, otag, src = keys
        succ = g[g["success"] & g["is_f_final_finite"]]
        n_total = len(g)
        n_succ = len(succ)
        per_problem_rows.append(
            {
                "problem": problem,
                "route_key": route_key,
                "constraint_tag": ctag,
                "objective_tag": otag,
                "init_source": src,
                "success_rate": n_succ / n_total if n_total > 0 else 0.0,
                "success_any": n_succ > 0,
                "best_nfev": succ["nfev"].min() if n_succ > 0 else np.nan,
                "best_time": succ["time_sec"].min() if n_succ > 0 else np.nan,
                "best_f_final": succ["f_final"].min() if n_succ > 0 else np.nan,
            }
        )

    if not per_problem_rows:
        return pd.DataFrame()

    per_problem_df = pd.DataFrame(per_problem_rows)
    rows = []
    group_cols = ["route_key", "constraint_tag", "objective_tag", "init_source"]
    for keys, g in per_problem_df.groupby(group_cols):
        route_key, ctag, otag, src = keys
        rows.append(
            {
                "route_key": route_key,
                "constraint_tag": ctag,
                "objective_tag": otag,
                "init_source": src,
                "n_problems": int(len(g)),
                "problem_weighted_success_rate": float(g["success_rate"].mean()),
                "problem_success_any_rate": float(g["success_any"].mean()),
                "problem_median_best_nfev": float(g["best_nfev"].median()),
                "problem_median_best_time": float(g["best_time"].median()),
                "problem_median_best_f_final": float(g["best_f_final"].median()),
            }
        )

    return pd.DataFrame(rows)


def build_problem_arm_summary_v2(
    df: pd.DataFrame,
    cfg=None,
    time_cap_sec: float = None,
    nfev_cap: float = None,
) -> pd.DataFrame:
    """Problem-level arm summary with failure-inclusive capped costs.

    Output granularity: problem x init_source (plus route metadata).
    """
    time_cap_sec = float(time_cap_sec if time_cap_sec is not None else _cfg_get(cfg, "time_cap_sec", 60.0))
    nfev_cap = float(nfev_cap if nfev_cap is not None else _cfg_get(cfg, "nfev_cap", 1_000_000.0))

    work = df.copy()
    work["nfev"] = pd.to_numeric(work.get("nfev"), errors="coerce")
    work["time_sec"] = pd.to_numeric(work.get("time_sec"), errors="coerce")
    work["f_final"] = pd.to_numeric(work.get("f_final"), errors="coerce")
    work["success"] = work.get("success", False).astype(bool)
    finite_col = work.get("is_f_final_finite")
    if finite_col is None:
        finite_col = pd.Series(np.isfinite(work["f_final"].to_numpy()), index=work.index)
    finite_col = finite_col.astype(bool)

    is_success = work["success"] & finite_col
    work["time_cost_capped"] = np.where(is_success, work["time_sec"], time_cap_sec)
    work["nfev_cost_capped"] = np.where(is_success, work["nfev"], nfev_cap)

    rows = []
    group_cols = [
        "problem",
        "has_bounds",
        "route_key",
        "constraint_tag",
        "objective_tag",
        "init_source",
    ]
    for keys, g in work.groupby(group_cols):
        problem, has_bounds, route_key, ctag, otag, src = keys
        succ = g[(g["success"]) & (g["f_final"].notna()) & finite_col.loc[g.index]]
        n_total = int(len(g))
        n_succ = int(len(succ))

        is_random_family = str(src).startswith("random")
        total_time = float(g["time_sec"].sum()) if is_random_family else np.nan
        total_nfev = float(g["nfev"].sum()) if is_random_family else np.nan
        is_llm_family = str(src) in {"llm", "llm_post", "llm_raw"}
        llm_total_time = float(g["time_sec"].sum()) if is_llm_family else np.nan
        llm_total_nfev = float(g["nfev"].sum()) if is_llm_family else np.nan

        rows.append(
            {
                "problem": problem,
                "has_bounds": bool(has_bounds),
                "route_key": route_key,
                "constraint_tag": ctag,
                "objective_tag": otag,
                "init_source": src,
                "n_runs": n_total,
                "success_rate": (n_succ / n_total) if n_total > 0 else 0.0,
                "success_any": n_succ > 0,
                "best_nfev": succ["nfev"].min() if n_succ > 0 else np.nan,
                "best_time": succ["time_sec"].min() if n_succ > 0 else np.nan,
                "best_f_final": succ["f_final"].min() if n_succ > 0 else np.nan,
                "total_time_sec": total_time,
                "total_nfev": total_nfev,
                "llm_total_time_sec": llm_total_time,
                "llm_total_nfev": llm_total_nfev,
                "best_time_cost_capped": float(g["time_cost_capped"].min()) if n_total > 0 else np.nan,
                "best_nfev_cost_capped": float(g["nfev_cost_capped"].min()) if n_total > 0 else np.nan,
                "fallback_rate": float(g["fallback"].mean()) if "fallback" in g.columns and n_total > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_route_key_problem_stats_v2(problem_arm_df: pd.DataFrame) -> pd.DataFrame:
    """Route-level problem-weighted aggregation from problem-level summary."""
    rows = []
    group_cols = ["route_key", "constraint_tag", "objective_tag", "init_source"]
    for keys, g in problem_arm_df.groupby(group_cols):
        route_key, ctag, otag, src = keys
        row = {
            "route_key": route_key,
            "constraint_tag": ctag,
            "objective_tag": otag,
            "init_source": src,
            "n_problems": int(len(g)),
            "problem_weighted_success_rate": float(g["success_any"].mean()) if len(g) else np.nan,
            "problem_median_best_nfev": float(g["best_nfev"].median()),
            "problem_median_best_time": float(g["best_time"].median()),
            "problem_median_best_f_final": float(g["best_f_final"].median()),
            "problem_median_best_nfev_cost_capped": float(g["best_nfev_cost_capped"].median()),
            "problem_median_best_time_cost_capped": float(g["best_time_cost_capped"].median()),
            "problem_median_total_time_sec": float(g["total_time_sec"].median()) if str(src).startswith("random") else np.nan,
            "problem_median_total_nfev": float(g["total_nfev"].median()) if str(src).startswith("random") else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_problem_arm_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build problem-level arm summary with bucket labels.

    Aggregation granularity: bucket x problem x init_source.
    """
    df = assign_bucket(runs_df)

    problem_col = _pick_col(df, "problem", "name")
    if problem_col is None:
        raise ValueError("runs dataframe missing both 'problem' and 'name' columns")
    if "init_source" not in df.columns:
        raise ValueError("runs dataframe missing required 'init_source' column")

    c_col = _pick_col(df, "ConstraintTag", "constraint_tag")
    o_col = _pick_col(df, "ObjectiveTag", "objective_tag")

    if c_col is None:
        df["ConstraintTag"] = np.nan
        c_col = "ConstraintTag"
    if o_col is None:
        df["ObjectiveTag"] = np.nan
        o_col = "ObjectiveTag"

    if "success" not in df.columns:
        df["success"] = False
    if "time_sec" not in df.columns:
        df["time_sec"] = np.nan
    if "nfev" not in df.columns:
        df["nfev"] = np.nan
    if "f_final" not in df.columns:
        df["f_final"] = np.nan

    finite_col = df.get("is_f_final_finite")
    if finite_col is None:
        finite_col = pd.Series(np.isfinite(pd.to_numeric(df["f_final"], errors="coerce")), index=df.index)
    finite_col = finite_col.astype(bool)

    success_col = df["success"].astype(bool)

    rows = []
    group_cols = ["bucket", problem_col, "init_source"]
    for keys, g in df.groupby(group_cols):
        bucket, problem, src = keys
        mask = success_col.loc[g.index] & finite_col.loc[g.index]
        succ = g.loc[mask]

        n_runs = int(len(g))
        success_any = bool(mask.any())

        row = {
            "bucket": bucket,
            "problem": problem,
            "init_source": src,
            "ConstraintTag": g[c_col].iloc[0] if len(g) else np.nan,
            "ObjectiveTag": g[o_col].iloc[0] if len(g) else np.nan,
            "route_key": g["route_key"].iloc[0] if "route_key" in g.columns and len(g) else np.nan,
            "n_runs": n_runs,
            "success_any": success_any,
            "best_time_sec": pd.to_numeric(succ["time_sec"], errors="coerce").min() if len(succ) else np.nan,
            "best_nfev": pd.to_numeric(succ["nfev"], errors="coerce").min() if len(succ) else np.nan,
            "best_f_final": pd.to_numeric(succ["f_final"], errors="coerce").min() if len(succ) else np.nan,
            "backup_trigger_rate": float(g["backup_triggered"].mean()) if "backup_triggered" in g.columns else np.nan,
            "fallback_rate": float(g["fallback"].mean()) if "fallback" in g.columns else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_bucket_report(problem_arm_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Problem-weighted bucket x arm report."""
    rows = []
    group_cols = ["bucket", "init_source"]
    for keys, g in problem_arm_summary_df.groupby(group_cols):
        bucket, src = keys
        rows.append(
            {
                "bucket": bucket,
                "init_source": src,
                "problem_weighted_success_rate": float(g["success_any"].mean()) if len(g) else np.nan,
                "problem_median_best_time_sec": float(g["best_time_sec"].median()),
                "problem_median_best_nfev": float(g["best_nfev"].median()),
                "problem_median_best_f_final": float(g["best_f_final"].median()),
                "problem_mean_backup_trigger_rate": float(g["backup_trigger_rate"].mean()) if "backup_trigger_rate" in g.columns else np.nan,
                "problem_mean_fallback_rate": float(g["fallback_rate"].mean()) if "fallback_rate" in g.columns else np.nan,
                "n_problems": int(g["problem"].nunique()),
            }
        )
    return pd.DataFrame(rows)



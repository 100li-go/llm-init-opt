"""
职责：成对统计检验（LLM vs Random），回答 RQ1–RQ4。
"""
import numpy as np
import pandas as pd
from scipy import stats


def _pick_random_baseline(summary: pd.DataFrame) -> str | None:
    present = set(summary.get("init_source", pd.Series(dtype=str)).astype(str).unique())
    if "random_raw" in present:
        return "random_raw"
    if "random" in present:
        return "random"
    return None


def _pick_llm_arm(summary: pd.DataFrame) -> str | None:
    present = set(summary.get("init_source", pd.Series(dtype=str)).astype(str).unique())
    if "llm_post" in present:
        return "llm_post"
    if "llm" in present:
        return "llm"
    return None


def _wilcoxon_safe(left: pd.Series, right: pd.Series, alternative: str):
    if len(left) == 0:
        return np.nan, np.nan
    left = pd.to_numeric(left, errors="coerce").astype(float)
    right = pd.to_numeric(right, errors="coerce").astype(float)
    try:
        stat, p = stats.wilcoxon(left, right, alternative=alternative, zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        # Happens when all paired differences are zero.
        return 0.0, 1.0


def _pair_metric(summary: pd.DataFrame, left_src: str, right_src: str, metric: str):
    l = summary[summary["init_source"] == left_src].set_index("problem")
    r = summary[summary["init_source"] == right_src].set_index("problem")
    common = l.index.intersection(r.index)
    if len(common) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0, 0
    l = l.loc[common]
    r = r.loc[common]
    mask = l[metric].notna() & r[metric].notna()
    l = l.loc[mask, metric]
    r = r.loc[mask, metric]
    return l, r, int(mask.sum()), int(len(common))


def compare_all_arms(summary: pd.DataFrame) -> pd.DataFrame:
    """Pairwise arm comparison over problem-level summary metrics."""
    metrics = {
        "success_any": "greater",
        "best_nfev": "less",
        "best_time": "less",
        "best_f_final": "less",
        "best_nfev_cost_capped": "less",
        "best_time_cost_capped": "less",
    }
    present = set(summary.get("init_source", pd.Series(dtype=str)).astype(str).unique())
    llm_arm = _pick_llm_arm(summary)
    pairs = []
    if llm_arm is not None and "llm_raw" in present:
        pairs.append((llm_arm, "llm_raw"))
    if llm_arm is not None and "random_raw" in present:
        pairs.append((llm_arm, "random_raw"))
    if llm_arm is not None and "random_post" in present:
        pairs.append((llm_arm, "random_post"))
    if llm_arm is not None and "random" in present:
        pairs.append((llm_arm, "random"))
    if llm_arm is not None and "cutest" in present:
        pairs.append((llm_arm, "cutest"))
    if "random_raw" in present and "cutest" in present:
        pairs.append(("random_raw", "cutest"))
    if "random_post" in present and "cutest" in present:
        pairs.append(("random_post", "cutest"))
    if "random_raw" in present and "random_post" in present:
        pairs.append(("random_raw", "random_post"))
    if "random" in present and "cutest" in present:
        pairs.append(("random", "cutest"))

    total_n = int(summary["problem"].nunique()) if "problem" in summary.columns else 0
    llm_present_n = int(summary[summary["init_source"] == llm_arm]["problem"].nunique()) if (total_n and llm_arm is not None) else 0
    llm_present_rate = (llm_present_n / total_n) if total_n > 0 else np.nan

    rows = []
    for metric, alt in metrics.items():
        if metric not in summary.columns:
            continue
        for left, right in pairs:
            l, r, paired_n, pair_total = _pair_metric(summary, left, right, metric)
            stat, p = _wilcoxon_safe(l, r, alternative=alt)
            l_num = l.astype(float)
            r_num = r.astype(float)
            med_l = float(l_num.median()) if len(l_num) else np.nan
            med_r = float(r_num.median()) if len(r_num) else np.nan
            med_diff = float((l_num - r_num).median()) if len(l_num) else np.nan
            med_ratio = float((l_num / r_num).median()) if len(l_num) and np.all(r_num != 0) else np.nan

            rows.append(
                {
                    "metric": metric,
                    "compare_pair": f"{left}_vs_{right}",
                    "alternative": alt,
                    "wilcoxon_stat": stat,
                    "p_value": p,
                    "paired_n": paired_n,
                    "pair_total_n": pair_total,
                    "median_left": med_l,
                    "median_right": med_r,
                    "median_diff": med_diff,
                    "median_ratio": med_ratio,
                    "llm_present_rate": llm_present_rate,
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_by_bucket(problem_arm_summary_df: pd.DataFrame, left: str = "llm", right: str = "random") -> pd.DataFrame:
    """Run paired Wilcoxon tests by bucket at problem granularity."""
    metric_cfg = [
        ("success_any", "greater"),
        ("best_time_sec", "less"),
        ("best_nfev", "less"),
        ("best_f_final", "less"),
    ]

    rows = []
    left_eff = left
    if left == "llm":
        left_eff = _pick_llm_arm(problem_arm_summary_df) or "llm"

    for bucket, sub in problem_arm_summary_df.groupby("bucket"):
        ldf = sub[sub["init_source"] == left_eff].set_index("problem")
        rdf = sub[sub["init_source"] == right].set_index("problem")

        total_l = int(len(ldf))
        total_r = int(len(rdf))
        common = ldf.index.intersection(rdf.index)

        for metric, alternative in metric_cfg:
            if metric not in sub.columns:
                rows.append(
                    {
                        "bucket": bucket,
                        "metric": metric,
                        "left": left_eff,
                        "right": right,
                        "paired_n": 0,
                        "total_n_left": total_l,
                        "total_n_right": total_r,
                        "p_value": np.nan,
                        "median_left": np.nan,
                        "median_right": np.nan,
                        "median_diff": np.nan,
                        "error": f"metric_missing:{metric}",
                    }
                )
                continue

            l = ldf.loc[common, metric] if len(common) else pd.Series(dtype=float)
            r = rdf.loc[common, metric] if len(common) else pd.Series(dtype=float)

            if metric != "success_any":
                mask = l.notna() & r.notna()
                l = l[mask]
                r = r[mask]
            else:
                l = l.fillna(False).astype(bool)
                r = r.fillna(False).astype(bool)

            paired_n = int(len(l))
            if paired_n == 0:
                rows.append(
                    {
                        "bucket": bucket,
                        "metric": metric,
                        "left": left_eff,
                        "right": right,
                        "paired_n": 0,
                        "total_n_left": total_l,
                        "total_n_right": total_r,
                        "p_value": np.nan,
                        "median_left": np.nan,
                        "median_right": np.nan,
                        "median_diff": np.nan,
                        "error": "no_paired_samples",
                    }
                )
                continue

            l_num = l.astype(float)
            r_num = r.astype(float)
            stat, p = _wilcoxon_safe(l_num, r_num, alternative=alternative)

            rows.append(
                {
                    "bucket": bucket,
                    "metric": metric,
                    "left": left_eff,
                    "right": right,
                    "paired_n": paired_n,
                    "total_n_left": total_l,
                    "total_n_right": total_r,
                    "p_value": p,
                    "median_left": float(l_num.median()),
                    "median_right": float(r_num.median()),
                    "median_diff": float((l_num - r_num).median()),
                    "error": "" if np.isfinite(p) else "wilcoxon_failed",
                    "wilcoxon_stat": stat,
                }
            )

    return pd.DataFrame(rows)

def compare_success_rate(summary: pd.DataFrame):
    """RQ1：LLM vs Random 成功率，McNemar 或 Wilcoxon"""
    random_src = _pick_random_baseline(summary)
    llm_src = _pick_llm_arm(summary)
    if random_src is None:
        raise ValueError("No random baseline arm found (expected random_raw or random)")
    if llm_src is None:
        raise ValueError("No LLM arm found (expected llm_post or llm)")
    r = summary[summary["init_source"] == random_src].set_index("problem")
    l = summary[summary["init_source"] == llm_src].set_index("problem")
    common = r.index.intersection(l.index)
    r, l = r.loc[common], l.loc[common]
    stat, p = stats.wilcoxon(l["success_rate"], r["success_rate"],
                              alternative="greater", zero_method="wilcox")
    return {"wilcoxon_stat": stat, "p_value": p, "n": len(common)}

def compare_nfev(summary: pd.DataFrame):
    """RQ2：在成功样本上，LLM best_nfev vs Random best_nfev"""
    random_src = _pick_random_baseline(summary)
    llm_src = _pick_llm_arm(summary)
    if random_src is None:
        raise ValueError("No random baseline arm found (expected random_raw or random)")
    if llm_src is None:
        raise ValueError("No LLM arm found (expected llm_post or llm)")
    r = summary[summary["init_source"] == random_src].set_index("problem")
    l = summary[summary["init_source"] == llm_src].set_index("problem")
    common = r.index.intersection(l.index)
    r, l = r.loc[common], l.loc[common]
    mask = r["best_nfev"].notna() & l["best_nfev"].notna()
    r, l = r[mask], l[mask]
    stat, p = stats.wilcoxon(l["best_nfev"], r["best_nfev"],
                              alternative="less", zero_method="wilcox")
    ratio = (l["best_nfev"] / r["best_nfev"]).median()
    return {"wilcoxon_stat": stat, "p_value": p, "median_ratio": ratio, "n": mask.sum()}
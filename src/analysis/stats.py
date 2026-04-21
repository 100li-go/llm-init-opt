"""
职责：成对统计检验（LLM vs Random），回答 RQ1–RQ4。
"""
import numpy as np
import pandas as pd
from scipy import stats

def compare_success_rate(summary: pd.DataFrame):
    """RQ1：LLM vs Random 成功率，McNemar 或 Wilcoxon"""
    r = summary[summary["init_source"] == "random"].set_index("problem")
    l = summary[summary["init_source"] == "llm"].set_index("problem")
    common = r.index.intersection(l.index)
    r, l = r.loc[common], l.loc[common]
    stat, p = stats.wilcoxon(l["success_rate"], r["success_rate"],
                              alternative="greater", zero_method="wilcox")
    return {"wilcoxon_stat": stat, "p_value": p, "n": len(common)}

def compare_nfev(summary: pd.DataFrame):
    """RQ2：在成功样本上，LLM best_nfev vs Random best_nfev"""
    r = summary[summary["init_source"] == "random"].set_index("problem")
    l = summary[summary["init_source"] == "llm"].set_index("problem")
    common = r.index.intersection(l.index)
    r, l = r.loc[common], l.loc[common]
    mask = r["best_nfev"].notna() & l["best_nfev"].notna()
    r, l = r[mask], l[mask]
    stat, p = stats.wilcoxon(l["best_nfev"], r["best_nfev"],
                              alternative="less", zero_method="wilcox")
    ratio = (l["best_nfev"] / r["best_nfev"]).median()
    return {"wilcoxon_stat": stat, "p_value": p, "median_ratio": ratio, "n": mask.sum()}
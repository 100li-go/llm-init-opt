"""
职责：给定已加载的 pycutest 问题对象，提取适合填入 Prompt 的结构化摘要。
输出为纯 Python dict（可 json.dumps），不含大向量（超过 20 维只给统计量）。

关键接口：
  extract_meta(p, category) -> dict
"""
import numpy as np
import pycutest
from typing import Optional

def extract_meta(p, category: str) -> dict:
    x0 = np.array(p.x0, dtype=float)
    n = p.n
    bl = np.array(p.bl, dtype=float) if p.bl is not None else np.full(n, -np.inf)
    bu = np.array(p.bu, dtype=float) if p.bu is not None else np.full(n, +np.inf)

    # ── x0 统计 ──────────────────────────────────
    x0_stats = {
        "min": float(np.min(x0)),
        "max": float(np.max(x0)),
        "mean": float(np.mean(x0)),
        "std": float(np.std(x0)),
        "norm": float(np.linalg.norm(x0)),
        # 仅当 n<=20 时附带完整向量（方便LLM）
        "values": x0.tolist() if n <= 20 else None,
    }

    # ── bounds 统计 ───────────────────────────────
    has_lb = np.isfinite(bl)
    has_ub = np.isfinite(bu)
    widths = (bu - bl)[has_lb & has_ub]
    bounds_stats = {
        "n_with_finite_lb": int(has_lb.sum()),
        "n_with_finite_ub": int(has_ub.sum()),
        "n_with_both_bounds": int((has_lb & has_ub).sum()),
        "width_median": float(np.median(widths)) if len(widths) > 0 else None,
        "width_min": float(np.min(widths)) if len(widths) > 0 else None,
        "width_max": float(np.max(widths)) if len(widths) > 0 else None,
    }

    # ── 全局尺度 S（供 LLM 参考，与 initializer 一致）──
    if len(widths) > 0:
        S = float(np.clip(np.median(widths), 1e-3, 1e6))
    else:
        S = float(max(1.0, np.median(np.abs(x0)) + 1.0))

    return {
        "problem_name": p.name,
        "n": n,
        "category": category,
        "x0_stats": x0_stats,
        "bounds_stats": bounds_stats,
        "scale_S": S,
    }
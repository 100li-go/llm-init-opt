"""
职责：基于 CUTEst 问题对象与 problem_properties 计算细粒度标签。

关键接口：
  classify_tags(p, props, tol=1e-10, eq_majority=0.8) -> dict
"""
from __future__ import annotations

import numpy as np

INF_SENTINEL = 1.0e20


def _as_array(value, n: int, fill: float) -> np.ndarray:
    if value is None:
        return np.full(n, fill, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        # CUTEst 常规不会触发；异常时回退为全无界，避免崩溃。
        return np.full(n, fill, dtype=float)
    return arr


def _is_finite(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (np.abs(arr) < INF_SENTINEL * 0.999)


def _as_bool_array(value, m: int, default: bool) -> np.ndarray:
    if value is None:
        return np.full(m, default, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 1 or arr.shape[0] != m:
        return np.full(m, default, dtype=bool)
    return arr


def _has_box_constraints(bl: np.ndarray, bu: np.ndarray) -> bool:
    return bool(np.any(_is_finite(bl)) or np.any(_is_finite(bu)))


def classify_constraint_tag(p, tol: float = 1e-10, eq_majority: float = 0.8) -> tuple[str, bool, bool]:
    """返回 (ConstraintTag, has_box, constraints_all_linear)。"""
    n = int(p.n)
    m = int(getattr(p, "m", 0) or 0)

    bl = _as_array(getattr(p, "bl", None), n, -np.inf)
    bu = _as_array(getattr(p, "bu", None), n, np.inf)
    has_box = _has_box_constraints(bl, bu)

    if m <= 0:
        return ("B" if has_box else "U", has_box, True)

    cl = _as_array(getattr(p, "cl", None), m, -np.inf)
    cu = _as_array(getattr(p, "cu", None), m, np.inf)

    # m>0 但所有约束均为 (-inf, +inf) 视为无一般约束
    active = _is_finite(cl) | _is_finite(cu)
    if not np.any(active):
        return ("B" if has_box else "U", has_box, True)

    is_linear = _as_bool_array(getattr(p, "is_linear_cons", None), m, default=True)
    if np.any(~is_linear[active]):
        return ("NLC", has_box, False)

    is_eq = _as_bool_array(getattr(p, "is_eq_cons", None), m, default=False)
    # 缺少 is_eq_cons 时，用 cl≈cu 回退。
    if getattr(p, "is_eq_cons", None) is None:
        is_eq = np.isclose(cl, cu, atol=tol, rtol=0.0)

    eq_ratio = float(np.mean(is_eq[active]))
    if np.all(is_eq[active]) or eq_ratio >= eq_majority:
        return ("LEB", has_box, True)
    return ("LCB", has_box, True)


def classify_objective_tag(props: dict, constraints_all_linear: bool) -> str:
    objective = str((props or {}).get("objective", "") or "").strip().lower()
    degree = (props or {}).get("degree", None)
    regular = (props or {}).get("regular", True)

    if objective == "sum of squares":
        return "LS"

    try:
        degree_value = int(degree)
    except Exception:
        degree_value = None

    if degree_value == 2 and constraints_all_linear:
        # QP is merged into SmoothNLP to keep route buckets compact and consistent.
        return "SmoothNLP"

    if isinstance(regular, str):
        reg = regular.strip().lower() not in {"false", "0", "no"}
    else:
        reg = bool(regular)
    if reg is False:
        return "Nonsmooth"

    return "SmoothNLP"


def classify_tags(p, props: dict, tol: float = 1e-10, eq_majority: float = 0.8) -> dict:
    ctag, has_box, all_linear = classify_constraint_tag(p, tol=tol, eq_majority=eq_majority)
    otag = classify_objective_tag(props or {}, constraints_all_linear=all_linear)
    return {
        "ConstraintTag": ctag,
        "ObjectiveTag": otag,
        "route_key": f"{ctag}:{otag}",
        "has_box": has_box,
        "constraints_all_linear": all_linear,
    }


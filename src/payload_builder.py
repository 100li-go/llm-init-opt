"""Build token-budget payload for pycutest warm-start prompt generation."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


DEFAULT_JACOBIAN_TOPK = 8
DEFAULT_ROW_NNZ_CAP = 20
DEFAULT_JAC_ABS_TOL = 1e-12
DEFAULT_EQ_TOL = 1e-12
DEFAULT_VIOL_TOL = 1e-8
DEFAULT_INF_BOUND_THRESHOLD = 1e20
DEFAULT_PAYLOAD_FULL_N_MAX = 120
DEFAULT_PAYLOAD_FULL_M_MAX = 80
DEFAULT_PAYLOAD_HEAD_K = 50
DEFAULT_PAYLOAD_FULL_CONSTRAINT_M_MAX = 80


def _cfg_get(cfg: Any, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "llm") and isinstance(cfg.llm, dict):
        return cfg.llm.get(key, default)
    return default


def _load_default_cfg():
    try:
        from src.config import CFG

        return CFG
    except Exception:
        return None


def _safe_array(value, n: int, fill: float) -> np.ndarray:
    if value is None:
        return np.full(n, fill, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        return np.full(n, fill, dtype=float)
    return arr


def _safe_bool_array(value, n: int, default: bool) -> np.ndarray:
    if value is None:
        return np.full(n, default, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 1 or arr.shape[0] != n:
        return np.full(n, default, dtype=bool)
    return arr


def _is_finite_bound(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.isfinite(values) & (np.abs(values) < threshold)


def _constraint_violation(c0: np.ndarray, cl: np.ndarray, cu: np.ndarray) -> np.ndarray:
    v_low = np.maximum(cl - c0, 0.0)
    v_high = np.maximum(c0 - cu, 0.0)
    return np.maximum(v_low, v_high)


def _sense(cl_i: float, cu_i: float, eq_tol: float, inf_thr: float) -> str:
    if abs(cl_i - cu_i) <= eq_tol:
        return "eq"
    cl_fin = np.isfinite(cl_i) and abs(cl_i) < inf_thr
    cu_fin = np.isfinite(cu_i) and abs(cu_i) < inf_thr
    if not cl_fin and cu_fin:
        return "le"
    if cl_fin and not cu_fin:
        return "ge"
    if cl_fin and cu_fin:
        return "range"
    return "ineq"


def _row_to_sparse(row: np.ndarray, abs_tol: float, nnz_cap: int) -> list[dict]:
    nz = np.where(np.abs(row) > abs_tol)[0]
    if nz.size == 0:
        return []

    if nz.size > nnz_cap:
        vals = np.abs(row[nz])
        sel = np.argpartition(vals, -nnz_cap)[-nnz_cap:]
        nz = nz[sel]

    order = np.argsort(-np.abs(row[nz]))
    nz = nz[order]
    return [{"j": int(j), "val": float(row[j])} for j in nz]


def _classify_constraint_tag(m: int, has_bounds: bool, cl: np.ndarray, cu: np.ndarray, is_linear_cons: np.ndarray) -> str:
    if m == 0:
        return "B" if has_bounds else "U"

    if np.any(~is_linear_cons):
        return "NLC"

    is_eq = np.isclose(cl, cu, atol=DEFAULT_EQ_TOL, rtol=0.0)
    return "LEB" if bool(np.all(is_eq)) else "LCB"


def _classify_objective_tag(props: Optional[dict]) -> str:
    objective = str((props or {}).get("objective", "")).strip().lower()
    return "LS" if objective == "sum of squares" else "SmoothNLP"


def _normalize_problem_properties(props: Optional[dict]) -> Optional[dict]:
    if props is None:
        return None
    return dict(props)


def _build_constraints_payload(
    p,
    x0: np.ndarray,
    m: int,
    constraint_tag: str,
    jacobian_topk: int,
    row_nnz_cap: int,
    jacobian_abs_tol: float,
    inf_bound_threshold: float,
) -> Optional[dict]:
    if m == 0:
        return {"tag": constraint_tag}

    cl = _safe_array(getattr(p, "cl", None), m, -np.inf)
    cu = _safe_array(getattr(p, "cu", None), m, np.inf)

    c0 = None
    J0 = None
    try:
        c0, J0 = p.cons(x0, gradient=True)
        c0 = np.asarray(c0, dtype=float)
        J0 = np.asarray(J0, dtype=float)
    except Exception:
        try:
            c0 = np.asarray(p.cons(x0), dtype=float)
            _, J0 = p.lagjac(x0)
            J0 = np.asarray(J0, dtype=float)
        except Exception:
            return {
                "tag": constraint_tag,
                "cl": cl.tolist(),
                "cu": cu.tolist(),
                "c0": None,
                "max_violation0": None,
                "sum_violation0": None,
                "n_violated0": None,
                "topk": [],
            }

    violation = _constraint_violation(c0, cl, cu)
    max_violation0 = float(np.max(violation)) if violation.size else 0.0
    sum_violation0 = float(np.sum(violation)) if violation.size else 0.0
    n_violated0 = int(np.sum(violation > DEFAULT_VIOL_TOL))

    k = min(int(max(1, jacobian_topk)), m)
    if m <= k:
        top_idx = np.arange(m)
    else:
        top_idx = np.argpartition(violation, -k)[-k:]
        top_idx = top_idx[np.argsort(-violation[top_idx])]

    top_rows = []
    for i in top_idx:
        s = _sense(float(cl[i]), float(cu[i]), DEFAULT_EQ_TOL, inf_bound_threshold)
        top_rows.append(
            {
                "i": int(i),
                "type": "eq" if s == "eq" else "ineq",
                "sense": s,
                "cl": float(cl[i]),
                "cu": float(cu[i]),
                "c0": float(c0[i]),
                "violation": float(violation[i]),
                "jacobian_row_sparse": _row_to_sparse(J0[int(i), :], jacobian_abs_tol, row_nnz_cap),
            }
        )

    return {
        "tag": constraint_tag,
        "cl": cl.tolist(),
        "cu": cu.tolist(),
        "c0": c0.tolist(),
        "max_violation0": max_violation0,
        "sum_violation0": sum_violation0,
        "n_violated0": n_violated0,
        "topk": top_rows,
    }


def _slice_prefix(values, head_k: int):
    if values is None:
        return None
    return values[: min(len(values), int(max(0, head_k)))]


def build_payload(p, props=None, top_k: int = DEFAULT_JACOBIAN_TOPK, head_k: int = 32, cfg: Any = None) -> Dict[str, Any]:
    """Build LLM prompt payload from a pycutest.CUTEstProblem.

    The output schema is intentionally stable and uses English snake_case keys.
    """
    _ = head_k  # Backward compatibility: prefer config key payload_head_k.
    effective_cfg = cfg or _load_default_cfg()

    jacobian_topk = int(_cfg_get(effective_cfg, "jacobian_topk", top_k))
    row_nnz_cap = int(_cfg_get(effective_cfg, "jacobian_row_nnz_cap", DEFAULT_ROW_NNZ_CAP))
    jacobian_abs_tol = float(_cfg_get(effective_cfg, "jacobian_abs_tol", DEFAULT_JAC_ABS_TOL))
    inf_bound_threshold = float(_cfg_get(effective_cfg, "inf_bound_threshold", DEFAULT_INF_BOUND_THRESHOLD))
    payload_full_n_max = int(_cfg_get(effective_cfg, "payload_full_n_max", DEFAULT_PAYLOAD_FULL_N_MAX))
    payload_full_m_max = int(_cfg_get(effective_cfg, "payload_full_m_max", DEFAULT_PAYLOAD_FULL_M_MAX))
    payload_head_k = int(_cfg_get(effective_cfg, "payload_head_k", DEFAULT_PAYLOAD_HEAD_K))
    payload_full_constraint_m_max = int(
        _cfg_get(effective_cfg, "payload_full_constraint_m_max", DEFAULT_PAYLOAD_FULL_CONSTRAINT_M_MAX)
    )

    x0 = np.asarray(p.x0, dtype=float)
    n = int(p.n)
    m = int(getattr(p, "m", 0) or 0)
    bl = _safe_array(getattr(p, "bl", None), n, -np.inf)
    bu = _safe_array(getattr(p, "bu", None), n, np.inf)

    finite_lb = _is_finite_bound(bl, inf_bound_threshold)
    finite_ub = _is_finite_bound(bu, inf_bound_threshold)
    has_bounds = bool(np.any(finite_lb | finite_ub))

    is_linear_cons = _safe_bool_array(getattr(p, "is_linear_cons", None), m, default=True)
    cl = _safe_array(getattr(p, "cl", None), m, -np.inf) if m > 0 else np.array([], dtype=float)
    cu = _safe_array(getattr(p, "cu", None), m, np.inf) if m > 0 else np.array([], dtype=float)
    constraint_tag = _classify_constraint_tag(m, has_bounds, cl, cu, is_linear_cons)
    objective_tag = _classify_objective_tag(props)
    route_key = f"{constraint_tag}:{objective_tag}"
    use_full_vectors = (n <= payload_full_n_max) and (m <= payload_full_m_max)
    vector_policy = "full" if use_full_vectors else "prefix"

    try:
        f0 = float(p.obj(x0))
    except Exception:
        f0 = None

    try:
        g0 = np.asarray(p.grad(x0), dtype=float)
        gnorm0 = float(np.linalg.norm(g0))
        g0_out = g0.tolist() if use_full_vectors else g0[: min(n, payload_head_k)].tolist()
    except Exception:
        gnorm0 = None
        g0_out = None

    x0_norm2 = float(np.linalg.norm(x0))
    scales = {
        "x0_norm2": x0_norm2,
        "suggested_step_norm": float(min(1.0, 0.25 * x0_norm2 + 1e-6)),
        "max_abs_x_suggested": float(max(10.0, 10.0 * float(np.max(np.abs(x0))), 100.0)),
    }

    constraints_payload = _build_constraints_payload(
        p=p,
        x0=x0,
        m=m,
        constraint_tag=constraint_tag,
        jacobian_topk=jacobian_topk,
        row_nnz_cap=row_nnz_cap,
        jacobian_abs_tol=jacobian_abs_tol,
        inf_bound_threshold=inf_bound_threshold,
    )

    x0_out = x0.tolist() if use_full_vectors else x0[: min(n, payload_head_k)].tolist()
    if has_bounds:
        bl_out = bl.tolist() if use_full_vectors else bl[: min(n, payload_head_k)].tolist()
        bu_out = bu.tolist() if use_full_vectors else bu[: min(n, payload_head_k)].tolist()
    else:
        bl_out = None
        bu_out = None

    if constraints_payload is not None and not use_full_vectors:
        keep_full_constraint_vectors = (
            constraint_tag in {"LEB", "LCB", "NLC"} and m <= payload_full_constraint_m_max
        )
        if not keep_full_constraint_vectors:
            constraints_payload["cl"] = _slice_prefix(constraints_payload.get("cl"), payload_head_k)
            constraints_payload["cu"] = _slice_prefix(constraints_payload.get("cu"), payload_head_k)
            constraints_payload["c0"] = _slice_prefix(constraints_payload.get("c0"), payload_head_k)

    payload: Dict[str, Any] = {
        "name": getattr(p, "name", "UNKNOWN"),
        "n": n,
        "m": m,
        "ConstraintTag": constraint_tag,
        "ObjectiveTag": objective_tag,
        "route_key": route_key,
        "x0": x0_out,
        "bounds": {
            "has_bounds": has_bounds,
            "bl": bl_out,
            "bu": bu_out,
        },
        "objective": {
            "tag": objective_tag,
            "f0": f0,
            "gnorm0": gnorm0,
            "g0": g0_out,
        },
        "constraints": constraints_payload,
        "scales": scales,
    }

    if not use_full_vectors:
        payload["vector_policy"] = vector_policy
        payload["head_k"] = int(payload_head_k)
        payload["n_total"] = int(n)

    problem_properties = _normalize_problem_properties(props)
    if problem_properties is not None:
        payload["problem_properties"] = problem_properties

    return payload


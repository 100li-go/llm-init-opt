"""
职责：校验 LLM 输出的候选 x JSON。

关键接口：
  validate_candidate(raw: dict, n: int) -> dict
"""
from __future__ import annotations

import numpy as np


def _validate_vec(x, n: int) -> list:
    if not isinstance(x, list):
        raise ValueError("LLM output vector must be a list")
    if len(x) != int(n):
        raise ValueError(f"LLM x length mismatch: expected {n}, got {len(x)}")

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != int(n):
        raise ValueError("LLM x must be a 1-D numeric vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError("LLM x contains non-finite values")
    return arr.tolist()


def validate_candidate(raw: dict, n: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("LLM output must be a JSON object")

    x = raw.get("x", None)
    if not isinstance(x, list):
        raise ValueError("LLM output must contain key 'x' with a list value")
    return {"x": _validate_vec(x, n)}


def validate_candidate_multi(raw: dict, n: int, k: int = 5) -> dict:
    """Validate multi-candidate output.

    Accepts:
    - {'xs': [[...], ...]} or {'x_list': [[...], ...]}
    - legacy {'x': [...]} (wrapped to K=1)
    """
    if not isinstance(raw, dict):
        raise ValueError("LLM output must be a JSON object")

    xs = raw.get("xs", None)
    if xs is None:
        xs = raw.get("x_list", None)

    if xs is None and "x" in raw:
        xs = [raw.get("x")]

    if not isinstance(xs, list) or len(xs) == 0:
        raise ValueError("LLM multi output must contain non-empty 'xs' or 'x_list'")
    if len(xs) > int(k):
        xs = xs[: int(k)]

    validated = []
    for i, x in enumerate(xs):
        try:
            validated.append(_validate_vec(x, n))
        except Exception as e:
            raise ValueError(f"LLM xs[{i}] invalid: {e}")

    out = {"xs": validated}
    if "x" in raw and isinstance(raw.get("x"), list):
        out["x"] = raw.get("x")
    for key in ("raw_text", "model"):
        if key in raw:
            out[key] = raw[key]
    return out


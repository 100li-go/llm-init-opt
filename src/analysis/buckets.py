"""Bucket definitions and assignment helpers for route-level reporting."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


BUCKETS = [
    {
        "name": "NLC",
        "constraint_tags": {"NLC"},
        "objective_tags": {"LS", "SmoothNLP"},
    },
    {
        "name": "LEB",
        "constraint_tags": {"LEB"},
        "objective_tags": {"LS", "SmoothNLP"},
    },
    {
        "name": "LCB",
        "constraint_tags": {"LCB"},
        "objective_tags": {"LS", "SmoothNLP"},
    },
    {
        "name": "UB_LS",
        "constraint_tags": {"U", "B"},
        "objective_tags": {"LS"},
    },
    {
        "name": "UB_SmoothNLP",
        "constraint_tags": {"U", "B"},
        "objective_tags": {"SmoothNLP"},
    },
]


def _parse_route_key(route_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if route_key is None:
        return None, None
    text = str(route_key)
    if ":" not in text:
        return None, None
    ctag, otag = text.split(":", 1)
    return ctag.strip() or None, otag.strip() or None


def _assign_bucket_name(constraint_tag: Optional[str], objective_tag: Optional[str]) -> str:
    ctag = str(constraint_tag or "").strip()
    otag = str(objective_tag or "").strip()
    for spec in BUCKETS:
        if ctag in spec["constraint_tags"] and otag in spec["objective_tags"]:
            return spec["name"]
    return "OTHER"


def assign_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized `bucket` column added.

    Accepts either `ConstraintTag`/`ObjectiveTag` or lowercase
    `constraint_tag`/`objective_tag`. If tags are missing, tries `route_key`.
    """
    out = df.copy()

    c_col = "ConstraintTag" if "ConstraintTag" in out.columns else "constraint_tag" if "constraint_tag" in out.columns else None
    o_col = "ObjectiveTag" if "ObjectiveTag" in out.columns else "objective_tag" if "objective_tag" in out.columns else None

    if c_col is None or o_col is None:
        route_vals = out.get("route_key")
        parsed = [
            _parse_route_key(v) if route_vals is not None else (None, None)
            for v in (route_vals.tolist() if route_vals is not None else [None] * len(out))
        ]
        ctags = [x[0] for x in parsed]
        otags = [x[1] for x in parsed]
    else:
        ctags = out[c_col].astype(str).tolist()
        otags = out[o_col].astype(str).tolist()

    out["bucket"] = [_assign_bucket_name(c, o) for c, o in zip(ctags, otags)]
    return out


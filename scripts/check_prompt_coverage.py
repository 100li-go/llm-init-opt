"""收集并断言覆盖目标路由/分层的 Prompt 样本。

本脚本扫描 CUTEst 问题，构建 payload 和 prompt，将匹配的 prompt 写入 JSON 文件，
并输出带有断言结果的汇总表格。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class TargetSpec:
    label: str
    matcher: Callable[[dict, dict], bool]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check prompt coverage targets and write sample prompt.json files")
    p.add_argument("--config", default=None, help="Path to config yaml (optional)")
    p.add_argument(
        "--output-dir",
        default="results/prompt_samples/coverage_check",
        help="Base output directory for prompt samples",
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="Optional cap on scanned problems (0 means no cap)",
    )
    p.add_argument(
        "--use-cutest-catalog",
        action="store_true",
        help="Also scan pycutest.find_problems() catalog after problem_set.json",
    )
    return p.parse_args()


def _load_cfg(config_path: Optional[str]):
    if config_path:
        os.environ["CONFIG_PATH"] = config_path
    from src.config import CFG

    return CFG


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "llm") and isinstance(cfg.llm, dict):
        return cfg.llm.get(key, default)
    return default


def _load_candidate_names(cfg, use_cutest_catalog: bool) -> List[str]:
    names: List[str] = []

    # 1) Prefer already selected problem_set for speed.
    try:
        from src.problem_selector import load_problem_set

        metas = load_problem_set(str(cfg.paths["problem_set"]))
        names.extend(m.name for m in metas)
    except Exception:
        pass

    # 2) Optional full CUTEst catalog expansion.
    if use_cutest_catalog:
        try:
            import pycutest

            groups = ["unconstrained", "bounds", "linear", "quadratic", "other"]
            for g in groups:
                try:
                    names.extend(pycutest.find_problems(constraints=g))
                except Exception:
                    continue
        except Exception:
            pass

    # Preserve order while deduplicating.
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _is_prefix(payload: dict) -> bool:
    return str(payload.get("vector_policy", "full")).lower() == "prefix"


def _is_full(payload: dict) -> bool:
    return not _is_prefix(payload)


def _sense_has_direction(payload: dict) -> bool:
    cons = payload.get("constraints")
    if not isinstance(cons, dict):
        return False
    topk = cons.get("topk") or []
    senses = {str(t.get("sense")) for t in topk if t.get("sense") is not None}
    return len(senses.intersection({"le", "ge", "range"})) > 0


def _constraints_full_kept(payload: dict) -> bool:
    cons = payload.get("constraints")
    if not isinstance(cons, dict):
        return False
    m = int(payload.get("m", 0) or 0)
    cl = cons.get("cl")
    cu = cons.get("cu")
    c0 = cons.get("c0")
    return isinstance(cl, list) and isinstance(cu, list) and isinstance(c0, list) and len(cl) == m and len(cu) == m and len(c0) == m


def _build_target_specs(thresholds: dict) -> List[TargetSpec]:
    n_full = thresholds["payload_full_n_max"]
    m_small = thresholds["payload_full_constraint_m_max"]

    return [
        TargetSpec(
            "U:SmoothNLP full",
            lambda p, t: p.get("route_key") == "U:SmoothNLP"
            and int(p.get("m", 0) or 0) == 0
            and int(p.get("n", 0) or 0) <= n_full
            and _is_full(p),
        ),
        TargetSpec(
            "U:SmoothNLP prefix（n大）",
            lambda p, t: p.get("route_key") == "U:SmoothNLP"
            and int(p.get("m", 0) or 0) == 0
            and int(p.get("n", 0) or 0) > n_full
            and _is_prefix(p),
        ),
        TargetSpec(
            "B:SmoothNLP/B:LS full（has_bounds=true）",
            lambda p, t: p.get("route_key") in {"B:SmoothNLP", "B:LS"}
            and bool((p.get("bounds") or {}).get("has_bounds"))
            and int(p.get("m", 0) or 0) == 0
            and int(p.get("n", 0) or 0) <= n_full
            and _is_full(p),
        ),
        TargetSpec(
            "B:SmoothNLP/B:LS prefix（has_bounds=true，n大）",
            lambda p, t: p.get("route_key") in {"B:SmoothNLP", "B:LS"}
            and bool((p.get("bounds") or {}).get("has_bounds"))
            and int(p.get("m", 0) or 0) == 0
            and int(p.get("n", 0) or 0) > n_full
            and _is_prefix(p),
        ),
        TargetSpec(
            "LEB:SmoothNLP prefix + m小",
            lambda p, t: p.get("route_key") == "LEB:SmoothNLP"
            and int(p.get("m", 0) or 0) > 0
            and int(p.get("n", 0) or 0) > n_full
            and int(p.get("m", 0) or 0) <= m_small
            and _is_prefix(p),
        ),
        TargetSpec(
            "LCB:SmoothNLP prefix + m小",
            lambda p, t: p.get("route_key") == "LCB:SmoothNLP"
            and int(p.get("m", 0) or 0) > 0
            and int(p.get("n", 0) or 0) > n_full
            and int(p.get("m", 0) or 0) <= m_small
            and _is_prefix(p)
            and _sense_has_direction(p),
        ),
        TargetSpec(
            "NLC:LS prefix + m小",
            lambda p, t: p.get("route_key") == "NLC:LS"
            and int(p.get("m", 0) or 0) > 0
            and int(p.get("n", 0) or 0) > n_full
            and int(p.get("m", 0) or 0) <= m_small
            and _is_prefix(p),
        ),
    ]


def _run_assertions(label: str, payload: dict, thresholds: dict) -> List[str]:
    errs: List[str] = []
    n = int(payload.get("n", 0) or 0)
    m = int(payload.get("m", 0) or 0)
    bounds = payload.get("bounds") or {}
    has_bounds = bool(bounds.get("has_bounds"))
    cons = payload.get("constraints")
    vp = payload.get("vector_policy", "full")
    head_k = int(payload.get("head_k", thresholds["payload_head_k"]))
    cap = int(thresholds["jacobian_row_nnz_cap"])

    if _is_full(payload):
        if vp not in ("full", None):
            errs.append("full档 vector_policy 异常")
        if not isinstance(payload.get("x0"), list) or len(payload["x0"]) != n:
            errs.append("full档 x0 长度!=n")
        g0 = (payload.get("objective") or {}).get("g0")
        if g0 is not None and (not isinstance(g0, list) or len(g0) != n):
            errs.append("full档 objective.g0 长度!=n")
        if has_bounds:
            bl = bounds.get("bl")
            bu = bounds.get("bu")
            if not isinstance(bl, list) or len(bl) != n:
                errs.append("full档 has_bounds=true 但 bl 长度!=n")
            if not isinstance(bu, list) or len(bu) != n:
                errs.append("full档 has_bounds=true 但 bu 长度!=n")
    else:
        if str(vp) != "prefix":
            errs.append("prefix档 vector_policy!=prefix")
        if "head_k" not in payload or "n_total" not in payload:
            errs.append("prefix档缺少 head_k/n_total")
        x0 = payload.get("x0")
        if not isinstance(x0, list) or len(x0) != min(head_k, n):
            errs.append("prefix档 x0 长度!=head_k")

    # U/B constraints must be null.
    if payload.get("ConstraintTag") in {"U", "B"}:
        if cons is not None:
            errs.append("U/B 约束类 constraints 应为 null")

    # constrained checks.
    if payload.get("ConstraintTag") in {"LEB", "LCB", "NLC"}:
        if not isinstance(cons, dict):
            errs.append("约束类 constraints 缺失")
        else:
            topk = cons.get("topk") or []
            if len(topk) == 0:
                errs.append("约束类 topk 为空")
            for row in topk:
                sparse = row.get("jacobian_row_sparse") or []
                if len(sparse) > cap:
                    errs.append("jacobian_row_sparse 超过 nnz_cap")
                    break

            if _is_prefix(payload) and m <= int(thresholds["payload_full_constraint_m_max"]):
                if not _constraints_full_kept(payload):
                    errs.append("prefix+m小 但 cl/cu/c0 未全量保留")

    # LCB sense direction requirement.
    if label.startswith("LCB:"):
        if not _sense_has_direction(payload):
            errs.append("LCB 样例 topk 未包含 le/ge/range")

    return errs


def main() -> int:
    args = _parse_args()
    cfg = _load_cfg(args.config)

    thresholds = {
        "payload_full_n_max": int(_cfg_get(cfg, "payload_full_n_max", 120)),
        "payload_full_m_max": int(_cfg_get(cfg, "payload_full_m_max", 80)),
        "payload_head_k": int(_cfg_get(cfg, "payload_head_k", 50)),
        "payload_full_constraint_m_max": int(_cfg_get(cfg, "payload_full_constraint_m_max", 120)),
        "jacobian_row_nnz_cap": int(_cfg_get(cfg, "jacobian_row_nnz_cap", 20)),
    }

    from src.payload_builder import build_payload
    from src.prompt_router import write_prompt_json

    problem_names = _load_candidate_names(cfg, use_cutest_catalog=args.use_cutest_catalog)
    if args.max_problems and args.max_problems > 0:
        problem_names = problem_names[: args.max_problems]

    targets = _build_target_specs(thresholds)
    chosen: Dict[str, Optional[dict]] = {t.label: None for t in targets}
    errors: Dict[str, str] = {t.label: "" for t in targets}

    import pycutest

    for name in problem_names:
        if all(chosen.values()):
            break
        try:
            p = pycutest.import_problem(name)
            props = pycutest.problem_properties(name)
            payload = build_payload(p, props=props, cfg=cfg)
        except Exception:
            continue

        for t in targets:
            if chosen[t.label] is not None:
                continue
            if t.matcher(payload, thresholds):
                chosen[t.label] = payload

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    rows = []
    any_fail = False
    for t in targets:
        payload = chosen[t.label]
        if payload is None:
            rows.append(
                {
                    "target": t.label,
                    "name": "NONE",
                    "route": "NONE",
                    "n": "-",
                    "m": "-",
                    "has_bounds": "-",
                    "vector_policy": "-",
                    "result": "FAIL",
                    "reason": "未找到满足条件的问题（可尝试放宽阈值或增大扫描范围）",
                }
            )
            any_fail = True
            continue

        assert_errors = _run_assertions(t.label, payload, thresholds)
        out_path = write_prompt_json(payload, cfg=cfg, output_dir=out_base)

        if assert_errors:
            any_fail = True
            result = "FAIL"
            reason = "; ".join(assert_errors)
        else:
            result = "PASS"
            reason = f"saved={out_path}"

        rows.append(
            {
                "target": t.label,
                "name": str(payload.get("name", "UNKNOWN")),
                "route": str(payload.get("route_key", "UNKNOWN")),
                "n": str(payload.get("n", "-")),
                "m": str(payload.get("m", "-")),
                "has_bounds": str(bool((payload.get("bounds") or {}).get("has_bounds"))),
                "vector_policy": str(payload.get("vector_policy", "full")),
                "result": result,
                "reason": reason,
            }
        )

    headers = ["target", "name", "route", "n", "m", "has_bounds", "vector_policy", "result", "reason"]
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)

    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(r[h].ljust(widths[h]) for h in headers))

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Prompt routing and prompt-file writing for pycutest warm-start generation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class PromptSpec:
    route_key: str
    system_prompt: str
    user_prompt: str


SYSTEM_PROMPT = (
    "你是数值优化 warm-start 专家。"
    "只输出一个 JSON 数组（单行），例如 [0.1, -1.2, ...]。"
    "不要输出 JSON 对象，不要输出任何解释、注释、markdown。"
    "数组长度必须等于 n，且全部元素必须是有限实数。"
    "当 bounds.has_bounds=true 时，必须满足所有有限边界；"
    "对 |bl|>=1e20 或 |bu|>=1e20 的边界，按无界处理。"
    "优先级：先降低一般约束违反（重点 max_violation），再改进目标值，避免极端值。"
)

SYSTEM_PROMPT_MULTI_TEMPLATE = (
    "你是数值优化 warm-start 专家。"
    "只输出一个 JSON 数组（单行），其长度必须恰好等于 K={k}。"
    "该数组的每个元素都必须是长度为 n 的 JSON 数组。"
    "所有数值必须有限。"
    "当 bounds.has_bounds=true 时，必须满足所有有限边界；"
    "对 |bl|>=1e20 或 |bu|>=1e20 的边界，按无界处理。"
    "禁止输出 JSON 对象、解释文本、注释或 markdown。"
)


SUPPORTED_ROUTE_KEYS = {
    "NLC:SmoothNLP",
    "NLC:LS",
    "LEB:SmoothNLP",
    "LEB:LS",
    "LCB:SmoothNLP",
    "LCB:LS",
    "U:SmoothNLP",
    "U:LS",
    "B:SmoothNLP",
    "B:LS",
}


ROUTE_STRATEGIES = {
    "NLC": (
        "策略：优先降低 max_violation0，允许相对 x0 有一定偏离。\n"
        "优先处理 constraints.topk 中 violation 最大的约束；"
        "利用每条约束的 jacobian_row_sparse 判断变量调整方向与力度。\n"
        "避免极端值，必要时参考 scales.max_abs_x_suggested 控制步长。"
    ),
    "LEB": (
        "策略：优先降低线性等式残差（cl==cu），尽量同时降低 topk 等式 violation。\n"
        "使用 topk 的 jacobian_row_sparse 做协同修正，避免只修复单条约束导致其他等式变差。"
    ),
    "LCB": (
        "策略：优先降低 max_violation0 与 n_violated0。\n"
        "对 sense=le 的约束朝减小约束值方向调整；"
        "对 sense=ge 的约束朝增大约束值方向调整；"
        "对 range 约束朝区间内部移动。"
    ),
    "U": (
        "策略：无一般约束，重点降低 objective.f0 或残差指标；"
        "保持变量尺度稳定，参考 scales。"
    ),
    "B": (
        "策略：仅有边界约束，先确保所有有限边界满足，再优化 objective。\n"
        "可在 x0 附近微调，避免触发过大步长。"
    ),
}

FEASIBLE_ROUTE_STRATEGIES = {
    "NLC": (
        "策略：当前约束已基本可行（max_violation0 接近 0）。"
        "优先保持可行性，不显著增大 violation；"
        "在此基础上改进 objective，并避免极端值。\n"
        "可参考 topk 的 jacobian_row_sparse 做小幅、稳定的方向性调整。"
    ),
    "LEB": (
        "策略：当前线性等式已基本满足（max_violation0 接近 0）。"
        "优先保持等式可行性，不显著增大 violation；"
        "在此基础上改进 objective。\n"
        "可小幅调整并尽量保持 topk 等式接近满足。"
    ),
    "LCB": (
        "策略：当前线性约束已基本可行（max_violation0 接近 0）。"
        "优先保持可行性，不显著增大 violation 与 n_violated0；"
        "在此基础上改进 objective。"
    ),
}


def _multi_candidate_role_hint(constraint_tag: str, k: int) -> str:
    k = int(k)
    if k != 5:
        return (
            f"请输出 {k} 个彼此不同的候选初值，保持数值稳定并满足边界/约束。"
            "优先给出可行且目标改进明显的候选。\n\n"
        )

    if constraint_tag == "U":
        return (
            "5 个候选的分工如下（按顺序给出 5 个数组）：\n"
            "(1) 保守解：尽量接近 x0（步长不超过 scales.suggested_step_norm 的 0.1 倍）。\n"
            "(2) 梯度小步下降：沿 -objective.g0 方向小步（约 0.5×suggested_step_norm）。\n"
            "(3) 梯度中步下降：沿 -objective.g0 方向中步（约 1.0×suggested_step_norm）。\n"
            "(4) 梯度+坐标扰动：在(2)/(3)基础上，对 |g0| 最大的 1~3 个变量加不同符号小扰动。\n"
            "(5) 探索解：与 x0 明显不同但不极端（受 scales.max_abs_x_suggested 约束）。\n"
            "若 objective.g0 缺失，则用围绕 x0 的不同方向小扰动替代 (2)(3)(4)。\n\n"
        )
    if constraint_tag == "B":
        return (
            "5 个候选的分工如下（按顺序给出 5 个数组）：\n"
            "(1) 保守可行解：在边界内尽量接近 x0，优先向边界内侧移动。\n"
            "(2) 梯度小步下降（保持可行）：沿 -objective.g0 小步并满足全部有限边界。\n"
            "(3) 梯度中步下降（保持可行）：沿 -objective.g0 中步并满足全部有限边界。\n"
            "(4) 边界探索解：1~2 个变量接近有限边界（不越界），其余保持温和。\n"
            "(5) 保守+扰动：x0 附近不同方向的小扰动版本，需可行且与前 4 个明显不同。\n"
            "若 objective.g0 缺失，则以边界内不同方向扰动 + 边界探索替代 (2)(3)。\n\n"
        )
    if constraint_tag == "LEB":
        return (
            "5 个候选的分工如下（按顺序给出 5 个数组）：\n"
            "(1) 保守可行解：尽量接近 x0；若 max_violation0 很小则优先保持等式可行性。\n"
            "(2) 等式修复-协同：利用 constraints.topk 的 jacobian_row_sparse 协同降低等式 violation。\n"
            "(3) 等式修复-偏重 top1：优先修复 topk[0]，同时保持总体 max_violation0 下降。\n"
            "(4) 目标改进-可行优先：在(1)/(2)附近沿 -objective.g0 小步改进 objective，避免 violation 显著上升。\n"
            "(5) 探索解：与前 4 个明显不同，且等式 violation 不显著变差。\n"
            "若 objective.g0 缺失，则 (4) 改为不同方向小扰动且保持 violation 不增大。\n\n"
        )
    if constraint_tag == "LCB":
        return (
            "5 个候选的分工如下（按顺序给出 5 个数组）：\n"
            "(1) 保守解：尽量接近 x0，优先不让 max_violation0 变差。\n"
            "(2) 强可行化：优先降低 max_violation0 与 n_violated0，并按 sense=le/ge/range 方向修复。\n"
            "(3) 可行化-偏重 top1：主要修复 topk[0]，与(2)使用不同变量组合/方向。\n"
            "(4) 目标改进-可行约束：在(1)/(2)附近沿 -objective.g0 小步改进 objective，同时保持 violation 不显著上升。\n"
            "(5) 探索解：与前 4 个明显不同，避免极端值，且仍以降低 violation 为优先。\n"
            "若 objective.g0 缺失，则 (4) 改为不同方向小扰动并保持 violation 不显著增大。\n\n"
        )
    if constraint_tag == "NLC":
        return (
            "5 个候选的分工如下（按顺序给出 5 个数组）：\n"
            "(1) 保守解：尽量接近 x0；若 max_violation0 接近 0，则优先保持可行性不变差。\n"
            "(2) 强可行化-协同：利用 topk 的 jacobian_row_sparse 协同降低 max_violation0。\n"
            "(3) 强可行化-另一方向：与(2)不同变量组合/符号方向，仍以降低 violation 为目标。\n"
            "(4) 目标改进-受限可行化：在(2)/(3)附近沿 -objective.g0 小步改进 objective，并保持 violation 不显著变差。\n"
            "(5) 探索解：与前 4 个明显不同，允许相对 x0 更大偏移但避免极端值。\n"
            "若 objective.g0 缺失，则 (4) 改为(2)/(3)附近的不同方向小扰动且保持 violation 不显著增大。\n\n"
        )

    return (
        f"请输出 {k} 个彼此不同的候选初值，并保持数值稳定。"
        "优先满足可行性与目标改进。\n\n"
    )


def _cfg_get(cfg: Any, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "prompts") and isinstance(cfg.prompts, dict):
        return cfg.prompts.get(key, default)
    return default


def _resolve_output_dir(cfg: Any) -> Path:
    prompts_cfg = getattr(cfg, "prompts", {}) if cfg is not None else {}
    if isinstance(prompts_cfg, dict) and prompts_cfg.get("output_dir"):
        return Path(prompts_cfg["output_dir"])

    if cfg is not None and hasattr(cfg, "paths") and isinstance(cfg.paths, dict):
        if cfg.paths.get("llm_prompts_dir"):
            return Path(cfg.paths["llm_prompts_dir"])

    fallback = _cfg_get(cfg, "output_dir", "llm_prompts")
    return Path(fallback)


def _route_strategy(route_key: str, constraint_tag: str, payload: Dict[str, Any]) -> str:
    constraints = payload.get("constraints")
    if isinstance(constraints, dict):
        max_v = constraints.get("max_violation0")
        if max_v is not None and constraint_tag in FEASIBLE_ROUTE_STRATEGIES:
            try:
                if float(max_v) <= 1e-8:
                    return FEASIBLE_ROUTE_STRATEGIES[constraint_tag]
            except (TypeError, ValueError):
                pass

    if route_key in SUPPORTED_ROUTE_KEYS:
        return ROUTE_STRATEGIES.get(constraint_tag, ROUTE_STRATEGIES["U"])
    return (
        "策略：优先可行性与数值稳定性。"
        "若存在约束，先降 violation；否则先改进目标。"
    )


def build_prompt_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact payload for LLM prompts by removing duplicated metadata."""
    view = copy.deepcopy(payload if isinstance(payload, dict) else {})
    constraint_tag = str(
        view.get("ConstraintTag")
        or view.get("constraint_tag")
        or ((str(view.get("route_key", "")).split(":", 1)[0]) if ":" in str(view.get("route_key", "")) else "")
        or "U"
    )

    constraints = view.get("constraints")
    if isinstance(constraints, dict):
        constraints.setdefault("tag", constraint_tag)
    else:
        view["constraints"] = {"tag": constraint_tag}

    for k in ("name", "route_key", "ConstraintTag", "ObjectiveTag", "constraint_tag"):
        view.pop(k, None)

    objective = view.get("objective")
    if isinstance(objective, dict):
        objective.pop("tag", None)

    return view


def select_prompt(
    payload: Dict[str, Any],
    cfg: Any,
    multi_output: bool = False,
    multi_k: int = 5,
    prompt_payload: Dict[str, Any] | None = None,
) -> PromptSpec:
    constraint_tag = str(payload.get("ConstraintTag", "U"))
    objective_tag = str(payload.get("ObjectiveTag", "SmoothNLP"))
    route_key = str(payload.get("route_key") or f"{constraint_tag}:{objective_tag}")

    strategy = _route_strategy(route_key, constraint_tag, payload)
    prefix_hint = ""
    if str(payload.get("vector_policy", "")).lower() == "prefix":
        prefix_hint = (
            "\n注意：x0/bounds/gradient 可能只提供了前缀 head_k 维的信息，"
            "请在输出长度为 n 的数组时保持未给出维度数值温和（例如接近 0 或延续 x0 的均值），"
            "并优先满足 topk 约束与尺度 scales。"
        )
    diversity_hint = _multi_candidate_role_hint(constraint_tag, int(multi_k)) if multi_output else ""
    llm_payload = prompt_payload if isinstance(prompt_payload, dict) else build_prompt_payload(payload)
    user_prompt = (
        f"{strategy}{prefix_hint}\n\n"
        f"{diversity_hint}"
        "下面给出 payload（UTF-8 JSON，按其中字段进行推理）：\n"
        f"{json.dumps(llm_payload, ensure_ascii=False, indent=2)}"
    )

    system_prompt = SYSTEM_PROMPT_MULTI_TEMPLATE.format(k=int(multi_k)) if multi_output else SYSTEM_PROMPT
    return PromptSpec(route_key=route_key, system_prompt=system_prompt, user_prompt=user_prompt)


def build_prompt_record(payload: Dict[str, Any], cfg=None, multi_output: bool = False, multi_k: int = 5) -> Dict[str, Any]:
    prompt_payload = build_prompt_payload(payload)
    spec = select_prompt(payload, cfg, multi_output=multi_output, multi_k=multi_k, prompt_payload=prompt_payload)
    return {
        "name": str(payload.get("name", "UNKNOWN")),
        "route_key": spec.route_key,
        "payload": prompt_payload,
        "system_prompt": spec.system_prompt,
        "user_prompt": spec.user_prompt,
    }


def write_prompt_json(
    payload: Dict[str, Any],
    cfg=None,
    output_dir: str | Path | None = None,
    multi_output: bool = False,
    multi_k: int = 5,
) -> Path:
    record = build_prompt_record(payload, cfg, multi_output=multi_output, multi_k=multi_k)
    base_dir = Path(output_dir) if output_dir is not None else _resolve_output_dir(cfg)
    out_dir = base_dir / record["route_key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['name']}.prompt.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


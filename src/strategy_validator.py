"""
职责：校验 LLM 输出的策略列表，修复可修复的问题，过滤不可修复项。
保证进入 initializer 的策略全部合法。

关键接口：
  validate_and_fix(raw_strategies: list, cfg: Config, has_bounds: bool) -> list[dict]
"""
from typing import List
from src.config import Config

_VALID_MODES = {"near_x0", "center_of_bounds", "interior_from_bounds"}

def validate_and_fix(raw: list, cfg: Config, has_bounds: bool) -> List[dict]:
    init_cfg = cfg.init
    valid = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            continue
        # mode
        mode = s.get("mode", "near_x0")
        if mode not in _VALID_MODES:
            mode = "near_x0"
        if not has_bounds and mode in {"center_of_bounds", "interior_from_bounds"}:
            mode = "near_x0"
        # alpha
        try:
            alpha = float(s.get("alpha", init_cfg["alpha_default"]))
            alpha = max(init_cfg["alpha_min"], min(init_cfg["alpha_max"], alpha))
        except Exception:
            alpha = init_cfg["alpha_default"]
        # sparsity
        sparsity = s.get("sparsity", 1.0)
        choices = init_cfg["sparsity_choices"]
        if sparsity not in choices:
            # snap to nearest
            sparsity = min(choices, key=lambda c: abs(c - float(sparsity)))
        # seed
        try:
            seed = int(s.get("seed", i * 137 + 42))
        except Exception:
            seed = i * 137 + 42
        valid.append({"mode": mode, "alpha": alpha, "sparsity": sparsity, "seed": seed})
    return valid
from src.strategy_validator import validate_and_fix
from src.config import CFG


def test_invalid_mode_fixed():
    raw = [{"mode": "invalid_mode", "alpha": 0.5, "sparsity": 1.0, "seed": 1}]
    result = validate_and_fix(raw, CFG, has_bounds=False)
    assert result[0]["mode"] == "near_x0"


def test_bounds_mode_replaced_when_no_bounds():
    raw = [{"mode": "center_of_bounds", "alpha": 0.5, "sparsity": 1.0, "seed": 1}]
    result = validate_and_fix(raw, CFG, has_bounds=False)
    assert result[0]["mode"] == "near_x0"


def test_alpha_clipped():
    raw = [{"mode": "near_x0", "alpha": 999.0, "sparsity": 1.0, "seed": 1}]
    result = validate_and_fix(raw, CFG, has_bounds=False)
    assert result[0]["alpha"] <= CFG.init["alpha_max"]


def test_sparsity_snapped():
    raw = [{"mode": "near_x0", "alpha": 0.5, "sparsity": 0.5, "seed": 1}]
    result = validate_and_fix(raw, CFG, has_bounds=False)
    assert result[0]["sparsity"] in CFG.init["sparsity_choices"]


def test_empty_input():
    result = validate_and_fix([], CFG, has_bounds=False)
    assert result == []

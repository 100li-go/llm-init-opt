import numpy as np
import pytest
from unittest.mock import MagicMock
from src.initializer import Initializer
from src.config import CFG


def make_mock_problem(n=5, has_bounds=False):
    p = MagicMock()
    p.n = n
    p.m = 0
    p.name = "MOCK"
    p.x0 = np.ones(n)
    if has_bounds:
        p.bl = np.zeros(n)
        p.bu = np.ones(n) * 2.0
    else:
        p.bl = None
        p.bu = None
    p.obj = MagicMock(return_value=1.0)
    return p


def test_get_cutest_does_not_crash():
    p = make_mock_problem(n=5, has_bounds=False)
    init = Initializer(p, "A", CFG)
    result = init.get_cutest()
    assert result is not None


def test_get_random_count():
    p = make_mock_problem(n=5, has_bounds=False)
    init = Initializer(p, "A", CFG)
    results = init.get_random(K=3)
    assert len(results) == 3


def test_get_llm_near_x0():
    p = make_mock_problem(n=10, has_bounds=False)
    init = Initializer(p, "A", CFG)
    strategies = [{"mode": "near_x0", "alpha": 0.5, "sparsity": 1.0, "seed": 42}]
    results = init.get_llm(strategies)
    assert len(results) == 1


def test_clip_applied_for_category_b():
    p = make_mock_problem(n=5, has_bounds=True)
    init = Initializer(p, "B", CFG)
    results = init.get_random(K=5)
    for r in results:
        assert np.all(r.x0 >= 0.0 - 1e-9)
        assert np.all(r.x0 <= 2.0 + 1e-9)

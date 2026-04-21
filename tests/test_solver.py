import numpy as np
from unittest.mock import MagicMock
from src.solver import solve
from src.initializer import InitResult
from src.config import CFG


def make_rosenbrock_problem():
    """用简单的 Rosenbrock 函数模拟 pycutest 问题对象"""
    p = MagicMock()
    p.n = 2
    p.bl = None
    p.bu = None

    def obj(x, gradient=False):
        x = np.asarray(x, dtype=float)
        f = (1 - x[0]) ** 2 + (x[1] - x[0] ** 2) ** 2
        if gradient:
            g = np.array([
                -2 * (1 - x[0]) - 4 * x[0] * (x[1] - x[0] ** 2),
                2 * (x[1] - x[0] ** 2),
            ])
            return f, g
        return f

    p.obj = obj
    return p


def test_solve_category_a_returns_record():
    p = make_rosenbrock_problem()
    init = InitResult(x0=np.array([0.0, 0.0]), f0=2.0, is_f0_finite=True)
    rec = solve(p, init, "A", CFG)
    assert rec.nfev >= 0
    assert rec.time_sec >= 0


def test_solve_returns_record_on_exception():
    p = MagicMock()
    p.n = 2
    p.bl = None
    p.bu = None
    p.obj = MagicMock(side_effect=RuntimeError("bad problem"))
    init = InitResult(x0=np.array([0.0, 0.0]), f0=1.0, is_f0_finite=True)
    rec = solve(p, init, "A", CFG)
    assert rec.success is False
    assert rec.exception_type != ""

"""
职责：
  统一封装 scipy.optimize.minimize 调用，含超时保护、异常捕获、
  返回结构化 RunRecord（不含 pandas 依赖）。

关键接口：
  solve(p, x0_init: InitResult, category: str, cfg: Config) -> SolveRecord
"""
import time
import signal
import numpy as np
import scipy.optimize as spopt
from dataclasses import dataclass
from src.initializer import InitResult
from src.config import Config

@dataclass
class SolveRecord:
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    njev: int
    f_final: float
    is_f_final_finite: bool
    time_sec: float
    exception_type: str = ""

def _timeout_handler(signum, frame):
    raise TimeoutError("solver timeout")

def make_grad(p):
    """兼容不同版本 pycutest 的梯度接口"""
    def grad(x):
        try:
            _, g = p.obj(x, gradient=True)
        except TypeError:
            g = p.grad(x)
        return g
    return grad

def solve(p, init: InitResult, category: str, cfg: Config) -> SolveRecord:
    solver_cfg = cfg.solver[f"category_{category}"]
    method = solver_cfg["method"]
    maxiter = solver_cfg["maxiter"]

    # 构造 options & bounds
    if method == "BFGS":
        options = {"maxiter": maxiter, "gtol": solver_cfg["gtol"]}
        bounds = None
    else:  # L-BFGS-B
        options = {"maxiter": maxiter, "ftol": solver_cfg["ftol"]}
        bounds = list(zip(
            p.bl if p.bl is not None else [None]*p.n,
            p.bu if p.bu is not None else [None]*p.n,
        ))

    # 梯度函数（兼容不同版本 pycutest）
    grad = make_grad(p)

    t0 = time.perf_counter()
    try:
        # 超时保护（Unix only）
        timeout = cfg.solver.get("timeout_sec", 120)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        except (AttributeError, OSError):
            pass  # Windows 下跳过

        res = spopt.minimize(
            fun=p.obj,
            x0=init.x0,
            jac=grad,
            method=method,
            bounds=bounds,
            options=options,
        )
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass

        elapsed = time.perf_counter() - t0
        return SolveRecord(
            success=bool(res.success),
            status=int(res.status),
            message=str(res.message),
            nit=int(res.nit),
            nfev=int(res.nfev),
            njev=int(getattr(res, "njev", 0)),
            f_final=float(res.fun),
            is_f_final_finite=bool(np.isfinite(res.fun)),
            time_sec=elapsed,
        )
    except TimeoutError:
        return SolveRecord(
            success=False, status=-1, message="timeout",
            nit=0, nfev=0, njev=0,
            f_final=float("nan"), is_f_final_finite=False,
            time_sec=time.perf_counter() - t0,
            exception_type="TimeoutError",
        )
    except Exception as e:
        return SolveRecord(
            success=False, status=-2, message=str(e),
            nit=0, nfev=0, njev=0,
            f_final=float("nan"), is_f_final_finite=False,
            time_sec=time.perf_counter() - t0,
            exception_type=type(e).__name__,
        )
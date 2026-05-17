"""
职责：
  统一封装求解器调用，含超时保护、异常捕获，
  返回结构化 RunRecord（不含 pandas 依赖）。

关键接口：
  solve(p, x0_init: InitResult, has_bounds: bool, cfg: Config, constraint_tag: str = "U", objective_tag: str = "SmoothNLP") -> SolveRecord
"""
import signal
import time
from dataclasses import dataclass

import numpy as np
import scipy.optimize as spopt

from src.config import Config
from src.initializer import InitResult


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
    solver_method: str = ""
    route_key: str = ""
    solver_chain: str = ""
    primary_solver: str = ""
    primary_hit: bool = False
    backup_triggered: bool = False


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


def _build_bounds(p):
    return list(
        zip(
            p.bl if p.bl is not None else [None] * p.n,
            p.bu if p.bu is not None else [None] * p.n,
        )
    )


def _solve_minimize(p, init_x: np.ndarray, method_alias: str, cfg: Config) -> spopt.OptimizeResult:
    grad = make_grad(p)
    method_upper = method_alias.upper()

    if method_upper == "BFGS":
        method = "BFGS"
        options = {
            "maxiter": int(cfg.solver.get("category_A", {}).get("maxiter", 500)),
            "gtol": float(cfg.solver.get("category_A", {}).get("gtol", 1e-6)),
        }
        bounds = None
    elif method_upper == "L-BFGS-B":
        method = "L-BFGS-B"
        options = {
            "maxiter": int(cfg.solver.get("category_B", {}).get("maxiter", 500)),
            "ftol": float(cfg.solver.get("category_B", {}).get("ftol", 1e-12)),
        }
        bounds = _build_bounds(p)
    elif method_upper == "SQP":
        method = "SLSQP"
        options = {
            "maxiter": int(cfg.solver.get("category_B", {}).get("maxiter", 500)),
            "ftol": float(cfg.solver.get("category_B", {}).get("ftol", 1e-9)),
        }
        bounds = _build_bounds(p)
    elif method_upper == "IPOPT":
        # SciPy 无 IPOPT，使用 trust-constr 近似其内点行为。
        method = "trust-constr"
        options = {
            "maxiter": int(cfg.solver.get("category_B", {}).get("maxiter", 500)),
            "gtol": float(cfg.solver.get("category_A", {}).get("gtol", 1e-6)),
            "xtol": 1e-10,
            "verbose": 0,
        }
        bounds = _build_bounds(p)
    else:
        raise ValueError(f"Unsupported solver alias: {method_alias}")

    return spopt.minimize(
        fun=p.obj,
        x0=init_x,
        jac=grad,
        method=method,
        bounds=bounds,
        options=options,
    )


def _solve_least_squares(p, init_x: np.ndarray) -> spopt.OptimizeResult:
    # 优先使用残差接口；否则退化为单残差 sqrt(max(f,0))。
    if hasattr(p, "res"):
        def residual(xx):
            return np.asarray(p.res(xx), dtype=float)
    else:
        def residual(xx):
            f = float(p.obj(xx))
            return np.array([np.sqrt(max(0.0, f))], dtype=float)

    lb = np.asarray(p.bl, dtype=float) if p.bl is not None else np.full(p.n, -np.inf)
    ub = np.asarray(p.bu, dtype=float) if p.bu is not None else np.full(p.n, np.inf)

    return spopt.least_squares(
        fun=residual,
        x0=init_x,
        bounds=(lb, ub),
        max_nfev=500,
        method="trf",
    )


def _normalize_chain(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _select_solver_chain(has_bounds: bool, cfg: Config, constraint_tag: str, objective_tag: str) -> list[str]:
    policy = cfg.solver.get("policy", {})
    exact = policy.get("route_exact", {})
    by_constraint = policy.get("route_constraint", {})

    route_key = f"{constraint_tag}:{objective_tag}"
    chain = _normalize_chain(exact.get(route_key, []))
    if not chain:
        chain = _normalize_chain(by_constraint.get(constraint_tag, []))

    if not chain:
        if objective_tag == "LS" and constraint_tag in {"U", "B"}:
            chain = ["LEAST_SQUARES"]
        elif has_bounds:
            chain = ["L-BFGS-B"]
        else:
            chain = ["BFGS"]

    return [c.upper() for c in chain]


def _record_from_result(
    res,
    elapsed: float,
    method_used: str,
    route_key: str,
    chain: list[str],
) -> SolveRecord:
    primary = chain[0] if chain else ""
    primary_hit = method_used == primary
    if method_used == "LEAST_SQUARES":
        # least_squares 的目标是 0.5*||r||^2；转回 f 近似为 2*cost
        f_final = float(2.0 * getattr(res, "cost", np.nan))
        nfev = int(getattr(res, "nfev", 0))
        njev = int(getattr(res, "njev", 0) or 0)
        nit = int(getattr(res, "nit", 0) or 0)
        status = int(getattr(res, "status", 0) or 0)
        msg = str(getattr(res, "message", ""))
        success = bool(getattr(res, "success", False))
    else:
        f_final = float(getattr(res, "fun", np.nan))
        nfev = int(getattr(res, "nfev", 0) or 0)
        njev = int(getattr(res, "njev", 0) or 0)
        nit = int(getattr(res, "nit", 0) or 0)
        status = int(getattr(res, "status", 0) or 0)
        msg = str(getattr(res, "message", ""))
        success = bool(getattr(res, "success", False))

    return SolveRecord(
        success=success,
        status=status,
        message=msg,
        nit=nit,
        nfev=nfev,
        njev=njev,
        f_final=f_final,
        is_f_final_finite=bool(np.isfinite(f_final)),
        time_sec=elapsed,
        solver_method=method_used,
        route_key=route_key,
        solver_chain="->".join(chain),
        primary_solver=primary,
        primary_hit=primary_hit,
        backup_triggered=not primary_hit,
    )


def solve(
    p,
    init: InitResult,
    has_bounds: bool,
    cfg: Config,
    constraint_tag: str = "U",
    objective_tag: str = "SmoothNLP",
) -> SolveRecord:
    route_key = f"{constraint_tag}:{objective_tag}"
    chain = _select_solver_chain(
        has_bounds=has_bounds,
        cfg=cfg,
        constraint_tag=constraint_tag,
        objective_tag=objective_tag,
    )

    t0 = time.perf_counter()
    timeout = int(cfg.solver.get("timeout_sec", 120))
    failures = []

    try:
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        except (AttributeError, OSError):
            pass

        for method_alias in chain:
            try:
                if method_alias == "LEAST_SQUARES":
                    res = _solve_least_squares(p, init.x0)
                else:
                    res = _solve_minimize(p, init.x0, method_alias, cfg)

                rec = _record_from_result(
                    res,
                    time.perf_counter() - t0,
                    method_alias,
                    route_key,
                    chain,
                )
                if rec.success and rec.is_f_final_finite:
                    return rec

                failures.append(f"{method_alias}: {rec.message}")
            except Exception as e:
                failures.append(f"{method_alias}: {type(e).__name__}: {e}")
                continue

        return SolveRecord(
            success=False,
            status=-3,
            message="; ".join(failures) if failures else "all solver candidates failed",
            nit=0,
            nfev=0,
            njev=0,
            f_final=float("nan"),
            is_f_final_finite=False,
            time_sec=time.perf_counter() - t0,
            exception_type="SolverChainFailed",
            solver_method="->".join(chain),
            route_key=route_key,
            solver_chain="->".join(chain),
            primary_solver=(chain[0] if chain else ""),
            primary_hit=False,
            backup_triggered=(len(chain) > 1),
        )
    except TimeoutError:
        return SolveRecord(
            success=False,
            status=-1,
            message="timeout",
            nit=0,
            nfev=0,
            njev=0,
            f_final=float("nan"),
            is_f_final_finite=False,
            time_sec=time.perf_counter() - t0,
            exception_type="TimeoutError",
            solver_method="->".join(chain),
            route_key=route_key,
            solver_chain="->".join(chain),
            primary_solver=(chain[0] if chain else ""),
            primary_hit=False,
            backup_triggered=(len(chain) > 1),
        )
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass


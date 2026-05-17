"""
职责：
  接收问题信息（x0, bl, bu），按五种来源生成初值：
    - cutest：直接返回 x0
    - random_raw：K 个随机初值（仅 clip + fallback）
    - random_post：random_raw 后接同一套候选后处理
    - llm_raw：LLM 输出仅做 clip + fallback
    - llm_post：LLM 输出做分层后处理
  统一 clip（有bounds时），统一失效回退机制。

关键接口：
  Initializer(p, has_bounds, cfg)
    .get_cutest() -> InitResult
    .get_random(K) -> List[InitResult]                  # random_raw（仅 clip + fallback）
    .get_random_post(K, constraint_tag) -> List[InitResult]
    .get_llm_raw(candidate) -> List[InitResult]
    .get_llm_post(candidate, constraint_tag) -> List[InitResult]
    .get_llm(candidate, constraint_tag) -> List[InitResult]  # backward compatible alias

数据类：
  InitResult(x0, f0, is_f0_finite, fallback, fallback_reason)
"""
import numpy as np
import scipy.optimize as spopt
from dataclasses import dataclass
from typing import List


@dataclass
class InitResult:
    x0: np.ndarray
    f0: float
    is_f0_finite: bool
    fallback: bool = False
    fallback_reason: str = ""


class Initializer:
    def __init__(self, p, has_bounds: bool, cfg):
        self.p = p
        self.has_bounds = bool(has_bounds)
        self.cfg = cfg
        self.n = int(p.n)
        self.m = int(getattr(p, "m", 0) or 0)

        bl = np.array(p.bl, dtype=float) if p.bl is not None else np.full(self.n, -np.inf)
        bu = np.array(p.bu, dtype=float) if p.bu is not None else np.full(self.n, np.inf)
        self.bl, self.bu = bl, bu
        self.has_lb = np.isfinite(bl)
        self.has_ub = np.isfinite(bu)
        self.has_both = self.has_lb & self.has_ub

        widths = (bu - bl)[self.has_both]
        if len(widths) > 0:
            self.S = float(np.clip(np.median(widths), 1e-3, 1e6))
        else:
            self.S = float(max(1.0, np.median(np.abs(p.x0)) + 1.0))

        self.x0_cutest = np.array(p.x0, dtype=float)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self.has_bounds:
            x = np.clip(
                x,
                np.where(self.has_lb, self.bl, x),
                np.where(self.has_ub, self.bu, x),
            )
        return x

    def _eval_f0(self, x: np.ndarray) -> tuple:
        try:
            f = float(self.p.obj(x))
            return f, bool(np.isfinite(f))
        except Exception:
            return float("nan"), False

    def _make_result(self, x: np.ndarray, allow_retry: bool = True) -> InitResult:
        x = self._clip(x)
        f, ok = self._eval_f0(x)
        if not ok and allow_retry:
            alpha_r = self.cfg.init["alpha_retry"]
            rng = np.random.default_rng(0)
            x_retry = self.x0_cutest + rng.uniform(-alpha_r * self.S, alpha_r * self.S, self.n)
            x_retry = self._clip(x_retry)
            f2, ok2 = self._eval_f0(x_retry)
            if ok2:
                return InitResult(x_retry, f2, True, fallback=False)
            f3, ok3 = self._eval_f0(self.x0_cutest)
            return InitResult(self.x0_cutest.copy(), f3, ok3, fallback=True, fallback_reason="f0_invalid_after_retry")

        if not ok:
            f3, ok3 = self._eval_f0(self.x0_cutest)
            return InitResult(self.x0_cutest.copy(), f3, ok3, fallback=True, fallback_reason="f0_invalid")

        return InitResult(x, f, True)

    def _solver_bounds(self):
        bounds = []
        for i in range(self.n):
            lb = float(self.bl[i]) if np.isfinite(self.bl[i]) else None
            ub = float(self.bu[i]) if np.isfinite(self.bu[i]) else None
            bounds.append((lb, ub))
        return bounds

    def _constraint_arrays(self):
        if self.m <= 0:
            return None
        cl = np.array(getattr(self.p, "cl", np.full(self.m, -np.inf)), dtype=float)
        cu = np.array(getattr(self.p, "cu", np.full(self.m, np.inf)), dtype=float)
        if cl.shape[0] != self.m or cu.shape[0] != self.m:
            return None

        is_eq_raw = getattr(self.p, "is_eq_cons", None)
        if is_eq_raw is None:
            is_eq = np.isclose(cl, cu, atol=1e-10, rtol=0.0)
        else:
            is_eq = np.asarray(is_eq_raw, dtype=bool)
            if is_eq.shape[0] != self.m:
                is_eq = np.isclose(cl, cu, atol=1e-10, rtol=0.0)

        is_linear = np.asarray(getattr(self.p, "is_linear_cons", np.ones(self.m, dtype=bool)), dtype=bool)
        if is_linear.shape[0] != self.m:
            is_linear = np.ones(self.m, dtype=bool)

        return cl, cu, is_eq, is_linear

    @staticmethod
    def _violation(c: np.ndarray, cl: np.ndarray, cu: np.ndarray) -> np.ndarray:
        v_low = np.maximum(cl - c, 0.0)
        v_high = np.maximum(c - cu, 0.0)
        return np.maximum(v_low, v_high)

    def _optimize_penalty(self, x_start: np.ndarray, fun, maxiter: int) -> np.ndarray:
        try:
            res = spopt.minimize(
                fun=fun,
                x0=x_start,
                method="L-BFGS-B",
                bounds=self._solver_bounds(),
                options={"maxiter": int(maxiter), "ftol": 1e-12},
            )
            x_new = np.asarray(res.x, dtype=float)
            if x_new.shape[0] == self.n and np.all(np.isfinite(x_new)):
                return self._clip(x_new)
        except Exception:
            pass
        return self._clip(x_start)

    def _postprocess_candidate(self, x_candidate: np.ndarray, constraint_tag: str) -> np.ndarray:
        # 4.1 box 投影（必做）
        x = self._clip(np.asarray(x_candidate, dtype=float))

        cons_meta = self._constraint_arrays()
        if cons_meta is None or not hasattr(self.p, "cons"):
            return x

        cl, cu, is_eq, _ = cons_meta

        # 4.2 线性等式投影（数值近似：最小化等式残差）
        if constraint_tag in {"LEB", "LCB"} and np.any(is_eq):
            eq_idx = np.where(is_eq)[0]
            b_eq = cl[eq_idx]

            def eq_residual(xx):
                c = np.asarray(self.p.cons(xx), dtype=float)
                r = c[eq_idx] - b_eq
                return 0.5 * float(np.dot(r, r))

            x = self._optimize_penalty(x, eq_residual, maxiter=40)

        # 4.3 LCB 不等式可行化（惩罚下降）
        if constraint_tag == "LCB":
            ineq_idx = np.where(~is_eq)[0]

            def ineq_penalty(xx):
                c = np.asarray(self.p.cons(xx), dtype=float)
                v = self._violation(c[ineq_idx], cl[ineq_idx], cu[ineq_idx]) if ineq_idx.size > 0 else np.zeros(1)
                return float(np.dot(v, v))

            x = self._optimize_penalty(x, ineq_penalty, maxiter=25)

        # 4.4 NLC 非线性可行化（惩罚下降）
        if constraint_tag == "NLC":

            def nlc_penalty(xx):
                c = np.asarray(self.p.cons(xx), dtype=float)
                v = self._violation(c, cl, cu)
                return float(np.dot(v, v))

            x = self._optimize_penalty(x, nlc_penalty, maxiter=30)

        return self._clip(x)

    def get_cutest(self) -> InitResult:
        f, ok = self._eval_f0(self.x0_cutest)
        return InitResult(self.x0_cutest.copy(), f, ok)

    def get_random(self, K: int) -> List[InitResult]:
        """Legacy random_raw: only random sampling + clip + f0 fallback."""
        results = []
        seed_base = self.cfg.init["random_seed_base"]
        alpha = self.cfg.init["alpha_default"]
        for k in range(K):
            rng = np.random.default_rng(seed_base + k)
            x = self.x0_cutest.copy()
            ib = self.has_both
            if ib.any():
                x[ib] = rng.uniform(self.bl[ib], self.bu[ib])
            iu = ~ib
            if iu.any():
                x[iu] = self.x0_cutest[iu] + rng.uniform(-alpha * self.S, alpha * self.S, iu.sum())
            results.append(self._make_result(x))
        return results

    def get_random_post(self, K: int, constraint_tag: str) -> List[InitResult]:
        """random_post: random samples followed by shared feasibility-oriented postprocess."""
        raw = self.get_random(K)
        out: List[InitResult] = []
        for r in raw:
            try:
                x_post = self._postprocess_candidate(r.x0, constraint_tag=constraint_tag)
                out.append(self._make_result(x_post))
            except Exception:
                out.append(r)
        return out

    def _extract_candidate_xs(self, candidate: dict) -> List[np.ndarray]:
        if not candidate:
            return []

        xs = candidate.get("xs")
        if xs is None:
            xs = candidate.get("x_list")
        if xs is None and "x" in candidate:
            xs = [candidate["x"]]

        if not isinstance(xs, list):
            return []

        out: List[np.ndarray] = []
        for x in xs:
            try:
                x_raw = np.asarray(x, dtype=float)
            except Exception:
                continue
            if x_raw.ndim != 1 or x_raw.shape[0] != self.n or not np.all(np.isfinite(x_raw)):
                continue
            out.append(x_raw)
        return out

    def get_llm_raw(self, candidate: dict) -> List[InitResult]:
        """LLM raw arm: clip + f0 evaluation/fallback only."""
        xs = self._extract_candidate_xs(candidate)
        return [self._make_result(x) for x in xs]

    def get_llm_post(self, candidate: dict, constraint_tag: str) -> List[InitResult]:
        """LLM post arm: shared feasibility-oriented postprocess per candidate."""
        xs = self._extract_candidate_xs(candidate)
        out: List[InitResult] = []
        for x_raw in xs:
            try:
                x_post = self._postprocess_candidate(x_raw, constraint_tag=constraint_tag)
                out.append(self._make_result(x_post))
            except Exception:
                continue
        return out

    def get_llm(self, candidate: dict, constraint_tag: str) -> List[InitResult]:
        # Backward compatible alias: historical callers treat get_llm as postprocessed arm.
        return self.get_llm_post(candidate, constraint_tag)


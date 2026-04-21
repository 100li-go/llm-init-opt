"""
职责：
  接收问题信息（x0, bl, bu, 类别），按三种来源生成初值：
    - cutest：直接返回 x0
    - random：K 个随机初值
    - llm：按策略 JSON 实例化 K 个初值
  统一 clip（有bounds时），统一失效回退机制。

关键接口：
  Initializer(p, category, cfg)
    .get_cutest() -> np.ndarray
    .get_random(K) -> List[InitResult]
    .get_llm(strategies) -> List[InitResult]

数据类：
  InitResult(x0, f0, is_f0_finite, fallback, fallback_reason)
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class InitResult:
    x0: np.ndarray
    f0: float
    is_f0_finite: bool
    fallback: bool = False
    fallback_reason: str = ""

class Initializer:
    def __init__(self, p, category: str, cfg):
        self.p = p
        self.category = category
        self.cfg = cfg
        n = p.n
        self.n = n

        # bounds（统一为 float array，无界用 ±inf）
        bl = np.array(p.bl, dtype=float) if p.bl is not None else np.full(n, -np.inf)
        bu = np.array(p.bu, dtype=float) if p.bu is not None else np.full(n, +np.inf)
        self.bl, self.bu = bl, bu
        self.has_lb = np.isfinite(bl)
        self.has_ub = np.isfinite(bu)
        self.has_both = self.has_lb & self.has_ub

        # 全局尺度 S
        widths = (bu - bl)[self.has_both]
        if len(widths) > 0:
            self.S = float(np.clip(np.median(widths), 1e-3, 1e6))
        else:
            self.S = float(max(1.0, np.median(np.abs(p.x0)) + 1.0))

        self.x0_cutest = np.array(p.x0, dtype=float)

    # ── 工具方法 ────────────────────────────────────────
    def _clip(self, x: np.ndarray) -> np.ndarray:
        """若有 bounds，做 clip；无 bounds 直接返回"""
        if self.category == "B":
            x = np.clip(x,
                        np.where(self.has_lb, self.bl, x),
                        np.where(self.has_ub, self.bu, x))
        return x

    def _eval_f0(self, x: np.ndarray) -> tuple:
        """安全计算 f(x0)"""
        try:
            f = float(self.p.obj(x))
            return f, np.isfinite(f)
        except Exception:
            return float("nan"), False

    def _make_result(self, x: np.ndarray, alpha: float,
                     allow_retry: bool = True) -> InitResult:
        """生成 InitResult，含失效回退逻辑"""
        x = self._clip(x)
        f, ok = self._eval_f0(x)
        if not ok and allow_retry:
            # 缩小扰动重试（回到 x0_cutest 附近）
            alpha_r = self.cfg.init["alpha_retry"]
            x_retry = self.x0_cutest.copy()
            rng = np.random.default_rng(0)
            x_retry += rng.uniform(-alpha_r * self.S, alpha_r * self.S, self.n)
            x_retry = self._clip(x_retry)
            f2, ok2 = self._eval_f0(x_retry)
            if ok2:
                return InitResult(x_retry, f2, True, fallback=False)
            else:
                # 最终回退到 x0_cutest
                f3, ok3 = self._eval_f0(self.x0_cutest)
                return InitResult(self.x0_cutest.copy(), f3, ok3,
                                  fallback=True, fallback_reason="f0_invalid_after_retry")
        if not ok:
            f3, ok3 = self._eval_f0(self.x0_cutest)
            return InitResult(self.x0_cutest.copy(), f3, ok3,
                              fallback=True, fallback_reason="f0_invalid")
        return InitResult(x, f, True)

    # ── 三种来源 ────────────────────────────────────────
    def get_cutest(self) -> InitResult:
        f, ok = self._eval_f0(self.x0_cutest)
        return InitResult(self.x0_cutest.copy(), f, ok)

    def get_random(self, K: int) -> List[InitResult]:
        results = []
        seed_base = self.cfg.init["random_seed_base"]
        alpha = self.cfg.init["alpha_default"]
        for k in range(K):
            rng = np.random.default_rng(seed_base + k)
            x = self.x0_cutest.copy()
            # 有界维度：均匀采样 [bl, bu]
            ib = self.has_both
            if ib.any():
                x[ib] = rng.uniform(self.bl[ib], self.bu[ib])
            # 无界维度：x0 + Uniform(-alpha*S, +alpha*S)
            iu = ~ib
            if iu.any():
                x[iu] = self.x0_cutest[iu] + rng.uniform(
                    -alpha * self.S, alpha * self.S, iu.sum())
            results.append(self._make_result(x, alpha))
        return results

    def get_llm(self, strategies: list) -> List[InitResult]:
        results = []
        for s in strategies:
            x = self._instantiate_strategy(s)
            results.append(self._make_result(x, s["alpha"]))
        return results

    def _instantiate_strategy(self, s: dict) -> np.ndarray:
        mode = s["mode"]
        alpha = s["alpha"]
        sparsity = s["sparsity"]
        seed = s["seed"]
        rng = np.random.default_rng(seed)
        x = self.x0_cutest.copy()
        n_perturb = max(1, math.ceil(self.n * sparsity))
        dims = rng.choice(self.n, size=n_perturb, replace=False)

        if mode == "near_x0":
            noise = rng.uniform(-alpha * self.S, alpha * self.S, n_perturb)
            x[dims] += noise

        elif mode == "center_of_bounds":
            for i in dims:
                if self.has_both[i]:
                    center = (self.bl[i] + self.bu[i]) / 2.0
                    noise = rng.uniform(-alpha * self.S, alpha * self.S)
                    x[i] = center + noise
                else:
                    x[i] += rng.uniform(-alpha * self.S, alpha * self.S)

        elif mode == "interior_from_bounds":
            for i in dims:
                if self.has_both[i]:
                    width = self.bu[i] - self.bl[i]
                    # 向内缩 10% width，再小扰动
                    side = rng.integers(0, 2)
                    if side == 0:
                        x[i] = self.bl[i] + 0.1 * width
                    else:
                        x[i] = self.bu[i] - 0.1 * width
                    x[i] += rng.uniform(-alpha * self.S * 0.1,
                                         alpha * self.S * 0.1)
                else:
                    x[i] += rng.uniform(-alpha * self.S, alpha * self.S)
        return x
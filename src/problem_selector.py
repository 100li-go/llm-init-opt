"""
职责：
  1. 枚举本地 CUTEst 已安装问题
  2. 按过滤条件（n<=200, m=0）筛选
  3. 划分 A/B 类
  4. 固定种子抽样 300 题
  5. 输出 problem_set.json

关键接口：
  build_problem_set(cfg) -> List[ProblemMeta]
  save_problem_set(problems, path)
  load_problem_set(path) -> List[ProblemMeta]
"""
import json
import random
import pycutest
from dataclasses import dataclass, asdict
from typing import List, Optional
from src.config import Config

@dataclass
class ProblemMeta:
    name: str
    n: int
    m: int
    category: str          # "A"=无bounds  "B"=box bounds
    has_bounds: bool
    x0_norm: float         # ||x0||，用于后续尺度估计

def _classify(p) -> Optional[str]:
    """返回 'A'/'B'，不符合条件返回 None"""
    if p.m != 0:
        return None
    bl, bu = p.bl, p.bu
    has_finite_bound = (
        bl is not None and bu is not None and
        any((bl[i] > -1e19 or bu[i] < 1e19) for i in range(p.n))
    )
    return "B" if has_finite_bound else "A"

def build_problem_set(cfg: Config) -> List[ProblemMeta]:
    all_names = pycutest.find_problems(constraints="unconstrained") + \
                pycutest.find_problems(constraints="bounds")
    all_names = list(set(all_names))

    metas = []
    for name in all_names:
        try:
            props = pycutest.problem_properties(name)
            n = props["n"]
            m = props.get("m", 0) or 0
            if n > cfg.problem_selection["n_max"] or m > cfg.problem_selection["m_max"]:
                continue
            p = pycutest.import_problem(name)
            cat = _classify(p)
            if cat is None:
                continue
            import numpy as np
            metas.append(ProblemMeta(
                name=name, n=p.n, m=p.m,
                category=cat,
                has_bounds=(cat == "B"),
                x0_norm=float(np.linalg.norm(p.x0))
            ))
            p.close()  # 释放资源
        except Exception:
            continue

    rng = random.Random(cfg.problem_selection["seed"])
    n_sample = min(cfg.problem_selection["n_sample"], len(metas))
    sampled = rng.sample(metas, n_sample)
    return sampled

def save_problem_set(problems: List[ProblemMeta], path: str):
    with open(path, "w") as f:
        json.dump([asdict(p) for p in problems], f, indent=2)

def load_problem_set(path: str) -> List[ProblemMeta]:
    with open(path) as f:
        data = json.load(f)
    return [ProblemMeta(**d) for d in data]
"""
职责：
  1. 枚举本地 CUTEst 已安装问题
  2. 按过滤条件筛选
  3. 生成细粒度标签（ConstraintTag/ObjectiveTag）
  4. 固定种子抽样
  5. 输出 problem_set.json

关键接口：
  build_problem_set(cfg) -> List[ProblemMeta]
  save_problem_set(problems, path)
  load_problem_set(path) -> List[ProblemMeta]
"""
import json
from dataclasses import dataclass, asdict
from typing import List

from src.config import Config
from src.problem_tags import classify_tags


@dataclass
class ProblemMeta:
    name: str
    n: int
    m: int
    has_bounds: bool
    x0_norm: float  #初始点范数
    constraint_tag: str
    objective_tag: str
    route_key: str


PROBLEM_SET_KEY_MAP = {
    "name": "问题名",
    "n": "变量维度",
    "m": "约束数量",
    "has_bounds": "有界",
    "x0_norm": "初始点范数",
    "constraint_tag": "约束标签",
    "objective_tag": "目标标签",
    "route_key": "路由键",
}


def _to_cn_record(problem: ProblemMeta) -> dict:
    raw = asdict(problem)
    return {PROBLEM_SET_KEY_MAP[k]: v for k, v in raw.items()}


def _pick(d: dict, en_key: str, default=None):
    cn_key = PROBLEM_SET_KEY_MAP[en_key]
    return d.get(cn_key, default)


def build_problem_set(cfg: Config) -> List[ProblemMeta]:
    import pycutest
    target_n = int(cfg.problem_selection["n_sample"])

    groups = ["unconstrained", "bounds", "linear", "quadratic", "other"]
    all_names: List[str] = []
    for g in groups:
        all_names.extend(pycutest.find_problems(constraints=g))
    all_names = list(set(all_names))
    print(f"候选问题总数: {len(all_names)}")

    metas: List[ProblemMeta] = []
    for i, name in enumerate(all_names):
        if i % 50 == 0:
            print(f"  筛选进度: {i}/{len(all_names)}, 已入选: {len(metas)}")
        try:
            props = pycutest.problem_properties(name)
            n = int(props.get("n", 0) or 0)
            m = int(props.get("m", 0) or 0)

            if n <= 0 or n > int(cfg.problem_selection["n_max"]):
                continue
            if m > int(cfg.problem_selection["m_max"]):
                continue

            p = pycutest.import_problem(name)
            tags = classify_tags(p, props)

            metas.append(
                ProblemMeta(
                    name=name,
                    n=n,
                    m=m,
                    has_bounds=bool(tags["has_box"]),
                    x0_norm=float(0.0),
                    constraint_tag=tags["ConstraintTag"],
                    objective_tag=tags["ObjectiveTag"],
                    route_key=tags["route_key"],
                )
            )
            if len(metas) >= target_n:
                print(f"达到目标样本数 {target_n}，停止继续筛选。")
                break
        except Exception:
            continue

    print(f"最终样本数: {len(metas)} 题")
    return metas


def save_problem_set(problems: List[ProblemMeta], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_to_cn_record(p) for p in problems], f, indent=2, ensure_ascii=False)


def load_problem_set(path: str) -> List[ProblemMeta]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    metas: List[ProblemMeta] = []
    for d in data:
        has_bounds = bool(_pick(d, "has_bounds", False))
        ctag = _pick(d, "constraint_tag") or ("B" if has_bounds else "U")
        otag = _pick(d, "objective_tag") or "SmoothNLP"
        # Backward compatibility: old snapshots may still use QP.
        if str(otag) == "QP":
            otag = "SmoothNLP"
        route_key = str(_pick(d, "route_key", "") or "")
        if not route_key or route_key.endswith(":QP"):
            route_key = f"{ctag}:{otag}"
        metas.append(
            ProblemMeta(
                name=str(_pick(d, "name", "")),
                n=int(_pick(d, "n", 0)),
                m=int(_pick(d, "m", 0)),
                has_bounds=has_bounds,
                x0_norm=float(_pick(d, "x0_norm", 0.0)),
                constraint_tag=ctag,
                objective_tag=otag,
                route_key=route_key,
            )
        )
    return metas

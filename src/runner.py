"""
职责：
  对 problem_set.json 中每道题：
    1. 加载 pycutest 问题
    2. 生成三组初值（cutest / random / llm）
    3. 调用 solver，收集所有 RunRecord
    4. 拼成 DataFrame 行，累积写入 runs.parquet
  支持断点续跑（已跑过的 problem 跳过）。

关键接口：
  run_all(problem_set, cfg, strategies_dir, out_path)
"""
import json
import pandas as pd
import pycutest
from pathlib import Path
from tqdm import tqdm

from src.config import Config
from src.problem_selector import ProblemMeta
from src.initializer import Initializer
from src.solver import solve
from src.strategy_validator import validate_and_fix

def _load_strategies(problem_name: str, strategies_dir: Path,
                     cfg: Config, has_bounds: bool) -> list:
    path = strategies_dir / f"{problem_name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    return validate_and_fix(raw, cfg, has_bounds)

def _build_row(meta: ProblemMeta, init_source: str, init_id: int,
               init_res, solve_rec) -> dict:
    return {
        # 元信息
        "problem": meta.name,
        "n": meta.n,
        "category": meta.category,
        "has_bounds": meta.has_bounds,
        # 初值
        "init_source": init_source,
        "init_id": init_id,
        "f0": init_res.f0,
        "x0_norm": float((init_res.x0**2).sum()**0.5),
        "is_f0_finite": init_res.is_f0_finite,
        "fallback": init_res.fallback,
        "fallback_reason": init_res.fallback_reason,
        # 求解结果
        "success": solve_rec.success,
        "status": solve_rec.status,
        "message": solve_rec.message,
        "nit": solve_rec.nit,
        "nfev": solve_rec.nfev,
        "njev": solve_rec.njev,
        "f_final": solve_rec.f_final,
        "is_f_final_finite": solve_rec.is_f_final_finite,
        "time_sec": solve_rec.time_sec,
        "exception_type": solve_rec.exception_type,
    }

def run_all(problem_set: list, cfg: Config,
            strategies_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # 断点续跑：读取已完成的 problem 名
    done_problems = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["problem"])
        done_problems = set(existing["problem"].unique())

    for meta in tqdm(problem_set, desc="Problems"):
        if meta.name in done_problems:
            continue
        try:
            p = pycutest.import_problem(meta.name)
        except Exception:
            continue

        init_obj = Initializer(p, meta.category, cfg)
        strategies = _load_strategies(meta.name, strategies_dir,
                                      cfg, meta.has_bounds)
        # ── CUTEst baseline ──────────────────────────
        ci = init_obj.get_cutest()
        sr = solve(p, ci, meta.category, cfg)
        rows.append(_build_row(meta, "cutest", 0, ci, sr))

        # ── Random baseline ──────────────────────────
        for kid, ri in enumerate(init_obj.get_random(cfg.K)):
            sr = solve(p, ri, meta.category, cfg)
            rows.append(_build_row(meta, "random", kid, ri, sr))

        # ── LLM ──────────────────────────────────────
        if strategies:
            for kid, li in enumerate(init_obj.get_llm(strategies)):
                sr = solve(p, li, meta.category, cfg)
                rows.append(_build_row(meta, "llm", kid, li, sr))

        p.close()

    # 追加写入
    df_new = pd.DataFrame(rows)
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_parquet(out_path, index=False)
    print(f"Saved {len(df_all)} rows to {out_path}")
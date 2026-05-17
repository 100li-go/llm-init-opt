"""批量求解，写入 runs.parquet，每题用子进程强制超时"""
import json
import multiprocessing as mp

import pandas as pd
import pycutest
from tqdm import tqdm

from src.initializer import Initializer
from src.llm_output_validator import validate_candidate, validate_candidate_multi
from src.solver import solve

PROBLEM_TIMEOUT = 60


def _load_llm_candidate(problem_name, candidates_dir):
    path = candidates_dir / f"{problem_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else None


def _build_row(meta, init_source, init_id, init_res, solve_rec):
    return {
        "problem": meta.name,
        "n": meta.n,
        "has_bounds": meta.has_bounds,
        "constraint_tag": meta.constraint_tag,
        "objective_tag": meta.objective_tag,
        "route_key": meta.route_key,
        "init_source": init_source,
        "init_id": init_id,
        "f0": init_res.f0,
        "x0_norm": float((init_res.x0**2).sum() ** 0.5),
        "is_f0_finite": init_res.is_f0_finite,
        "fallback": init_res.fallback,
        "fallback_reason": init_res.fallback_reason,
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
        "solver_method": solve_rec.solver_method,
        "solver_chain": solve_rec.solver_chain,
        "primary_solver": solve_rec.primary_solver,
        "primary_hit": solve_rec.primary_hit,
        "backup_triggered": solve_rec.backup_triggered,
    }


def _worker(meta, cfg, candidates_dir, queue):
    try:
        p = pycutest.import_problem(meta.name)
        init_obj = Initializer(p, meta.has_bounds, cfg)
        candidate_raw = _load_llm_candidate(meta.name, candidates_dir)

        rows = []
        ci = init_obj.get_cutest()
        rows.append(
            _build_row(
                meta,
                "cutest",
                0,
                ci,
                solve(p, ci, meta.has_bounds, cfg, meta.constraint_tag, meta.objective_tag),
            )
        )

        for kid, ri in enumerate(init_obj.get_random(cfg.K)):
            rows.append(
                _build_row(
                    meta,
                    "random_raw",
                    kid,
                    ri,
                    solve(p, ri, meta.has_bounds, cfg, meta.constraint_tag, meta.objective_tag),
                )
            )

        for kid, ri in enumerate(init_obj.get_random_post(cfg.K, meta.constraint_tag)):
            rows.append(
                _build_row(
                    meta,
                    "random_post",
                    kid,
                    ri,
                    solve(p, ri, meta.has_bounds, cfg, meta.constraint_tag, meta.objective_tag),
                )
            )

        multi_cfg = cfg.llm.get("multi_output", {}) if isinstance(cfg.llm, dict) else {}
        multi_enabled = bool(multi_cfg.get("enabled", False))
        multi_k = int(multi_cfg.get("k", 5))

        if candidate_raw is not None:
            try:
                if multi_enabled:
                    candidate = validate_candidate_multi(candidate_raw, p.n, k=multi_k)
                else:
                    candidate = validate_candidate(candidate_raw, p.n)
            except Exception:
                candidate = None
            if candidate is not None:
                for kid, li in enumerate(init_obj.get_llm_raw(candidate)):
                    rows.append(
                        _build_row(
                            meta,
                            "llm_raw",
                            kid,
                            li,
                            solve(p, li, meta.has_bounds, cfg, meta.constraint_tag, meta.objective_tag),
                        )
                    )

                for kid, li in enumerate(init_obj.get_llm_post(candidate, meta.constraint_tag)):
                    rows.append(
                        _build_row(
                            meta,
                            "llm_post",
                            kid,
                            li,
                            solve(p, li, meta.has_bounds, cfg, meta.constraint_tag, meta.objective_tag),
                        )
                    )

        queue.put(rows)
    except Exception:
        queue.put([])


def run_all(problem_set, cfg, candidates_dir, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_problems = set()
    if out_path.exists():
        done_problems = set(pd.read_parquet(out_path, columns=["problem"])["problem"].unique())

    all_rows = []
    for meta in tqdm(problem_set, desc="Problems"):
        if meta.name in done_problems:
            continue
        queue = mp.Queue()
        proc = mp.Process(target=_worker, args=(meta, cfg, candidates_dir, queue))
        proc.start()
        proc.join(PROBLEM_TIMEOUT)
        if proc.is_alive():
            print(f"\n[SKIP] {meta.name}: timeout after {PROBLEM_TIMEOUT}s")
            proc.kill()
            proc.join()
            continue
        rows = queue.get() if not queue.empty() else []
        all_rows.extend(rows)

    df_new = pd.DataFrame(all_rows)
    if out_path.exists():
        df_all = pd.concat([pd.read_parquet(out_path), df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_parquet(out_path, index=False)
    print(f"Saved {len(df_all)} rows to {out_path}")


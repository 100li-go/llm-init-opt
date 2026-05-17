"""Audit route->solver mapping, 5-arm coverage, and comparison metrics for selected problem groups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.config import CFG
from src.problem_selector import load_problem_set


TARGET_CONSTRAINTS = {"NLC", "LEB", "LCB", "U", "B"}
TARGET_OBJECTIVES = {"LS", "SmoothNLP"}

EXPECTED_CHAIN = {
    "NLC:LS": ["IPOPT", "SQP"],
    "NLC:SmoothNLP": ["IPOPT", "SQP"],
    "LEB:LS": ["SQP", "IPOPT"],
    "LEB:SmoothNLP": ["SQP", "IPOPT"],
    "LCB:LS": ["IPOPT", "SQP"],
    "LCB:SmoothNLP": ["IPOPT", "SQP"],
    "U:LS": ["LEAST_SQUARES"],
    "B:LS": ["LEAST_SQUARES"],
    "U:SmoothNLP": ["L-BFGS-B"],
    "B:SmoothNLP": ["L-BFGS-B"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit selected routes for solver policy and arm coverage")
    p.add_argument("--runs", default="results/runs.parquet", help="runs parquet path")
    p.add_argument("--problem-set", default=None, help="problem_set path; default from config")
    p.add_argument("--candidates-dir", default=None, help="llm_candidates dir; default from config")
    return p.parse_args()


def _load_selected(problem_set_path: Path, candidates_dir: Path):
    metas = load_problem_set(str(problem_set_path))
    cand_names = {p.stem for p in candidates_dir.glob("*.json")}

    selected = []
    for m in metas:
        if m.name not in cand_names:
            continue
        if m.constraint_tag not in TARGET_CONSTRAINTS:
            continue
        if m.objective_tag not in TARGET_OBJECTIVES:
            continue
        selected.append(m)
    return selected


def _check_config_mapping() -> list[str]:
    issues = []
    route_exact = CFG.solver.get("policy", {}).get("route_exact", {})
    for route, expected in EXPECTED_CHAIN.items():
        got = route_exact.get(route)
        if got is None:
            issues.append(f"missing route_exact for {route}")
            continue
        got_u = [str(x).upper() for x in got]
        if got_u != expected:
            issues.append(f"{route}: expected {expected}, got {got_u}")
    return issues


def _metric_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("init_source"):
        succ = g[g["success"] & g["is_f_final_finite"]]
        rows.append(
            {
                "init_source": src,
                "n_runs": int(len(g)),
                "success_rate": float(len(succ) / len(g)) if len(g) else 0.0,
                "median_nfev": float(succ["nfev"].median()) if len(succ) else float("nan"),
                "median_time": float(succ["time_sec"].median()) if len(succ) else float("nan"),
                "best_f_final": float(succ["f_final"].min()) if len(succ) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("init_source")


def audit_runs(runs_path: Path, selected_names: set[str], expected_per_problem: dict[str, list[str]]) -> dict:
    df = pd.read_parquet(runs_path)
    df = df[df["problem"].isin(selected_names)].copy()

    arm_missing = []
    arm_count_violations = []
    mapping_violations = []

    expected_sources = {"cutest", "random_raw", "random_post", "llm_raw", "llm_post"}
    k_expected = int(CFG.K)
    llm_k_expected = int(CFG.llm.get("multi_output", {}).get("k", 1))

    for problem, g in df.groupby("problem"):
        got_sources = set(g["init_source"].astype(str))
        miss = sorted(expected_sources - got_sources)
        if miss:
            arm_missing.append((problem, miss))

        counts = Counter(g["init_source"].astype(str))
        if (
            counts.get("cutest", 0) != 1
            or counts.get("llm_raw", 0) != llm_k_expected
            or counts.get("llm_post", 0) != llm_k_expected
            or counts.get("random_raw", 0) != k_expected
            or counts.get("random_post", 0) != k_expected
        ):
            arm_count_violations.append((problem, dict(counts)))

        expected = expected_per_problem.get(problem)
        if not expected:
            continue
        exp_primary = expected[0]
        exp_chain = "->".join(expected)

        for _, row in g.iterrows():
            primary = str(row.get("primary_solver", "")).upper()
            chain = str(row.get("solver_chain", "")).upper()
            if primary != exp_primary or chain != exp_chain:
                mapping_violations.append(
                    {
                        "problem": problem,
                        "init_source": row.get("init_source"),
                        "expected_primary": exp_primary,
                        "got_primary": primary,
                        "expected_chain": exp_chain,
                        "got_chain": chain,
                    }
                )
                break

    return {
        "rows": int(len(df)),
        "problems_with_runs": int(df["problem"].nunique()),
        "arm_missing": arm_missing,
        "arm_count_violations": arm_count_violations,
        "mapping_violations": mapping_violations,
        "metric_table": _metric_table(df) if len(df) else pd.DataFrame(),
    }


def main() -> None:
    args = parse_args()

    problem_set_path = Path(args.problem_set or CFG.paths["problem_set"])
    candidates_dir = Path(args.candidates_dir or CFG.llm_candidates_dir)
    runs_path = Path(args.runs)

    selected = _load_selected(problem_set_path, candidates_dir)
    selected_names = {m.name for m in selected}
    combo_counts = Counter((m.constraint_tag, m.objective_tag) for m in selected)

    print(f"selected_problems={len(selected)}")
    print("combo_counts=")
    for key, val in sorted(combo_counts.items()):
        print(f"  {key}: {val}")

    cfg_issues = _check_config_mapping()
    if cfg_issues:
        print("config_mapping_check=FAILED")
        for i in cfg_issues:
            print(f"  - {i}")
    else:
        print("config_mapping_check=OK")

    expected_per_problem = {}
    for m in selected:
        rk = f"{m.constraint_tag}:{m.objective_tag}"
        expected_per_problem[m.name] = EXPECTED_CHAIN[rk]

    if not runs_path.exists():
        print(f"runs_file_missing={runs_path}")
        print("Cannot verify runtime 5-arm coverage/mapping from runs.parquet yet.")
        return

    report = audit_runs(runs_path, selected_names, expected_per_problem)
    print(f"runs_rows_filtered={report['rows']}")
    print(f"problems_with_runs={report['problems_with_runs']}")
    print(f"arm_missing_count={len(report['arm_missing'])}")
    print(f"arm_count_violation_count={len(report['arm_count_violations'])}")
    print(f"solver_mapping_violation_count={len(report['mapping_violations'])}")

    if report["arm_missing"]:
        print("arm_missing_examples=")
        for p, miss in report["arm_missing"][:10]:
            print(f"  {p}: missing {miss}")

    if report["arm_count_violations"]:
        print("arm_count_violation_examples=")
        for p, c in report["arm_count_violations"][:10]:
            print(f"  {p}: {c}")

    if report["mapping_violations"]:
        print("solver_mapping_violation_examples=")
        for r in report["mapping_violations"][:10]:
            print(
                f"  {r['problem']} ({r['init_source']}): expected {r['expected_chain']}, got {r['got_chain']}"
            )

    mt = report["metric_table"]
    if len(mt):
        print("metric_table=")
        print(mt.to_string(index=False))

    print("key_metrics=success_rate, median_nfev, median_time, best_f_final")


if __name__ == "__main__":
    main()


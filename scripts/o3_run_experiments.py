"""Step 3: 本地批量求解，写入 runs.parquet"""
from src.config import CFG
from src.problem_selector import load_problem_set
from src.runner import run_all
from pathlib import Path

if __name__ == "__main__":
    problems = load_problem_set(CFG.paths["problem_set"])
    out_path = CFG.results_dir / "runs.parquet"
    run_all(problems, CFG, CFG.strategies_dir, out_path)
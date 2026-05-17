"""Step 3: 本地批量求解，写入 runs.parquet"""
from src.config import CFG
from src.problem_selector import load_problem_set
from src.runner import run_all
from pathlib import Path

BLACKLIST = {"ODESSA"}


def _resolve_mode_paths(cfg):
    multi_cfg = cfg.llm.get("multi_output", {}) if isinstance(cfg.llm, dict) else {}
    enabled = bool(multi_cfg.get("enabled", False))
    suffix = str(multi_cfg.get("output_suffix", "_llmK5"))

    results_dir = cfg.results_dir
    candidates_dir = cfg.llm_candidates_dir
    if enabled:
        results_dir = Path(str(results_dir).rstrip("/\\") + suffix)
        candidates_dir = Path(str(candidates_dir).rstrip("/\\") + suffix)
    return enabled, suffix, results_dir, candidates_dir

if __name__ == "__main__":
    problems = [p for p in load_problem_set(CFG.paths["problem_set"]) if p.name not in BLACKLIST]
    enabled, suffix, results_dir, candidates_dir = _resolve_mode_paths(CFG)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "runs.parquet"
    print(f"[INFO] multi_output_enabled={enabled}, suffix={suffix}")
    print(f"[INFO] candidates_dir={candidates_dir}, out_path={out_path}")
    run_all(problems, CFG, candidates_dir, out_path)

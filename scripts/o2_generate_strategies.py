"""Step 2: 对每道题调用 DeepSeek，生成策略文件（可并行、可断点续跑）"""
import json
import pycutest
from pathlib import Path
from tqdm import tqdm
from src.config import CFG
from src.problem_selector import load_problem_set
from src.meta_extractor import extract_meta
from src.llm_client import generate_strategies

if __name__ == "__main__":
    problems = load_problem_set(CFG.paths["problem_set"])
    out_dir = CFG.strategies_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for meta in tqdm(problems, desc="LLM策略生成"):
        out_file = out_dir / f"{meta.name}.json"
        if out_file.exists():
            continue  # 断点续跑
        try:
            p = pycutest.import_problem(meta.name)
            m = extract_meta(p, meta.category)
            p.close()
            strategies = generate_strategies(m, CFG)
            with open(out_file, "w") as f:
                json.dump(strategies, f, indent=2)
        except Exception as e:
            print(f"[WARN] {meta.name}: {e}")
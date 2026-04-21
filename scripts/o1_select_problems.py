"""Step 1: 筛选并抽样问题集，输出 problem_set.json"""
from src.config import CFG
from src.problem_selector import build_problem_set, save_problem_set

if __name__ == "__main__":
    problems = build_problem_set(CFG)
    save_problem_set(problems, CFG.paths["problem_set"])
    print(f"Saved {len(problems)} problems to {CFG.paths['problem_set']}")
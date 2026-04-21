"""冒烟测试：用5道题验证完整链路"""
import json
from src.config import CFG
from src.problem_selector import load_problem_set
from src.meta_extractor import extract_meta
from src.llm_client import generate_strategies
from src.strategy_validator import validate_and_fix
from src.initializer import Initializer
from src.solver import solve
import pycutest

problems = load_problem_set(CFG.paths["problem_set"])[:5]

for meta in problems:
    print(f"\n{'='*50}")
    print(f"Problem: {meta.name}, n={meta.n}, cat={meta.category}")

    p = pycutest.import_problem(meta.name)

    # 1. 元信息
    m = extract_meta(p, meta.category)
    print(f"  Scale S={m['scale_S']:.3f}")

    # 2. LLM策略
    raw = generate_strategies(m, CFG)
    strategies = validate_and_fix(raw, CFG, meta.has_bounds)
    print(f"  Got {len(strategies)} valid strategies")

    # 3. 初值
    init_obj = Initializer(p, meta.category, CFG)
    cutest_init = init_obj.get_cutest()
    random_inits = init_obj.get_random(3)
    llm_inits = init_obj.get_llm(strategies[:3])
    print(f"  f0_cutest={cutest_init.f0:.4g}, finite={cutest_init.is_f0_finite}")
    print(f"  random fallbacks: {sum(r.fallback for r in random_inits)}/3")
    print(f"  llm fallbacks: {sum(r.fallback for r in llm_inits)}/3")

    # 4. 求解
    sr = solve(p, cutest_init, meta.category, CFG)
    print(f"  Solve: success={sr.success}, nfev={sr.nfev}, f={sr.f_final:.4g}")

    p.close()

print("\n✅ Smoke test passed!")

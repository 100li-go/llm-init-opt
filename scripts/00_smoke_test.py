"""冒烟测试：用5道题验证完整链路"""
import os
import multiprocessing as mp

import pycutest

from src.config import CFG
from src.initializer import Initializer
from src.llm_client import generate_candidate
from src.llm_output_validator import validate_candidate
from src.payload_builder import build_payload
from src.problem_selector import load_problem_set
from src.prompt_router import select_prompt
from src.solver import solve


def _worker(meta, has_api_key: bool, queue: mp.Queue):
    try:
        p = pycutest.import_problem(meta.name)

        props = pycutest.problem_properties(meta.name)
        payload = build_payload(p, props=props)
        prompt_spec = select_prompt(payload, CFG)

        candidate = None
        if has_api_key:
            raw = generate_candidate(payload, CFG, prompt_spec=prompt_spec)
            candidate = validate_candidate(raw, p.n)

        init_obj = Initializer(p, meta.has_bounds, CFG)
        cutest_init = init_obj.get_cutest()
        random_inits = init_obj.get_random(3)
        llm_inits = init_obj.get_llm(candidate, meta.constraint_tag) if candidate else []

        sr = solve(p, cutest_init, meta.has_bounds, CFG, meta.constraint_tag, meta.objective_tag)
        queue.put(
            {
                "ok": True,
                "route": prompt_spec.route_key,
                "has_llm_candidate": candidate is not None,
                "f0_cutest": cutest_init.f0,
                "f0_finite": cutest_init.is_f0_finite,
                "random_fallbacks": int(sum(r.fallback for r in random_inits)),
                "llm_fallbacks": int(sum(r.fallback for r in llm_inits)),
                "llm_count": len(llm_inits),
                "solve_success": bool(sr.success),
                "solve_nfev": sr.nfev,
                "solve_f": sr.f_final,
            }
        )
    except Exception as e:
        queue.put({"ok": False, "error": str(e)})


if __name__ == "__main__":
    problems = load_problem_set(CFG.paths["problem_set"])[:5]
    has_api_key = bool(CFG.llm.get("api_key") or os.environ.get("OPENAI_API_KEY"))
    max_attempts = 2

    if not has_api_key:
        print("[WARN] OPENAI_API_KEY 未设置：本次 smoke 将跳过 LLM 调用，仅验证 CUTEst/random/solver 链路。")

    failed = 0
    for meta in problems:
        print(f"\n{'=' * 50}")
        print(f"Problem: {meta.name}, n={meta.n}, has_bounds={meta.has_bounds}")

        result = None
        attempt_success = False
        for attempt in range(1, max_attempts + 1):
            queue = mp.Queue()
            proc = mp.Process(target=_worker, args=(meta, has_api_key, queue))
            proc.start()
            proc.join(120)

            if proc.is_alive():
                proc.kill()
                proc.join()
                print(f"  [WARN] timeout after 120s (attempt {attempt}/{max_attempts})")
                continue

            if proc.exitcode != 0:
                print(f"  [WARN] worker crashed (exitcode={proc.exitcode}, attempt {attempt}/{max_attempts})")
                continue

            result = queue.get() if not queue.empty() else {"ok": False, "error": "empty worker result"}
            if not result.get("ok", False):
                print(f"  [WARN] {result.get('error', 'unknown error')} (attempt {attempt}/{max_attempts})")
                continue

            attempt_success = True
            break

        if not attempt_success:
            failed += 1
            print("  [FAIL] worker failed after retries")
            continue

        print(f"  Route={result['route']}, has_bounds={meta.has_bounds}")
        print(f"  LLM candidate available: {result['has_llm_candidate']}")
        print(f"  f0_cutest={result['f0_cutest']:.4g}, finite={result['f0_finite']}")
        print(f"  random fallbacks: {result['random_fallbacks']}/3")
        print(f"  llm fallbacks: {result['llm_fallbacks']}/{result['llm_count']}")
        print(f"  Solve: success={result['solve_success']}, nfev={result['solve_nfev']}, f={result['solve_f']:.4g}")

    n_total = len(problems)
    n_ok = n_total - failed
    if n_ok == 0:
        raise SystemExit("\n❌ Smoke test failed: no problem completed successfully.")
    if failed > 0:
        print(f"\n⚠️ Smoke finished with partial failures: {failed}/{n_total} failed, {n_ok}/{n_total} succeeded.")
    else:
        print("\n✅ Smoke test passed!")

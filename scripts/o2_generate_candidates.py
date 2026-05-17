"""Step 2: 可分两阶段执行

1) 先构建并落盘 prompt 文档（payload + system/user prompt）
2) 再基于已落盘 prompt 文档调用 LLM 生成候选初值
"""
import argparse
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from tqdm import tqdm

from src.config import CFG
from src.llm_client import generate_candidate
from src.llm_output_validator import validate_candidate, validate_candidate_multi
from src.payload_builder import build_payload
from src.problem_selector import load_problem_set
from src.prompt_router import PromptSpec, build_prompt_payload, select_prompt


def _ensure_pycutest_cache_env() -> str:
    cache_dir = Path(__file__).resolve().parents[1] / "pycutest_cache_holder"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYCUTEST_CACHE", str(cache_dir))
    return os.environ["PYCUTEST_CACHE"]


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate one LLM candidate x per CUTEst problem")
    parser.add_argument("--limit", type=int, default=0, help="only process first N problems (0 means all)")
    parser.add_argument("--offset", type=int, default=0, help="skip first N problems before applying --limit")
    parser.add_argument("--timeout-sec", type=int, default=None, help="per-request LLM timeout seconds")
    parser.add_argument("--build-timeout-sec", type=int, default=None, help="payload build timeout seconds")
    parser.add_argument("--max-attempts", type=int, default=None, help="per-problem retry attempts")
    parser.add_argument("--llm-retries", type=int, default=1, help="retries inside one LLM attempt")
    parser.add_argument("--skip-crashy", action="store_true", help="skip known native-crash CUTEst problems")
    parser.add_argument("--prepare-prompts-only", action="store_true", help="only build/save prompt docs, do not call LLM")
    parser.add_argument("--use-prepared-prompts", action="store_true", help="load prompt docs from disk and call LLM")
    parser.add_argument("--prompts-dir", type=str, default=None, help="override prompt docs directory path")
    parser.add_argument("--overwrite", action="store_true", help="regenerate candidates even if output file exists")
    parser.add_argument("--llm-k5", action="store_true", help="enable multi-output mode (K candidates per problem)")
    parser.add_argument("--llm-k", type=int, default=5, help="K for multi-output mode")
    parser.add_argument("--mode-suffix", type=str, default="_llmK5", help="directory suffix for multi-output artifacts")
    return parser.parse_args()


def _resolve_mode_dirs(args, cfg):
    multi_cfg = cfg.llm.get("multi_output", {}) if isinstance(cfg.llm, dict) else {}
    enabled = bool(args.llm_k5 or multi_cfg.get("enabled", False))
    k = int(args.llm_k or multi_cfg.get("k", 5))
    suffix = str(multi_cfg.get("output_suffix", args.mode_suffix if enabled else ""))

    out_dir = cfg.llm_candidates_dir
    prompt_dir = Path(args.prompts_dir) if args.prompts_dir else cfg.llm_prompts_dir
    if enabled:
        out_dir = Path(str(out_dir).rstrip("/\\") + suffix)
        if args.prompts_dir:
            prompt_dir = Path(args.prompts_dir)
        else:
            prompt_dir = Path(str(prompt_dir).rstrip("/\\") + suffix)

    return enabled, k, suffix, out_dir, prompt_dir


def _payload_worker(problem_name: str, queue: mp.Queue):
    try:
        import pycutest

        p = pycutest.import_problem(problem_name)
        props = pycutest.problem_properties(problem_name)
        payload = _slim_payload(build_payload(p, props=props))
        queue.put({"ok": True, "payload": payload, "n_vars": int(p.n)})
    except Exception as e:
        queue.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _truncate_to_4_decimals_toward_zero(fv: float) -> float:
    s = str(fv)
    # Keep scientific notation as-is to avoid collapsing tiny/huge magnitudes.
    if "e" in s.lower():
        return fv
    try:
        d = Decimal(s)
        return float(d.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))
    except (InvalidOperation, ValueError):
        return fv


def _safe_num(v, clip: float = 1e8):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
            if fv != fv:
                return None
            if fv == float("inf"):
                return clip
            if fv == -float("inf"):
                return -clip
            if fv > clip:
                return clip
            if fv < -clip:
                return -clip
            # Limit decimal places in prompt payload: keep <=4 as-is, truncate >4 to 4.
            return _truncate_to_4_decimals_toward_zero(fv)
        except Exception:
            return None
    return v


def _sanitize_obj(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_obj(v) for v in obj]
    return _safe_num(obj)


def _slim_payload(payload: dict) -> dict:
    out = _sanitize_obj(payload)
    # Force key integer fields to remain integers in prompt payload.
    for k in ("n", "m", "n_violated0", "head_k", "n_total"):
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except Exception:
                pass

    constraints = out.get("constraints")
    if isinstance(constraints, dict):
        topk = constraints.get("topk")
        if isinstance(topk, list):
            for item in topk:
                if not isinstance(item, dict):
                    continue
                if "i" in item and item["i"] is not None:
                    try:
                        item["i"] = int(item["i"])
                    except Exception:
                        pass
                sparse = item.get("jacobian_row_sparse")
                if isinstance(sparse, list):
                    for sv in sparse:
                        if isinstance(sv, dict) and "j" in sv and sv["j"] is not None:
                            try:
                                sv["j"] = int(sv["j"])
                            except Exception:
                                pass

    return out


def _save_prompt_doc(prompt_path: Path, payload: dict, spec: PromptSpec, n_vars: int, mode: str = "legacy"):
    doc = {
        "mode": mode,
        "name": payload.get("name") or prompt_path.stem,
        "route_key": spec.route_key,
        "payload": payload,
        "system_prompt": spec.system_prompt,
        "user_prompt": spec.user_prompt,
    }
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def _load_prompt_doc(prompt_path: Path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    payload = doc.get("payload") or doc.get("问题载荷", {})
    n_vars = int(payload.get("n", doc.get("n", doc.get("变量维度", payload.get("变量维度", 0)))))
    spec = PromptSpec(
        route_key=str(doc.get("route_key") or doc.get("路由键") or payload.get("route_key") or payload.get("路由键") or "UNKNOWN"),
        system_prompt=str(doc.get("system_prompt", doc.get("系统提示词", ""))),
        user_prompt=str(doc.get("user_prompt", doc.get("用户提示词", ""))),
    )
    return payload, spec, n_vars


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = _parse_args()
    if args.prepare_prompts_only and args.use_prepared_prompts:
        raise ValueError("--prepare-prompts-only and --use-prepared-prompts cannot be used together")

    multi_enabled, multi_k, mode_suffix, out_dir, prompt_dir = _resolve_mode_dirs(args, CFG)

    problems = load_problem_set(CFG.paths["problem_set"])
    if args.offset and args.offset > 0:
        problems = problems[args.offset :]
    if args.limit and args.limit > 0:
        problems = problems[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    timeout_sec = int(args.timeout_sec or CFG.llm.get("request_timeout_sec", 60))
    build_timeout_sec = int(args.build_timeout_sec or CFG.llm.get("build_timeout_sec", 600))
    max_attempts = int(args.max_attempts or CFG.llm.get("problem_retry", 2))
    llm_retries = int(max(1, args.llm_retries))
    cache_path = _ensure_pycutest_cache_env()
    known_crashy = set(CFG.llm.get("cutest_crash_blacklist", []))
    failed = 0
    generated = 0
    prepared = 0
    skipped = 0

    print(
        f"[INFO] total_problems={len(problems)}, request_timeout_sec={timeout_sec}, "
        f"build_timeout_sec={build_timeout_sec}, max_attempts={max_attempts}, llm_retries={llm_retries}"
    )
    print(f"[INFO] PYCUTEST_CACHE={cache_path}")
    print(f"[INFO] prompt_dir={prompt_dir}, candidates_dir={out_dir}")
    print(f"[INFO] multi_output_enabled={multi_enabled}, multi_k={multi_k}, mode_suffix={mode_suffix}")
    if args.skip_crashy and known_crashy:
        print(f"[INFO] skip_crashy enabled, blacklist_size={len(known_crashy)}")

    for meta in tqdm(problems, desc="LLM候选初值生成"):
        t0 = time.perf_counter()
        out_file = out_dir / f"{meta.name}.json"
        prompt_file = prompt_dir / f"{meta.name}.json"
        if args.skip_crashy and meta.name in known_crashy:
            skipped += 1
            print(f"[SKIP] {meta.name}: in CUTEst crash blacklist")
            continue
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue

        payload = None
        n_vars = None
        prompt_spec = None

        if args.use_prepared_prompts:
            if not prompt_file.exists():
                failed += 1
                print(f"[WARN] {meta.name}: prepared prompt missing: {prompt_file}")
                print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")
                continue
            try:
                payload, prompt_spec, n_vars = _load_prompt_doc(prompt_file)
            except Exception as e:
                failed += 1
                print(f"[WARN] {meta.name}: failed loading prompt doc: {type(e).__name__}: {e}")
                print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")
                continue
        else:
            queue = mp.Queue()
            p_build = mp.Process(target=_payload_worker, args=(meta.name, queue))
            p_build.start()
            p_build.join(build_timeout_sec)

            if p_build.is_alive():
                p_build.kill()
                p_build.join()
                failed += 1
                print(f"[WARN] {meta.name}: payload build timeout ({build_timeout_sec}s)")
                print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")
                continue

            if p_build.exitcode != 0:
                failed += 1
                print(f"[WARN] {meta.name}: payload worker crashed exitcode={p_build.exitcode}")
                print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")
                continue

            result = queue.get() if not queue.empty() else {"ok": False, "error": "empty payload worker result"}
            if not result.get("ok", False):
                failed += 1
                print(f"[WARN] {meta.name}: payload build failed: {result.get('error', 'unknown error')}")
                print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")
                continue

            payload = result["payload"]
            n_vars = int(result["n_vars"])
            prompt_payload = build_prompt_payload(payload)
            prompt_spec = select_prompt(
                payload,
                CFG,
                multi_output=multi_enabled,
                multi_k=multi_k,
                prompt_payload=prompt_payload,
            )

            try:
                _save_prompt_doc(
                    prompt_file,
                    prompt_payload,
                    prompt_spec,
                    n_vars,
                    mode=("llm_k5" if multi_enabled else "legacy"),
                )
                prepared += 1
            except Exception as e:
                failed += 1
                print(f"[WARN] {meta.name}: failed saving prompt doc: {type(e).__name__}: {e}")
                print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")
                continue

        if args.prepare_prompts_only:
            print(f"[OK] {meta.name}: prompt prepared in {time.perf_counter() - t0:.1f}s")
            continue

        done = False
        for attempt in range(1, max_attempts + 1):
            try:
                raw_candidate = generate_candidate(
                    payload,
                    CFG,
                    prompt_spec=prompt_spec,
                    request_timeout_sec=timeout_sec,
                    max_retries_override=llm_retries,
                    multi_output=multi_enabled,
                    multi_k=multi_k,
                )
                if multi_enabled:
                    candidate = validate_candidate_multi(raw_candidate, n_vars, k=multi_k)
                else:
                    candidate = validate_candidate(raw_candidate, n_vars)
                with open(out_file, "w") as f:
                    json.dump(candidate, f, indent=2, ensure_ascii=False)
                generated += 1
                done = True
                print(f"[OK] {meta.name}: saved in {time.perf_counter() - t0:.1f}s")
                break
            except Exception as e:
                print(f"[WARN] {meta.name}: {e} ({attempt}/{max_attempts})")

        if not done:
            failed += 1
            print(f"[FAIL] {meta.name}: elapsed={time.perf_counter() - t0:.1f}s")

    if failed > 0:
        print(f"[DONE] prepared={prepared}, generated={generated}, skipped={skipped}, failed={failed}")
    else:
        print(f"[DONE] prepared={prepared}, generated={generated}, skipped={skipped}, failed=0")


"""Reproduce CUTEst native crashes by repeatedly building payload in an isolated process."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

from src.payload_builder import build_payload


def _ensure_pycutest_cache_env() -> str:
    cache_dir = Path(__file__).resolve().parents[1] / "pycutest_cache_holder"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYCUTEST_CACHE", str(cache_dir))
    return os.environ["PYCUTEST_CACHE"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce native crash for one CUTEst problem")
    p.add_argument("--problem", required=True, help="CUTEst problem name")
    p.add_argument("--trials", type=int, default=3, help="number of isolated build attempts")
    p.add_argument("--timeout-sec", type=int, default=120, help="timeout per trial")
    p.add_argument(
        "--output-dir",
        default="results/native_crash_repro",
        help="directory for per-problem repro json",
    )
    return p.parse_args()


def _worker(problem: str, queue) -> None:
    try:
        import pycutest

        p = pycutest.import_problem(problem)
        props = pycutest.problem_properties(problem)
        payload = build_payload(p, props=props)
        queue.put(
            {
                "ok": True,
                "n": int(p.n),
                "m": int(getattr(p, "m", 0) or 0),
                "route_key": payload.get("路由键"),
            }
        )
    except Exception as e:
        queue.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _run_trial(problem: str, timeout_sec: int) -> dict:
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(problem, queue))
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout_sec)

    elapsed = time.perf_counter() - t0
    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"ok": False, "exitcode": None, "timeout": True, "elapsed_sec": round(elapsed, 3)}

    result = queue.get() if not queue.empty() else {"ok": False, "error": "empty worker result"}
    result["exitcode"] = proc.exitcode
    result["timeout"] = False
    result["elapsed_sec"] = round(elapsed, 3)
    return result


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    cache_path = _ensure_pycutest_cache_env()

    trials = []
    for idx in range(int(max(1, args.trials))):
        rec = _run_trial(args.problem, int(max(1, args.timeout_sec)))
        rec["trial_id"] = idx
        trials.append(rec)

    crash_codes = sorted(
        {
            int(t["exitcode"])
            for t in trials
            if t.get("exitcode") is not None and int(t["exitcode"]) != 0
        }
    )

    summary = {
        "problem": args.problem,
        "pycutest_cache": cache_path,
        "trials": len(trials),
        "n_ok": sum(1 for t in trials if t.get("ok")),
        "n_timeout": sum(1 for t in trials if t.get("timeout")),
        "crash_exit_codes": crash_codes,
        "details": trials,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.problem}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[DONE] problem={args.problem}, trials={summary['trials']}, ok={summary['n_ok']}, "
        f"timeout={summary['n_timeout']}, crash_exit_codes={summary['crash_exit_codes']}"
    )
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()


"""Native crash extraction helpers for CUTEst worker crashes."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Set


LOG_CRASH_PATTERN = re.compile(
    r"^\[WARN\]\s+(?P<problem>[A-Za-z0-9_]+):\s+payload worker crashed exitcode=(?P<code>-?\d+)"
)


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def parse_log_crash_events(log_paths: Iterable[Path], target_codes: Set[int]) -> List[dict]:
    events: List[dict] = []
    for path in log_paths:
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = LOG_CRASH_PATTERN.match(line.strip())
            if not m:
                continue
            code = int(m.group("code"))
            if code not in target_codes:
                continue
            events.append(
                {
                    "problem": m.group("problem"),
                    "code": code,
                    "source": "log",
                    "source_file": str(path),
                    "detail": line.strip(),
                }
            )
    return events


def parse_runs_crash_events(df, target_codes: Set[int]) -> List[dict]:
    """Parse candidate crash events from runs dataframe.

    The dataframe can come from parquet or csv; required column is `problem`.
    Code extraction is heuristic: scan message/status/exception_type for
    patterns like 'exitcode=-11'.
    """
    if "problem" not in df.columns:
        return []

    events: List[dict] = []
    text_cols = [c for c in ["message", "status", "exception_type"] if c in df.columns]
    if not text_cols:
        return []

    for _, row in df.iterrows():
        blob_parts = []
        for col in text_cols:
            val = row.get(col)
            if val is not None:
                blob_parts.append(str(val))
        blob = " | ".join(blob_parts)

        code = None
        m = re.search(r"exitcode\s*=\s*(-?\d+)", blob)
        if m:
            code = _safe_int(m.group(1))
        if code is None:
            lowered = blob.lower()
            if "sigsegv" in lowered or "segmentation fault" in lowered:
                code = -11
            elif "sigabrt" in lowered or "aborted" in lowered:
                code = -6

        if code is None or code not in target_codes:
            continue

        events.append(
            {
                "problem": str(row["problem"]),
                "code": code,
                "source": "runs",
                "source_file": "runs",
                "detail": blob[:400],
            }
        )

    return events


def build_crash_registry(events: List[dict], target_codes: Set[int]) -> Dict[str, Any]:
    per_problem = defaultdict(lambda: {"event_count": 0, "codes": defaultdict(int), "sources": set(), "samples": []})

    for e in events:
        problem = e["problem"]
        slot = per_problem[problem]
        slot["event_count"] += 1
        slot["codes"][int(e["code"])] += 1
        slot["sources"].add(e["source"])
        if len(slot["samples"]) < 3:
            slot["samples"].append(e["detail"])

    problems = []
    for problem, item in sorted(per_problem.items(), key=lambda kv: (-kv[1]["event_count"], kv[0])):
        codes = {str(k): int(v) for k, v in sorted(item["codes"].items(), key=lambda kv: kv[0])}
        problems.append(
            {
                "problem": problem,
                "event_count": int(item["event_count"]),
                "crash_codes": codes,
                "sources": sorted(item["sources"]),
                "sample_details": item["samples"],
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_exit_codes": sorted(int(c) for c in target_codes),
        "event_count": int(len(events)),
        "problem_count": int(len(problems)),
        "problems": problems,
    }


def save_registry_json(registry: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


"""
职责：
  1. 构造 Prompt（支持按 route_key 路由模板）
  2. 调用 DeepSeek API，带重试
  3. 解析 JSON 响应，返回单个候选 x 输出对象

关键接口：
  generate_candidate(payload, cfg, prompt_spec=None, request_timeout_sec=None, max_retries_override=None) -> dict
"""
import json
import os
import time
from json import JSONDecoder
from typing import Optional

import openai  # DeepSeek 兼容 OpenAI SDK

from src.config import Config
from src.prompt_router import PromptSpec, select_prompt


def _extract_candidate_json(text: str, multi_output: bool = False, multi_k: int = 5) -> dict:
    """从模型输出中稳健抽取候选向量，兼容数组/对象与多候选模式。"""
    def _normalize(parsed):
        if isinstance(parsed, dict):
            if "xs" in parsed and isinstance(parsed["xs"], list):
                xs = parsed["xs"][: int(multi_k)]
                return {"xs": xs}
            if "x_list" in parsed and isinstance(parsed["x_list"], list):
                xs = parsed["x_list"][: int(multi_k)]
                return {"xs": xs}
            if "x" in parsed:
                return {"xs": [parsed["x"]]} if multi_output else {"x": parsed["x"]}
            return None

        if isinstance(parsed, list):
            # multi-output array-of-arrays
            if len(parsed) > 0 and all(isinstance(it, list) for it in parsed):
                xs = parsed[: int(multi_k)]
                return {"xs": xs}
            # single vector array
            return {"xs": [parsed]} if multi_output else {"x": parsed}
        return None

    try:
        parsed = json.loads(text)
        norm = _normalize(parsed)
        if norm is not None:
            return norm
    except Exception:
        pass

    decoder = JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch in "{["):
        try:
            obj, _ = decoder.raw_decode(text[start:])
            norm = _normalize(obj)
            if norm is not None:
                return norm
        except Exception:
            continue
    raise ValueError(f"No JSON array/object candidate found in LLM output: {text[:300]}")


def generate_candidate(
    payload: dict,
    cfg: Config,
    prompt_spec: Optional[PromptSpec] = None,
    request_timeout_sec: Optional[int] = None,
    max_retries_override: Optional[int] = None,
    multi_output: bool = False,
    multi_k: int = 5,
) -> dict:
    spec = prompt_spec or select_prompt(payload, cfg, multi_output=multi_output, multi_k=multi_k)
    timeout_sec = float(request_timeout_sec or cfg.llm.get("request_timeout_sec", 60))
    max_retries = int(max_retries_override or cfg.llm["max_retries"])

    client = openai.OpenAI(
        api_key=cfg.llm.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        timeout=timeout_sec,
        max_retries=0,
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.llm["model"],
                temperature=cfg.llm["temperature"],
                max_tokens=cfg.llm["max_tokens"],
                messages=[
                    {"role": "system", "content": spec.system_prompt},
                    {"role": "user", "content": spec.user_prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            out = _extract_candidate_json(raw, multi_output=multi_output, multi_k=multi_k)
            out["raw_text"] = raw
            out["model"] = cfg.llm["model"]
            return out
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(cfg.llm["retry_delay_sec"] * (attempt + 1))
            else:
                raise RuntimeError(
                    f"LLM call failed after {max_retries} retries: {type(e).__name__}: {e}"
                )

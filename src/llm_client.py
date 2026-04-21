"""
职责：
  1. 构造 Prompt（含元信息 + 策略格式要求）
  2. 调用 DeepSeek API，带重试
  3. 解析 JSON 响应，返回原始策略列表
  （校验交给 strategy_validator.py）

关键接口：
  generate_strategies(meta: dict, cfg: Config) -> list[dict]
"""
import json
import time
import re
import openai   # DeepSeek 兼容 OpenAI SDK
from src.config import Config

_SYSTEM_PROMPT = """
You are an expert in numerical optimization.
Given a description of a CUTEst benchmark problem, your task is to suggest
initialization strategies for gradient-based local optimizers (BFGS / L-BFGS-B).
Each strategy is a small JSON object. Output ONLY a JSON array of exactly {K} strategies.
No explanation, no markdown fences—just the raw JSON array.

Each strategy object MUST have exactly these fields:
  "mode":     one of ["near_x0", "center_of_bounds", "interior_from_bounds"]
  "alpha":    float in [{alpha_min}, {alpha_max}], perturbation strength
  "sparsity": one of {sparsity_choices}, fraction of dims to perturb
  "seed":     integer, for reproducibility

Rules:
- "center_of_bounds" and "interior_from_bounds" are only meaningful when has_bounds=true.
  If has_bounds=false, use "near_x0" instead.
- Aim for diversity: vary mode, alpha, and sparsity across strategies.
- Alpha close to 0.05 means gentle perturbation; 1.0 means explore widely.
""".strip()

def _build_user_prompt(meta: dict, cfg: Config) -> str:
    init_cfg = cfg.init
    system_filled = _SYSTEM_PROMPT.format(
        K=cfg.llm["K_strategies"],
        alpha_min=init_cfg["alpha_min"],
        alpha_max=init_cfg["alpha_max"],
        sparsity_choices=init_cfg["sparsity_choices"],
    )
    meta_str = json.dumps(meta, indent=2, ensure_ascii=False)
    user = (
        f"Problem metadata:\n{meta_str}\n\n"
        f"Generate {cfg.llm['K_strategies']} initialization strategies."
    )
    return system_filled, user

def _extract_json_array(text: str) -> list:
    """从模型输出中稳健地抽取第一个 JSON 数组"""
    # 先尝试直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 找第一个 [...] 块
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON array found in LLM output: {text[:300]}")

def generate_strategies(meta: dict, cfg: Config) -> list:
    system_prompt, user_prompt = _build_user_prompt(meta, cfg)
    client = openai.OpenAI(
        api_key=cfg.llm.get("api_key"),          # 也可从环境变量读
        base_url="https://api.deepseek.com/v1",
    )
    for attempt in range(cfg.llm["max_retries"]):
        try:
            resp = client.chat.completions.create(
                model=cfg.llm["model"],
                temperature=cfg.llm["temperature"],
                max_tokens=cfg.llm["max_tokens"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            return _extract_json_array(raw)
        except Exception as e:
            if attempt < cfg.llm["max_retries"] - 1:
                time.sleep(cfg.llm["retry_delay_sec"] * (attempt + 1))
            else:
                raise RuntimeError(f"LLM call failed after {cfg.llm['max_retries']} retries: {e}")
"""
职责：加载 config.yaml，提供全局配置对象；所有模块从此处 import，避免散落硬编码。
支持通过环境变量 CONFIG_PATH 指定配置文件路径，便于从任意目录运行脚本。
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env into os.environ if missing."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

@dataclass
class Config:
    problem_selection: dict
    init: dict
    solver: dict
    llm: dict
    paths: dict

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    # 便捷属性
    @property
    def K(self) -> int:
        return self.init["K"]

    @property
    def llm_candidates_dir(self) -> Path:
        return Path(self.paths["llm_candidates_dir"])

    @property
    def llm_prompts_dir(self) -> Path:
        return Path(self.paths["llm_prompts_dir"])

    @property
    def results_dir(self) -> Path:
        return Path(self.paths["results_dir"])

CFG = Config.from_yaml(os.environ.get("CONFIG_PATH", "config.yaml"))
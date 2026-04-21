"""
职责：加载 config.yaml，提供全局配置对象；所有模块从此处 import，避免散落硬编码。
支持通过环境变量 CONFIG_PATH 指定配置文件路径，便于从任意目录运行脚本。
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

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
    def strategies_dir(self) -> Path:
        return Path(self.paths["strategies_dir"])

    @property
    def results_dir(self) -> Path:
        return Path(self.paths["results_dir"])

CFG = Config.from_yaml(os.environ.get("CONFIG_PATH", "config.yaml"))
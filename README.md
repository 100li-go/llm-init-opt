# LLM Init Opt

验证 DeepSeek LLM 生成的初值策略是否在 CUTEst 基准测试中优于随机初值（BFGS / L-BFGS-B）。

## 环境配置

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_deepseek_api_key"
# 设置 CUTEst 环境变量（参考 pycutest 文档）
export CUTEST=/path/to/cutest
export MASTSIF=/path/to/mastsif
```

## 执行顺序

```bash
make smoke       # 冒烟测试（5题，先验证链路是否通畅）
make select      # 筛选300题，生成 problem_set.json
make strategies  # 调用 DeepSeek API，生成 llm_strategies/
make run         # 批量求解，写入 results/runs.parquet（后台运行）
make analyze     # 聚合分析，输出图表与报告
```

## 输出文件

| 文件 | 说明 |
|---|---|
| `problem_set.json` | 300道题的元信息（固定种子，可复现） |
| `llm_strategies/*.json` | 每题的LLM策略（K=10条） |
| `results/runs.parquet` | 全量run级别数据 |
| `results/summary.parquet` | 聚合汇总表（best/median指标） |
| `figures/perf_profile_nfev.pdf` | nfev Performance Profile（A/B类） |
| `figures/perf_profile_time.pdf` | time Performance Profile（A/B类） |
| `results/report.txt` | 文字结论报告（RQ1-RQ4） |

## 模块说明

| 模块 | 职责 |
|---|---|
| `src/config.py` | 配置加载，支持 `CONFIG_PATH` 环境变量 |
| `src/problem_selector.py` | 题目筛选（n≤200, m=0）与固定种子抽样 |
| `src/meta_extractor.py` | 提取元信息摘要（供LLM Prompt使用） |
| `src/llm_client.py` | DeepSeek API调用，含重试机制 |
| `src/strategy_validator.py` | 校验并修复LLM输出的策略 |
| `src/initializer.py` | 三种来源初值生成（cutest/random/llm），含clip和回退 |
| `src/solver.py` | SciPy BFGS/L-BFGS-B封装，含超时保护 |
| `src/runner.py` | 批量调度，支持断点续跑 |
| `src/aggregator.py` | best/median/成功率聚合 |
| `src/analysis/stats.py` | Wilcoxon成对检验（RQ1-RQ3） |
| `src/analysis/performance_profile.py` | Dolan-Moré Performance Profile绘图 |
| `src/analysis/report.py` | 文字结论报告生成 |

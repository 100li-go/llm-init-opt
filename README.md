# LLM Init Opt

验证 DeepSeek LLM 直接生成候选初值 `x`（并做分层后处理）是否在 CUTEst 基准测试中优于随机初值，并区分 `random_raw` 与 `random_post` 两条随机基线。

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
make select      # 筛选问题，生成 problem_set.json
make candidates  # 调用 DeepSeek API，生成 llm_candidates/
make run         # 批量求解，写入 results/runs.parquet（后台运行）
make analyze     # 聚合分析，输出图表与报告

# 可选：native crash 归因与复现
python scripts/o3_extract_native_crash_registry.py --runs results/runs.parquet --logs-dir logs
python scripts/repro_native_crash.py --problem HS101 --trials 3 --timeout-sec 120

# 可选：prompt 覆盖检查（自动挑样本并断言）
python scripts/check_prompt_coverage.py --output-dir results/prompt_samples/coverage_check
```

## 输出文件

| 文件 | 说明 |
|---|---|
| `problem_set.json` | 问题元信息（含 ConstraintTag/ObjectiveTag） |
| `llm_candidates/*.json` | 每题的 LLM 候选初值（兼容 `{"x": [...]}` 与 `{"xs": [[...], ...]}`） |
| `results/runs.parquet` | 全量 run 级别数据 |
| `results/summary.parquet` | 聚合汇总表（best/median 指标） |
| `results/summary_problem_arm_v2.parquet` | problem×arm 汇总（四 arm：`cutest`/`random_raw`/`random_post`/`llm`） |
| `results/route_key_problem_stats_v2.parquet` | route_key problem-weighted v2 指标 |
| `results/stats_all_arms.parquet` | 多 arm 成对对比统计（含 `llm_vs_random_raw`、`llm_vs_random_post`） |
| `figures/perf_profile_nfev.pdf` | nfev Performance Profile |
| `figures/perf_profile_time.pdf` | time Performance Profile |
| `results/report.txt` | 文字结论报告（RQ1-RQ4） |
| `results/native_crash_registry.json` | native crash(-6/-11) 题库与证据汇总 |
| `results/native_crash_repro/<problem>.json` | 单题复现实验结果 |

## 模块说明

| 模块 | 职责 |
|---|---|
| `src/config.py` | 配置加载，支持 `CONFIG_PATH` 环境变量 |
| `src/problem_selector.py` | 题目筛选与细粒度标签计算 |
| `src/problem_tags.py` | `ConstraintTag`/`ObjectiveTag` 规则化判定 |
| `src/payload_builder.py` | 构建高信息密度 payload（含约束违反诊断） |
| `src/prompt_router.py` | 按 `ConstraintTag` 选择 Prompt 模板（支持单输出与 K 输出） |
| `src/llm_client.py` | DeepSeek API 调用，返回候选 JSON（`x` 或 `xs`） |
| `src/llm_output_validator.py` | 校验 LLM 输出候选向量（`x` / `xs`）长度与有限性 |
| `src/initializer.py` | 四种来源初值生成（`cutest`/`random_raw`/`random_post`/`llm`）与候选后处理 |
| `src/solver.py` | SciPy BFGS/L-BFGS-B 封装，含超时保护 |
| `src/runner.py` | 批量调度，支持断点续跑 |
| `src/aggregator.py` | best/median/成功率聚合 |
| `src/analysis/stats.py` | Wilcoxon 成对检验（RQ1-RQ3） |
| `src/analysis/performance_profile.py` | Dolan-Moré Performance Profile 绘图 |
| `src/analysis/report.py` | 文字结论报告生成 |

## v2 统计口径说明

- `summary_problem_arm_v2` 以 **problem 级别** 对四 arm 聚合，避免 run 级偏置。
- 新增失败纳入指标：`best_time_cost_capped`、`best_nfev_cost_capped`。
  - 成功 run：取真实值。
  - 失败 run：替换为上限惩罚（`llm.time_cap_sec` / `llm.nfev_cap`）。
- random 家族（`random_raw` / `random_post`）的 K 次公平成本：`total_time_sec`、`total_nfev`（K 次总消耗）。

## 关键配置（config.yaml）

- capped cost：`llm.time_cap_sec`、`llm.nfev_cap`
- payload 两档：`llm.payload_full_n_max`、`llm.payload_full_m_max`、`llm.payload_head_k`、`llm.payload_full_constraint_m_max`
- Jacobian 提示稀疏化：`llm.jacobian_topk`、`llm.jacobian_row_nnz_cap`、`llm.jacobian_abs_tol`


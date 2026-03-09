# SpecDiff with DLLM

An experimental project for reproducing and validating the SpecDiff idea: combine speculative decoding with gradient-based scheduling in a multi-client setting, then evaluate throughput, latency, AoI, and fairness.

## Purpose

- Run speculative decoding with real models (Draft + Target).
- Dynamically allocate per-client draft length `S_i` under a fixed cloud capacity.
- Compare scheduling objectives:
  - `paper_log_r`
  - `r_plus_inv_alpha_aoi`
- Produce analyzable outputs (JSON + web dashboard).

## Key Features

- Real-model inference:
  - Target: `Qwen/Qwen2.5-7B-Instruct`
  - Draft: `Qwen/Qwen2.5-0.5B-Instruct`
- Multi-client scheduling simulation (currently fixed to 3 clients).
- Shared-batch verification stage on target model.
- Metrics:
  - Goodput
  - AoI (Age of Information)
  - Slot Latency
  - Fairness (`R/S`)
  - Speculative vs target-only baseline speed comparison
- Web dashboard (`app.py`) for visualization.

## Project Structure

```text
.
├── app.py            # Web server + dashboard + API
├── main.py           # CLI experiment entry
├── engine.py         # Real speculative decoding engine (Draft/Target)
├── scheduler.py      # Gradient-based resource scheduler
├── core.py           # State update and objective utilities
├── test_logic.py     # Core logic tests
├── requirements.txt
└── start.sh          # Environment bootstrap and startup script
```

## Requirements

- Python 3.10+ (`start.sh` uses Python 3.10 by default)
- NVIDIA GPU (24GB VRAM recommended, e.g., RTX 3090)
- CUDA-compatible PyTorch build
- Access to HuggingFace model hub

> Note: Models are loaded in 4-bit quantization (`bitsandbytes`) to reduce VRAM usage.

## Install and Run

### Option 1: One-command bootstrap (recommended)

```bash
bash start.sh
```

Optional:

```bash
TORCH_CUDA=cu121 bash start.sh
CONDA_ENV=spec bash start.sh
```

The script will:

1. Prepare conda/venv environment
2. Install PyTorch for selected CUDA version
3. Install project dependencies
4. Check CUDA availability
5. Run `main.py`

### Option 2: Manual CLI run

```bash
pip install -r requirements.txt
python main.py
```

Optional flags:

```bash
python main.py --objective-mode paper_log_r --objective-alpha 1.0
python main.py --objective-mode r_plus_inv_alpha_aoi --objective-alpha 1.0
```

## Web Dashboard

Start:

```bash
python app.py
```

Open:

```text
http://localhost:8000
```

The UI allows configuring slots, capacity, `eta`, `beta`, objective settings, and triggers `/api/run`.

## Output

- `app.py` attempts to save each run to:
  - `run_results/result_YYYYMMDD_HHMMSS.json`
- Main output fields include:
  - `slot_goodput_history`
  - `slot_aoi_s`
  - `slot_latency_ms`
  - `avg_allocations`
  - `avg_goodputs`
  - `fairness`
  - `summary` (including `speedup_ratio`)

## Tests

```bash
pytest -q
```

Current coverage includes:

- `mu_func` numerical stability
- Scheduler capacity constraint consistency
- Fairness incentive behavior
- Exponential smoothing convergence

## Notes

- Client count is currently fixed to 3 (memory constraint).
- First run can be slow because model weights may need downloading.
- By default, model cache is set under project root: `.model_cache/huggingface`.
  - You can override it with: `SPECDIFF_MODEL_CACHE_DIR=/path/to/cache`.

## License

No license is declared yet. Add a `LICENSE` file before open-source distribution.

---

# SpecDiff with DLLM（中文说明）

一个用于复现和验证 SpecDiff 思路的实验项目：在多客户端场景下，结合推测解码（Speculative Decoding）与梯度式资源调度，评估吞吐、时延、AoI 和公平性。

## 项目目标

- 使用真实模型进行推测解码实验（Draft + Target）。
- 在固定云端预算下，为多个客户端动态分配 Draft 长度 `S_i`。
- 比较不同调度目标下的系统表现：
  - `paper_log_r`
  - `r_plus_inv_alpha_aoi`
- 输出可分析的实验结果（JSON + 可视化仪表盘）。

## 主要特性

- 真实模型推理：
  - Target: `Qwen/Qwen2.5-7B-Instruct`
  - Draft: `Qwen/Qwen2.5-0.5B-Instruct`
- 多客户端调度仿真（当前固定为 3 个客户端）。
- 目标模型侧共享 batch 验证阶段。
- 指标统计：
  - Goodput
  - AoI（Age of Information）
  - Slot Latency
  - Fairness（`R/S`）
  - Speculative 与 target-only baseline 速度对比
- Web Dashboard（`app.py`）可视化查看结果。

## 目录结构

```text
.
├── app.py            # Web 服务 + 仪表盘 + API
├── main.py           # 命令行实验入口
├── engine.py         # 真实推测解码引擎（Draft/Target）
├── scheduler.py      # 梯度式资源调度
├── core.py           # 状态更新与目标函数相关工具
├── test_logic.py     # 核心逻辑测试
├── requirements.txt
└── start.sh          # 环境准备与启动脚本
```

## 环境要求

- Python 3.10+（`start.sh` 默认按 3.10 建环境）
- NVIDIA GPU（建议 24GB 显存，如 RTX 3090）
- CUDA 对应的 PyTorch 版本
- 可访问 HuggingFace 模型仓库

> 说明：项目默认加载 4-bit 量化模型（`bitsandbytes`），以降低显存占用。

## 安装与运行

### 方式 1：一键脚本（推荐）

```bash
bash start.sh
```

可选参数：

```bash
TORCH_CUDA=cu121 bash start.sh
CONDA_ENV=spec bash start.sh
```

脚本会自动：

1. 准备 conda/venv 环境
2. 安装 PyTorch（按 CUDA 版本）
3. 安装项目依赖
4. 检查 CUDA
5. 启动 `main.py`

### 方式 2：手动运行（CLI）

```bash
pip install -r requirements.txt
python main.py
```

可选参数：

```bash
python main.py --objective-mode paper_log_r --objective-alpha 1.0
python main.py --objective-mode r_plus_inv_alpha_aoi --objective-alpha 1.0
```

## Web 仪表盘

启动：

```bash
python app.py
```

访问：

```text
http://localhost:8000
```

页面支持设置 slots、capacity、`eta`、`beta`、objective 等参数，并调用 `/api/run` 返回实验结果。

## 输出结果

- `app.py` 每次运行会尝试保存结果到：
  - `run_results/result_YYYYMMDD_HHMMSS.json`
- 返回/保存的主要字段：
  - `slot_goodput_history`
  - `slot_aoi_s`
  - `slot_latency_ms`
  - `avg_allocations`
  - `avg_goodputs`
  - `fairness`
  - `summary`（含 `speedup_ratio` 等）

## 测试

```bash
pytest -q
```

当前测试覆盖：

- `mu_func` 数值稳定性
- 调度资源约束一致性
- 公平性激励方向
- 平滑更新收敛行为

## 注意事项

- 当前实现中，客户端数量固定为 3（显存约束）。
- 首次加载模型会较慢，且需要网络下载模型权重。
- 模型缓存默认在项目根目录：`.model_cache/huggingface`。
  - 可用环境变量覆盖：`SPECDIFF_MODEL_CACHE_DIR=/path/to/cache`。

## 许可证

暂未声明。若需要开源发布，建议补充 `LICENSE` 文件。

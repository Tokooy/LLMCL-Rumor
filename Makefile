# ===================================================================== #
# LLMCL-Rumor 常用命令
# --------------------------------------------------------------------- #
# 说明：这些目标只是把常用命令固化下来，方便查阅与复现；
# 训练类目标会真实占用 GPU/API 额度，请确认配置后再执行。
# ===================================================================== #

PYTHON ?= python
CONFIG ?= configs/base.yaml
EXPERIMENT ?= configs/experiments/proposed-3.yaml
DATASET ?= twitter15

.PHONY: help check syntax static tracked test prepare augment train-cl joint-align evaluate clean

help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- #
# 自检（不运行训练、不加载模型）
# --------------------------------------------------------------------- #
syntax:  ## AST 语法自检（每个 .py 文件）
	$(PYTHON) scripts/check_syntax.py --verbose

static:  ## AST 静态一致性自检（__all__ / 漏 import / 重名）
	$(PYTHON) scripts/check_static.py

tracked:  ## 仓库完整性自检（源码是否都被 git 追踪 / 是否被 gitignore 误伤）
	$(PYTHON) scripts/check_tracked.py

check: syntax static tracked  ## 执行全部三项静态自检

test:  ## 运行单元测试（无 GPU 环境会自动跳过 torch 用例）
	$(PYTHON) -m pytest tests -q

# --------------------------------------------------------------------- #
# 数据流水线
# --------------------------------------------------------------------- #
prepare:  ## 数据预处理（默认 twitter15，需先把原始文件放进 data/raw/）
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --dataset $(DATASET)

prepare-demo:  ## 用仓库自带的 10 条演示数据跑通预处理
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --dataset demo

augment:  ## 第 1 轮 LLM 数据增强
	$(PYTHON) scripts/augment_data.py --config $(EXPERIMENT) --dataset $(DATASET) --round 1

augment-demo:  ## 第 1 轮增强（规则后端，不需要 GPU 与模型权重）
	$(PYTHON) scripts/augment_data.py --config $(EXPERIMENT) --dataset demo \
		--round 1 --backend demo

# --------------------------------------------------------------------- #
# 训练与评测
# --------------------------------------------------------------------- #
train-cl:  ## 单轮次 CL 训练（对应 Proposed-1/2/3）
	$(PYTHON) scripts/train_cl.py --config $(EXPERIMENT) --dataset $(DATASET)

joint-align:  ## 联合对齐训练（对应 Proposed-4/5/6，论文 Algorithm 2）
	$(PYTHON) scripts/joint_align.py --config $(EXPERIMENT) --dataset $(DATASET)

evaluate:  ## 评测指定 checkpoint
	$(PYTHON) scripts/evaluate.py --config $(EXPERIMENT) --dataset $(DATASET) \
		--checkpoint $(CHECKPOINT)

# --------------------------------------------------------------------- #
# 清理
# --------------------------------------------------------------------- #
clean:  ## 清理 Python 缓存（不动 outputs/ 与 data/）
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache

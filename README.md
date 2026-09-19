# LLMCL-Rumor

**基于 LLM 增强对比学习的社交网络虚假信息检测** —— 论文方法的代码复现。

> 复现对象：《基于LLM增强对比学习的社交网络虚假信息检测方法》(2025-01-22)
> 代码起点：开源项目 `BERT/Bert-TextClassification-master`（BERT 文本分类基线），
> 本仓库在其基础上按论文方法重构为标准的深度学习项目结构。
> 原项目代码已完整保留在 `src/models/backbones/` 与 `docs/differences_from_baseline.md` 中可查。

---

## 1. 论文方法一句话概括

用 **LLM 做数据增强**（生成语义一致、表达多样的样本）→ 喂给 **BERT + MLP 投影头的对比学习网络**
（InfoNCE 拉近"原样本 ↔ 其增强样本"）→ 再用 **CL 的对比损失反过来指导 LLM 微调与合并**
（LoRA + TIES-Merging），形成"增强 → 训练 → 反馈 → 再增强"的闭环。

三个阶段与论文 §3 的对应关系：

| 论文阶段 | 本仓库实现位置 |
|---|---|
| 3.1 数据预处理与 Prompt 编排 | `data/processors/` + `src/llm/prompts.py` |
| 3.2 LLM 辅助的特征提取与标签预测 | `src/models/` + `src/training/losses.py` |
| 3.3 LLM 与 CL 网络的对齐（Algorithm 1 & 2） | `src/llm/ties_merge.py` + `src/training/joint_trainer.py` |

---

## 2. 目录结构与各子文件夹用途

```
LLMCL-Rumor/
├── configs/            【配置层】所有超参与路径，代码里不写死任何数字
├── data/               【数据层】原始数据 → 预处理 → 增强数据的全部落盘位置与读取代码
├── src/                【源码层】四层架构，依赖单向向下
│   ├── models/         句向量编码器 / 投影头 / 分类器 / 总装
│   ├── llm/            LLM 后端抽象、Prompt 编排、输出校验、LoRA 微调、TIES 合并
│   ├── training/       损失函数、CL 训练器、联合对齐训练器、评估、可视化
│   └── utils/          配置解析、日志、随机种子、JSONL 读写（不依赖 torch）
├── scripts/            【入口层】按流水线顺序排列的可执行脚本
├── tests/              【测试层】纯 CPU 单测，不需要模型权重（可在装完依赖后直接跑）
├── docs/               【文档层】结构说明、算法逐行对照、与原项目的差异说明
├── reference/          【参考层】原开源项目快照（只读，不参与运行）
└── outputs/            【产物层】checkpoints / logs / results / figures（不入库）
```

> **目录整理说明**：原开源项目 `Bert-TextClassification-master` 已完整移入
> `reference/baseline/`，并补上 `reference/README.md` 说明每个文件对应新项目的哪个位置。
> 这样仓库根目录只剩下"标准深度学习项目"该有的目录，
> 参考代码被明确隔离在 `reference/` 下，不会再与主流程混淆。

### 2.1 `configs/` —— 配置

| 文件 | 用途 |
|---|---|
| `base.yaml` | 全部默认超参：数据、模型、LLM、训练、评估、可视化 |
| `llm/qwen7b.yaml` | 基座换成 Qwen2.5-7B（论文主实验对象） |
| `llm/qwen13b.yaml` | 基座换成 Qwen2.5-13B（论文 Proposed-6） |
| `experiments/proposed-{1..6}.yaml` | 论文 Table 2 的 6 个变体，只写与默认值的差异 |
| `experiments/README.md` | 6 个变体的对照表与推定项说明 |
| `prompts/augment_default.txt` | 数据增强 Prompt 模板（对应论文 Fg.3） |

配置支持 `defaults:` 继承链，例如 `proposed-4.yaml → llm/qwen7b.yaml → base.yaml`。
加载后可用属性访问：`cfg.training.alignment.max_finetune_rounds`。

### 2.2 `data/` —— 数据

| 路径 | 用途 |
|---|---|
| `raw/` | 放置 Twitter15 / Twitter16 原始文件（**不入库**，见 `data/README.md`） |
| `processed/` | 预处理与增强后的 JSONL（**不入库**） |
| `samples/demo_twitter15.jsonl` | 我手写的极小样例（10 条），用于通读数据流与跑通冒烟测试 |
| `processors/twitter_rumor.py` | 原始格式 → 统一数据实例（`uid / string_value / replies / label`） |
| `processors/reply_flatten.py` | 回复树的展平与截断（BFS/DFS），生成编码器输入文本 |
| `dataset.py` | PyTorch `Dataset`：成对返回（原样本, 增强样本）供对比学习使用 |

### 2.3 `src/models/` —— 对比学习网络（论文 §3.2）

| 文件 | 用途 |
|---|---|
| `encoder.py` | BERT 句向量编码器，取 `[CLS]` 表示 |
| `projector.py` | MLP 投影头 `W2·σ(W1·h)`，论文式(1) |
| `classifier.py` | 全连接 + softmax 分类器，论文式(4) |
| `cl_model.py` | 总装：编码器 + 投影头 + 分类器，一次前向返回 logits / 投影特征 / 句向量 |
| `backbones/` | 原开源项目保留的其它文本分类骨干（BertCNN/LSTM/ATT/RCNN/DPCNN/HAN），**论文未使用**，仅供对照与消融 |

### 2.4 `src/llm/` —— LLM 侧（论文 §3.1 与 §3.3）

| 文件 | 用途 |
|---|---|
| `base.py` | 后端抽象接口：`generate()` / `finetune()` / `export_task_vector()` |
| `hf_backend.py` | `transformers` 本地后端（支持 LoRA 微调与导出任务向量） |
| `api_backend.py` | OpenAI 兼容接口后端（快速跑通；不支持梯度微调，微调轮次自动跳过） |
| `factory.py` | 按配置构建后端 |
| `prompts.py` | Prompt 编排：四条设计目标（全视角/格式一致/多样性/语义一致）逐条落地 |
| `parser.py` | 增强结果校验：JSON 结构一致性、回复数量一致性、uid 一致性、字符级多样性 |
| `augmentor.py` | 增强流水线：缓存、重试、并发、按 epoch 落盘 |
| `lora.py` | LoRA 注入、训练、任务向量导出 |
| `task_vector.py` | 任务向量 `τ = θ_ft - θ_base` 的表示与运算 |
| `ties_merge.py` | **Algorithm 1**：Trim → Elect Sign → Disjoint Merge → 合并回基座 |

### 2.5 `src/training/` —— 训练与评估

| 文件 | 用途 |
|---|---|
| `losses.py` | InfoNCE / 有监督对比损失，论文式(3) |
| `cl_trainer.py` | 单个增强轮次内的 CL 训练与验证 |
| `joint_trainer.py` | **Algorithm 2** 的联合对齐主循环（增强 → 训练 → λ/ω 更新 → 微调 → TIES 合并） |
| `evaluate.py` | ACC / 逐类 F1 / Avg F1，对应论文 Table 3–8 |
| `visualize.py` | t-SNE 特征分布图，对应论文 Fg.7–Fg.10 |
| `optim.py` | 优化器与学习率调度（含分层学习率、warmup） |

### 2.6 `scripts/` —— 入口

| 脚本 | 作用 |
|---|---|
| `prepare_data.py` | 原始数据 → `data/processed/<dataset>/{train,dev,test}.jsonl` |
| `augment_data.py` | 对训练集做第 k 轮 LLM 增强 → `augmented_round{k}.jsonl` |
| `train_cl.py` | 单轮次对比学习训练（对应 Proposed-1/2/3，M=0） |
| `joint_align.py` | 完整对齐流程（对应 Proposed-4/5/6，M>0） |
| `evaluate.py` | 加载 checkpoint 评测并导出指标 / t-SNE 图 |
| `check_syntax.py` | AST 语法自检（不导入任何依赖） |
| `check_static.py` | AST 静态一致性自检（`__all__` 覆盖 / 疑似漏 import / 顶层重名） |
| `check_tracked.py` | 仓库完整性自检（源码是否都被 git 追踪、是否被 gitignore 误伤） |

### 2.7 `reference/` —— 原开源项目快照

| 路径 | 用途 |
|---|---|
| `reference/baseline/` | `Bert-TextClassification-master` 原样快照（只读参考，**不参与运行**） |
| `reference/README.md` | 原项目每个文件对应新项目的哪个位置、为什么不能直接运行 |

保留它的三个理由：可追溯改造范围、可做骨干消融、可对照 `pytorch_pretrained_bert` 版本。
主流程**不引用**该目录下的任何文件。

---

## 3. 快速开始

> ⚠️ 本仓库交付时**未运行过任何训练或数据增强**（按需求约定），
> 请先按下面的顺序自检，再开始正式训练。

```bash
# 0) 环境
conda create -n llmcl python=3.10 -y && conda activate llmcl
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .          # 可选但推荐：让 src/ 与 data/ 在任意目录都能 import

# 1) 自检（纯 CPU，不需要模型权重和数据）
pytest tests -q
python scripts/check_syntax.py && python scripts/check_static.py && python scripts/check_tracked.py

# 2) 数据准备（把 Twitter15/16 放到 data/raw/twitter15/ 后执行）
python scripts/prepare_data.py --config configs/base.yaml --dataset twitter15

# 3) 用样例数据检查整条流水线（10 条样本，几分钟内跑完）
python scripts/prepare_data.py --config configs/base.yaml --dataset demo
python scripts/augment_data.py --config configs/base.yaml --round 1 --split train

# 4) 复现论文某个变体
python scripts/train_cl.py    --config configs/experiments/proposed-3.yaml
python scripts/joint_align.py --config configs/experiments/proposed-5.yaml
python scripts/evaluate.py    --config configs/experiments/proposed-5.yaml \
                              --checkpoint outputs/checkpoints/proposed-5/best.pt
```

数据格式与放置方式见 `data/README.md`；算法与实现的逐行对照见 `docs/`。

---

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/project_structure.md` | 目录结构的逐文件说明与依赖方向 |
| `docs/architecture.md` | 模型与训练流程的详细设计（含张量形状） |
| `docs/algorithms.md` | Algorithm 1 / Algorithm 2 与代码的逐行对照 |
| `docs/differences_from_baseline.md` | 与原开源项目的差异（含 `pytorch_pretrained_bert → transformers` 迁移说明） |
| `docs/implementation_notes.md` | 论文未明确处的实现选择与依据（λ 的定义、Proposed-6 的 M 值等） |
| `docs/reproduction_checklist.md` | 复现论文 Table 3–8 的操作清单 |

---

## 5. 引用

```bibtex
@article{llmcl_rumor_2025,
  title  = {基于LLM增强对比学习的社交网络虚假信息检测方法},
  year   = {2025}
}
```

依赖的关键方法：

- InfoNCE / 对比学习：[Hadsell et al. 2006](https://www.cs.toronto.edu/~hinton/absps/pami.pdf)（文献[33]）、
  [He et al. 2020, MoCo](https://arxiv.org/abs/1911.05722)（文献[34]）
- TIES-Merging（Algorithm 1 的算法原型）：[Yadav et al. 2024](https://arxiv.org/abs/2306.01708)（文献[35]）
- LoRA：[Hu et al. 2021](https://arxiv.org/abs/2106.09685)（文献[31]）
- Qwen：[Bai et al. 2023](https://arxiv.org/abs/2309.16609)（文献[36]）

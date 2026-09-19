# 项目结构说明

本文件逐文件说明目录用途，是 `README.md` 中"目录结构"一节的细化版本。

---

## 0. 依赖方向（重要）

```
                scripts/*
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
  src/training   src/llm   src/models
        └──────────┴──────────┘
                   │
                   ▼
              src/utils  （不依赖 torch/transformers）
                   │
                   ▼
                 data/*  （processors 不依赖 torch；dataset 依赖 torch）

  reference/baseline/*  ← 独立保留，不被上面任何一层引用
```

三条硬约束：

1. `src/utils` 与 `data/processors` **不导入 torch**，因此数据预处理可以在
   没有 GPU 的机器上运行；
2. `src/llm` 与 `src/models` 只通过 `src.utils` 与 `data.processors` 交互，
   两者之间**没有**直接依赖（LLM 不需要知道 CL 网络的内部结构）；
3. `src/training` 是唯一同时依赖 `src.llm` 与 `src/models` 的层——这正是论文
   "联合优化"的落点。

---

## 1. 顶层

| 路径 | 用途 |
|---|---|
| `README.md` | 项目总览、目录速查表、快速开始 |
| `requirements.txt` | 依赖清单（含安装顺序说明） |
| `pytest.ini` | 测试配置与 marker 定义 |
| `Makefile` | 常用命令（自检 / 数据 / 训练 / 评测） |
| `.gitignore` | 忽略数据、权重与训练产物 |
| `configs/` | 全部配置与 Prompt 模板 |
| `data/` | 数据与读取数据的代码 |
| `src/` | 源码（四层） |
| `scripts/` | 可执行入口 |
| `tests/` | 单元测试 |
| `docs/` | 文档 |
| `outputs/` | 训练产物（不入库） |
| `reference/baseline/` | 原开源项目快照（只读参考，见第 8 节） |

---

## 2. `configs/`

| 文件 | 用途 |
|---|---|
| `base.yaml` | 默认超参总表。分 `paths / device / data / model / llm / training / evaluation / visualization` 八节 |
| `llm/qwen7b.yaml` | 基座换成 Qwen2.5-7B（论文主实验对象） |
| `llm/qwen13b.yaml` | 基座换成 Qwen2.5-13B（论文 Proposed-6） |
| `experiments/proposed-{1..6}.yaml` | 论文 Table 2 的六个变体 |
| `experiments/README.md` | 六个变体的对照表与"推定项"说明 |
| `prompts/augment_default.txt` | 数据增强 Prompt 模板（论文 Fg.3） |

继承链示例（`proposed-4.yaml`）：

```
proposed-4.yaml  →  llm/qwen7b.yaml  →  base.yaml
   （实验差异）        （基座差异）        （默认值）
```

加载后可用属性访问：`config.training.alignment.max_finetune_rounds`；
`config.get_path("_config_sources", [])` 能列出这份配置最终由哪些文件合成，
排查"这个值到底来自哪"时很有用。

---

## 3. `data/`

| 路径 | 用途 |
|---|---|
| `raw/` | 原始 Twitter15/16（不入库，见 `data/README.md`） |
| `processed/` | 预处理与增强产物（不入库） |
| `samples/` | 仓库自带的 10 条演示数据与生成/校验脚本 |
| `processors/data_model.py` | 统一数据模型 `DataInstance` / `Reply` |
| `processors/reply_flatten.py` | 回复树展平与 token 预算分配 |
| `processors/twitter_rumor.py` | Twitter15/16 原始格式解析与数据划分 |
| `dataset.py` | `PairDataset`（原样本 ↔ 增强样本配对）与 collate 函数 |

---

## 4. `src/models/`

| 文件 | 对应论文 |
|---|---|
| `encoder.py` | §3.2 "基于 BERT 的特征提取网络" |
| `projector.py` | §3.2 式(1) 的 MLP 投影头 |
| `classifier.py` | §3.2 式(4) 的分类器 |
| `cl_model.py` | Fig.1 的 "CL classification network" 总装 |
| `tokenization.py` | BERT 分词（数据集契约） |
| `backbones/` | 原项目的其它骨干，**论文未使用**，仅供消融 |

---

## 5. `src/llm/`

| 文件 | 对应论文 |
|---|---|
| `prompts.py` | §3.1 的 Prompt 编排（Fg.3 的四条目标） |
| `parser.py` | §3.1 输出格式一致性与语义一致的校验 |
| `base.py` | 后端能力抽象 |
| `hf_backend.py` | 本地 Qwen（增强 + LoRA 微调 + 任务向量） |
| `api_backend.py` | OpenAI 兼容接口（只支持增强） |
| `demo_backend.py` | 规则伪增强（无模型也能跑通） |
| `augmentor.py` | 增强流水线（缓存/重试/并发/落盘） |
| `factory.py` | 按配置装配上述组件 |
| `lora.py` | §3.3 自举式微调样本构造与任务向量导出 |
| `task_vector.py` | Algorithm 1 的输入 τ |
| `ties_merge.py` | **Algorithm 1** 与式(6)(7)(8)(9) |

---

## 6. `src/training/`

| 文件 | 对应论文 |
|---|---|
| `losses.py` | 式(2)(3)(4) 的对比损失与联合损失 |
| `cl_trainer.py` | §3.2 的 CL 训练循环 |
| `joint_trainer.py` | **Algorithm 2** 的联合对齐 |
| `evaluate.py` | Table 3–8 的指标 |
| `visualize.py` | Fg.7–Fg.10 的 t-SNE 图 |

---

## 7. `scripts/`

| 脚本 | 用途 |
|---|---|
| `prepare_data.py` | 原始数据 → `train/dev/test.jsonl` |
| `augment_data.py` | 第 k 轮 LLM 增强 |
| `train_cl.py` | 单轮次 CL 训练（Proposed-1/2/3） |
| `joint_align.py` | 联合对齐（Proposed-4/5/6） |
| `evaluate.py` | 评测与可视化 |
| `check_syntax.py` | AST 语法自检（不导入依赖） |
| `check_static.py` | AST 静态一致性自检 |

---

## 8. `tests/` 与 `reference/`

`tests/` 四个文件分别覆盖：

| 文件 | 覆盖内容 |
|---|---|
| `test_ties_merge.py` | Algorithm 1 三步算子、式(7)(8)(9) |
| `test_losses.py` | 式(2)(3)(4)、配对语义、数值稳定性 |
| `test_prompts_and_parser.py` | Prompt 编排、输出校验、demo 后端端到端 |
| `test_data.py` | 数据模型、回复展平、Twitter 解析、配对数据集 |
| `test_config_and_metrics.py` | 配置继承、IO、评估指标 |

依赖 torch 的用例带 `@pytest.mark.torch` 并用 `importorskip`，无 torch 环境下自动 skip。

`reference/baseline/` 是原开源项目 `Bert-TextClassification-master` 的完整快照，
**只读参考**：它使用已停止维护的 `pytorch_pretrained_bert`，无法直接运行，
保留它是为了对照"哪些代码是论文需要的、哪些不是"。迁移说明见
`docs/differences_from_baseline.md`。

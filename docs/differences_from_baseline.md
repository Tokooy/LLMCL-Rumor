# 与原开源项目的差异说明

起点：`Bert-TextClassification-master`（下称"原项目"），
一份基于 `pytorch_pretrained_bert` 的 BERT 文本分类代码集。
完整快照保留在 `reference/baseline/`。

---

## 1. 目录结构：从"平铺脚本"到"分层项目"

### 原项目结构

```
Bert-TextClassification-master/
├── BertATT/       args.py + BertATT.py
├── BertCNN/       args.py + BertCNN.py
├── BertCNNPlus/   args.py + BertCNNPlus.py
├── BertDPCNN/     args.py + BertDPCNN.py
├── BertHAN/       args.py + BertHAN.py
├── BertLSTM/      args.py + BertLSTM.py
├── BertOrigin/    args.py + BertOrigin.py
├── BertRCNN/      args.py + BertRCNN.py
├── Models/        Conv.py Embedding.py Highway.py Linear.py LSTM.py
├── Processors/    MultiNewsProcessor.py MultiSentenceProcessor.py
├── Utils/         Classifier_utils.py load_datatsets.py load_multi_datasets.py
│                  MultiSentences_utils.py utils.py
├── main.py  multi_main.py  train_evalute.py
├── run_ag_news.py  run_dbpedia.py  run_imdb.py  run_Multi_CNews.py
├── run_SST2.py  run_THUCNews.py  run_yahoo_answers.py
├── run_yelp_review_full.py  run_yelp_review_polarity.py
└── docs/  README.md  requirements.txt(空)  LICENSE(空)
```

特点：**每个模型一个目录，且各带一份几乎相同的 `args.py`**；
`run_*.py` 与模型目录通过 `from X import args` 动态绑定。

### 新结构

```
LLMCL-Rumor/
├── configs/           配置与 Prompt 模板（超参不写在代码里）
├── data/              数据 + 读数据的代码
├── src/
│   ├── models/        模型组件（encoder / projector / classifier / 总装）
│   ├── llm/           LLM 侧（增强 / 微调 / 合并）
│   ├── training/      损失 / 训练器 / 评估 / 可视化
│   └── utils/         配置 / 日志 / 种子 / IO
├── scripts/           5 个流水线入口 + 2 个静态自检脚本
├── tests/             5 个测试文件
├── docs/              6 篇文档
└── reference/baseline/  原项目快照（只读）
```

改造理由：

| 问题 | 原项目做法 | 新做法 |
|---|---|---|
| 超参分散 | 7 份近乎重复的 `args.py` | 一份 `configs/base.yaml` + 差异化的实验配置 |
| 模型与训练耦合 | 每个 `BertXXX.py` 自带 `forward(labels=...)` 分支 | 模型只做前向，损失与训练循环分离 |
| 数据加载写死 tsv | `Utils/load_datatsets.py` 只认 `sentence\tlabel` | `data/processors/` 支持多种原始格式并统一成 `DataInstance` |
| 结果难以复现 | `run_*.py` 里硬编码绝对路径 | 相对仓库根的配置项，`config.get_path("_config_sources")` 可追溯来源 |

---

## 2. 依赖迁移：`pytorch_pretrained_bert` → `transformers`

这是**必须做**的改动，原因是前者已停止维护：

| 方面 | `pytorch_pretrained_bert` | `transformers` |
|---|---|---|
| 维护状态 | 2019 年后停止更新（已被 `pytorch-transformers` → `transformers` 取代） | 活跃 |
| 新版 PyTorch 兼容 | 有 `torch.load` 与 CUDA 版本相关的兼容问题 | 正常 |
| 前向返回值 | `(all_encoded_layers, pooled_output)` 或 4 元组 | `(last_hidden_state, pooler_output)` |
| 输出层控制 | `output_all_encoded_layers=False` | `output_hidden_states=False` |
| 优化器 | `BertAdam`（自带 warmup） | 无对应类，改用 `torch.optim.AdamW` + `LambdaLR` |
| 分词器 | `BertTokenizer.from_pretrained(vocab_file)` | `BertTokenizer.from_pretrained(model_path_or_name)` |

对应改动点：

* `src/models/encoder.py` 显式取 `last_hidden_state[:, 0]` 作为句向量，
  **不用** `pooler_output`——池化头带一层随机初始化的 tanh 映射，
  对"通用句表示"没有收益（这是有意的行为差异，不是遗漏）；
* `src/training/cl_trainer.py::build_scheduler` 复刻了 `BertAdam` 的
  "线性 warmup + 线性衰减"行为；
* 权重初始化沿用 BERT 惯例：`normal_(0, 0.02)` + bias 置零
  （原项目通过 `self.apply(self.init_bert_weights)` 实现）。

**兼容性说明**：如果一定要用 `pytorch_pretrained_bert`，只需替换
`src/models/encoder.py` 与 `src/models/tokenization.py` 两个文件，
其余代码不受影响——这正是把编码器单独成模块的原因。

---

## 3. 保留了什么

原项目里有价值、被新项目直接继承的部分：

| 原项目 | 新位置 | 说明 |
|---|---|---|
| `Utils/utils.py::classifiction_metric` | `src/training/evaluate.py` | 指标计算思路（改用 sklearn 的 `zero_division=0`，并新增论文口径的 Avg F1） |
| `train_evalute.py` 的早停与"按 dev 最优保存" | `src/training/cl_trainer.py::CLTrainer.run` | 保留 `save_best_on` 与 `early_stop_patience` 两项配置 |
| `main.py` 的 `no_decay` 参数分组 | `src/training/cl_trainer.py::build_optimizer` | bias / LayerNorm 不做权重衰减 |
| `Utils/Classifier_utils.py` 的特征转换流程 | `data/dataset.py` + `src/models/tokenization.py` | 保留 `[CLS] ... [SEP]` 与 `input_mask`/`segment_ids` 三件套 |
| 多模型对照的思路 | `src/models/backbones/` | 见下 |

---

## 4. `src/models/backbones/` 是什么

原项目的七个模型目录（BertOrigin / BertCNN / BertLSTM / BertATT / BertRCNN /
BertCNNPlus / BertDPCNN）以及 `BertHAN`、`Models/`、`Processors/` 全部原样保留，
放在 `src/models/backbones/` 下作为**只读参考实现**。

**论文只用到 BERT 编码器 + MLP 投影头 + 全连接分类器**，也就是新结构里的
`encoder.py` + `projector.py` + `classifier.py`。保留 backbones 的目的：

1. 便于对照"改造范围到底有多大"；
2. 便于做消融实验（"把 [CLS] 换成 CNN 池化会不会更好"）；
3. 如果将来要做"论文方法的 GNN 版本"，这些池化结构可以直接复用。

它们**不被主流程引用**，可以整体删除而不影响任何脚本。

---

## 5. 原项目里被舍弃的部分及原因

| 舍弃内容 | 原因 |
|---|---|
| `multi_main.py` / `multi_main.py` 的多任务逻辑 | 论文是单任务四分类 |
| `Processors/MultiNewsProcessor.py` 等多句处理 | 论文输入是"原帖 + 回复"的单序列，不是多句对 |
| `Utils/load_multi_datasets.py` | 同上 |
| 9 个 `run_*.py`（SST-2 / IMDB / THUCNews / Yelp / AG News / DBPedia / Yahoo / CNews / Multi-CNews） | 论文只用 Twitter15/16；新项目用 `--dataset` 参数替代 |
| `docs/BHNet...md`、`docs/Results.md` | 原项目的实验记录，与论文无关（快照里仍可查到） |
| `tensorboardX` | 换成 `torch.utils.tensorboard`（`tensorboardX` 已停止维护）；本仓库暂未启用 TensorBoard 写入，配置项 `visualization.log_tensorboard` 保留 |

---

## 6. 一句话总结

原项目提供了"**怎么用 BERT 做单序列文本分类**"这一层的可靠基线；
新项目在此基础上补齐了论文需要的三件事：

1. **数据层**：从"tsv 单句"扩展到"原帖 + 嵌套回复 + 标签外挂"；
2. **LLM 层**：数据增强、Prompt 编排、输出校验、LoRA 微调、TIES 合并；
3. **对齐层**：CL 训练与 LLM 微调的联合调度（Algorithm 2）。

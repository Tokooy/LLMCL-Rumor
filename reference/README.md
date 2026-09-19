# 原开源项目快照（只读参考）

本目录是 `Bert-TextClassification-master` 的**完整原样快照**，
即论文代码改造的起点。**它不参与本仓库的任何运行流程**，请不要在此目录下开发。

## 为什么保留

1. **可追溯**：论文方法是在这份代码上改出来的，保留快照就能随时对照
   "哪些是原有的、哪些是为论文新增的"；
2. **可消融**：`BertCNN` / `BertLSTM` / `BertATT` / `BertRCNN` /
   `BertCNNPlus` / `BertDPCNN` / `BertHAN` 七个骨干可以拿来做
   "把 [CLS] 换成别的池化会不会更好"的实验；
3. **可迁移**：如果确实需要用回 `pytorch_pretrained_bert`，
   这里就是那个版本的参考实现。

## 为什么不能直接运行

原项目依赖 `pytorch_pretrained_bert`——该库自 2019 年后停止维护，
在新版 PyTorch/CUDA 上无法安装。迁移到 `transformers` 的完整说明见
[`../../docs/differences_from_baseline.md`](../../docs/differences_from_baseline.md)。

## 与主流程的对应关系

| 原项目文件 | 新项目对应位置 | 说明 |
|---|---|---|
| `BertOrigin/BertOrigin.py` | `src/models/encoder.py` + `src/models/classifier.py` | 论文实际使用的结构 |
| `Utils/utils.py::classifiction_metric` | `src/training/evaluate.py` | 指标计算（新增论文口径的 Avg F1） |
| `Utils/Classifier_utils.py` | `data/dataset.py` + `src/models/tokenization.py` | 特征转换流程 |
| `train_evalute.py` | `src/training/cl_trainer.py` | 训练循环（早停、按 dev 最优保存） |
| `main.py` | `scripts/train_cl.py` | 入口与优化器参数分组 |
| `BertCNN/`、`BertLSTM/` …… | `src/models/backbones/` | 其余骨干（论文未使用） |
| `run_*.py`（9 个） | 由 `--dataset` 参数替代 | 原项目每个数据集一个脚本 |
| `multi_main.py`、`Processors/` | 未使用 | 多句/多任务逻辑，论文用不到 |

## 目录里各文件是什么

| 路径 | 用途 |
|---|---|
| `BertOrigin/` | 最基础的 BERT 分类（`[CLS]` → 全连接） |
| `BertCNN/` `BertCNNPlus/` | 在 BERT 输出上加卷积池化 |
| `BertLSTM/` `BertRCNN/` `BertATT/` `BertHAN/` `BertDPCNN/` | 其它池化/注意力变体 |
| `Models/` | 上述变体用的公共层（Conv / LSTM / Highway / Linear / Embedding） |
| `Processors/` `Utils/load_multi_datasets.py` | 多句任务的数据处理（本仓库未使用） |
| `Utils/` | 通用的数据加载、指标、设备工具 |
| `main.py` `train_evalute.py` | 训练与评测主循环 |
| `run_*.py` | 各数据集入口 |
| `docs/` | 原项目的实验记录（与论文无关） |
| `Bert.md` | 预训练权重与词表的下载说明 |

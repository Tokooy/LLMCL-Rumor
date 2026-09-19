# 模型与训练流程设计

本文件描述实现的**工程细节**（张量形状、数据流、损失组合），
与论文的文字描述逐条对应。算法层面的逐行对照见 `docs/algorithms.md`。

---

## 1. 整体数据流

```
data/processed/<dataset>/train.jsonl           原样本（uid/string_value/replies/label/text）
data/processed/<dataset>/augmented_round<k>.jsonl  第 k 轮 LLM 增强样本
                    │
                    ▼
        PairDataset（按 original_uid 配对）
                    │  __getitem__ → {"uid","label","original","augmented[]","n_augmented"}
                    ▼
        collate_pairs
                    │  份数一致 → {"input_ids":[B,K,L], ...}
                    │  份数不一致 → [{"input_ids":[B_k,L]}, ...]
                    ▼
   ┌──────────────────────────────────────────────┐
   │ ContrastiveModel.forward                     │
   │   input_ids [B,L] ─► BERT ─► h [B,768]       │
   │   h ─► MLP 投影头 ─► z [B,128]（已 L2 归一化）│
   │   z ─► 分类器 ─► logits [B,C]                │
   └──────────────────────────────────────────────┘
                    │
                    ▼
        CombinedLoss = ce_weight·CE(logits, y)
                     + cl_weight·InfoNCE(z, z', mask)
```

**为什么同时返回 `logits` 与 `projection`**：论文的联合目标既要标签预测（式(4)）
又要对比特征学习（式(3)），一次前向同时拿到两者可以避免重复编码——
这在 BERT 上意味着**省掉一半的前向开销**。

---

## 2. 张量形状契约

| 张量 | 形状 | 说明 |
|---|---|---|
| `input_ids` / `attention_mask` / `token_type_ids` | `[B, L]` | `L = data.max_seq_length`（默认 128） |
| `h`（句向量） | `[B, 768]` | BERT `last_hidden_state[:, 0]` |
| `z`（投影） | `[B, d]`，`d = model.projector.output_size`（默认 128） | 已 L2 归一化 |
| `logits` | `[B, C]`，`C = 4` | NR/FR/TR/UR |
| `augmented` 编码 | `[B, K, L]` 或 `[B_k, L]` 列表 | K = `data.augmented_per_sample` 或历史轮次数 |
| 相似度矩阵 | `[B, B]` | `z @ z'.T / τ` |

---

## 3. 对比损失（论文式(3)）的实现细节

### 3.1 配对模式

论文原文：""在每个 batch 中，(x_i, x'_i) 被配对为正样本，而 (x_i, x'_j) 被视为负样本"。

实现为 `paired_infonce_loss(anchor, positive)`：

```
similarity = cos(anchor, positive) / τ          # [B, B]
loss = mean( CE(similarity, arange(B)) , CE(similarity.T, arange(B)) )
```

展开来看，`similarity[i, j]` 就是 `cos(z_i, z'_j)`：

* `similarity[i, i]` 是样本 i 的正样本对；
* `similarity[i, j] (j≠i)` 是论文所说的"负样本" `(x_i, x'_j)`；
* 对称项 `similarity.T` 让 `z'` 也当一次锚点——这等价于论文的"参与损失计算的
  样本包括原始样本及其增强样本，数量为 2B"。

### 3.2 负样本屏蔽（必须注意的坑）

当 `copies_per_sample > 1` 或多轮增强样本一起使用时，同一个原样本会有多个副本
（`z'_i^(1)`、`z'_i^(2)` …）。若不处理，`z_i` 会把 `z'_i^(2)` 当作**负样本**，
这显然不对——它们来自同一个原样本。

因此 `ContrastiveLoss` 用 `group_ids` 构造掩码 `mask[i, j] = (group_ids[i] != group_ids[j])`，
把"同一原样本的其它副本"屏蔽出 logsumexp。屏蔽用 `torch.where(mask, sim, -inf)`
而**不是** `masked_fill`，因为后者在某些 PyTorch 版本下会产生 `0 * (-inf) = NaN` 梯度
（`tests/test_losses.py::test_gradients_are_finite_with_masking` 是这个问题的回归测试）。

### 3.3 多样性损失的取舍

`supervised_contrastive_loss`（SupCon）作为消融保留。它与论文描述的差别在于：
SupCon 会把**同标签的不同样本**也当作正样本，而论文明确把它们当作负样本。
之所以两种都实现，是为了回答审稿人常见的问题——"为什么不做有监督对比"。

---

## 4. 联合损失

```
loss = ce_weight · CrossEntropy(logits, y) + cl_weight · InfoNCE(z, z')
```

* `training.cl.ce_weight` / `training.cl.cl_weight` 默认均为 1.0；
* `training.cl.joint_objective=false` 时退化为纯分类（消融对照）；
* 评测阶段不传增强样本，自动只算交叉熵。

---

## 5. 优化器与调度

`build_optimizer` 遵循 BERT 微调的既有惯例（与原开源项目 `main.py` 的
`no_decay` 列表一致）：

* `bias` / `LayerNorm.weight` 不参与权重衰减；
* 默认 `layer_lr_decay = 1.0`（所有层同一学习率），与论文"所有实验的关键超参数
  （学习率、训练周期和批次大小）均保持一致"的表述相符；
* 需要时可通过 `layer_lr_decay < 1` 打开分层学习率衰减（llrd）做消融。

`build_scheduler` 为线性 warmup + 线性衰减，等价于原项目 `BertAdam` 的
`warmup` 行为。

---

## 6. 数据增强后的训练集重建

论文 Algorithm 2 里"每 T 个 epoch 增强一次"，增强后训练集变大。
`JointAlignmentTrainer.rebuild_train_loader` 的做法是：

1. 取出当前 `PairDataset` 的**原样本**（`dataset.samples` 的第一项）；
2. 用"原样本 + 累计的全部增强样本"重建 `PairDataset`，
   `augmented_round=0` 表示使用**所有历史轮次**的增强样本；
3. 重建 DataLoader，并重置优化器与调度器（重新 warmup）。

为什么重建而不是原地追加：`PairDataset` 在构造时就把"原样本 → 其增强样本列表"
的配对关系固定下来了，原地追加无法让已构造的样本看到新增强样本。

---

## 7. 评估口径

| 指标 | 实现 |
|---|---|
| ACC | `sklearn.metrics.accuracy_score` |
| 逐类 F1 | `classification_report` 的 `f1-score` |
| **Avg F1** | 四类 F1 的**算术平均** |
| 精确率/召回率 | 逐类 + 宏平均 |

Avg F1 的口径是从论文数值反推确认的：Table 3 的 Proposed-1 四个 F1 为
`90.68 / 69.42 / 75.73 / 74.12`，算术平均 `77.4875` 与表中 `77.48` 一致；
若是加权平均（按各类样本数）则不会是这个值。

---

## 8. 显存与规模

以论文配置（4×RTX 4090 24GB、Twitter15）为例的粗略估算：

| 组件 | 显存 | 说明 |
|---|---|---|
| BERT-base 训练 | ~4 GB | `L=128`、`B=32`、bf16 混合精度 |
| Qwen2.5-7B（bf16）推理 | ~15 GB | `device_map="auto"` 自动切分 |
| Qwen2.5-13B（bf16）推理 | ~26 GB | 单卡需 `load_in_8bit: true` |
| LoRA 适配器 + 任务向量 | < 1 GB | `r=8`，只含 `lora_A/lora_B` |

CL 训练与 LLM 推理**不会同时**占用显存：增强阶段暂停 CL 训练（CL 模型在显存里
但不做前向），`CLTrainer` 在每个微调周期后重建优化器时也不会新增显存。
若显存紧张，优先降低 `llm.augmentation.batch_size`。

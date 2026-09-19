# 复现检查清单

按顺序执行即可复现论文 Table 3–8。每一步都标注了**预期结果**与
**常见问题**，便于定位偏差。

---

## 步骤 0：环境

```bash
conda create -n llmcl python=3.10 -y && conda activate llmcl
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

| 检查项 | 预期 |
|---|---|
| `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| `nvidia-smi` 显存 | ≥ 24 GB（13B 实验需要） |
| `pytest tests -q` | 全部通过（无 torch 环境下 torch 用例显示 skipped） |
| `python scripts/check_syntax.py` | 所有 `.py` 文件通过 |
| `python scripts/check_static.py` | 0 条提示 |
| `python scripts/check_tracked.py` | 无被 gitignore 误伤的源码、无未追踪文件 |

> 第三项自检是针对"源码被 `.gitignore` 静默吞掉"的防呆——本项目开发期间就踩过一次
> （`models/` 模式连 `src/models/` 一起忽略了）。改完 `.gitignore` 或新增目录后
> 建议跑一次，确认远端不会缺文件。

---

## 步骤 1：数据准备

把 Twitter15/16 原始文件按 `data/README.md` 的布局放进 `data/raw/<dataset>/`，然后：

```bash
python scripts/prepare_data.py --config configs/base.yaml --dataset twitter15
python scripts/prepare_data.py --config configs/base.yaml --dataset twitter16
```

| 检查项 | 预期 | 论文 |
|---|---|---|
| 总样本数（Twitter15） | 1490 | Table 1：1490 |
| 总样本数（Twitter16） | 818 | Table 1：818 |
| 四类分布（Twitter15） | 374 / 370 / 372 / 374 | Table 1 |
| 四类分布（Twitter16） | 205 / 205 / 207 / 201 | Table 1 |
| 划分比例 | 约 70% / 10% / 20% | §4.1 |

**常见问题**

* 样本数少于论文值 → 原推文被删除。脚本会告警，请在论文中说明实际样本量；
* 回复数量为 0 → 数据只有传播树结构没有回复正文。
  此时把 `data.text_mode` 改成 `source_only`，或在论文中说明。

**先用演示数据验证链路（不需要真实数据集）**

```bash
python scripts/prepare_data.py --config configs/base.yaml --dataset demo
```

---

## 步骤 2：LLM 数据增强

```bash
# 第 1 轮
python scripts/augment_data.py --config configs/experiments/proposed-3.yaml \
    --dataset twitter15 --round 1

# 第 2、3 轮
python scripts/augment_data.py --config configs/experiments/proposed-3.yaml \
    --dataset twitter15 --round 2
python scripts/augment_data.py --config configs/experiments/proposed-3.yaml \
    --dataset twitter15 --round 3
```

| 检查项 | 预期 |
|---|---|
| `data/processed/twitter15/augmented_round{1,2,3}.jsonl` | 行数 = 训练集样本数 × `copies_per_sample` |
| 成功率（日志里的 `success_rate`） | ≥ 0.95，低于 0.9 会告警 |
| 词级重合度均值 | 越低越好（表示改写幅度大）；> 0.75 会记入 `quality.warnings` |
| 每行的 `original_uid` | 必须能在 `train.jsonl` 里找到对应样本 |
| 每行的 `augment_round` | 与 `--round` 一致 |

**常见问题**

* 成功率低 → 看 `quality.problems`：多为 JSON 解析失败，
  可降低 `llm.generation.temperature` 或提高 `llm.augmentation.max_retries`；
* 重复运行很慢 → 缓存默认开启（`llm.augmentation.cache_dir`），
  第二次运行会直接命中缓存（日志里的 `cached` 会等于总数）；
* 想快速验证流程 → `--backend demo` 用规则伪增强，不需要 GPU。

---

## 步骤 3：Proposed-1 / 2 / 3（M=0，无微调）

```bash
python scripts/train_cl.py --config configs/experiments/proposed-1.yaml --dataset twitter15
python scripts/train_cl.py --config configs/experiments/proposed-2.yaml --dataset twitter15
python scripts/train_cl.py --config configs/experiments/proposed-3.yaml --dataset twitter15
```

| 检查项 | 论文 Table 5（Twitter15） | 论文 Table 6（Twitter16） |
|---|---|---|
| Proposed-1 ACC | 78.11% | 79.63% |
| Proposed-2 ACC | 78.45% | 80.86% |
| Proposed-3 ACC | 79.12% | 83.33% |
| 趋势 | ACC 随增强轮次**缓慢上升** | 同左 |

日志里会打印与 Table 3–8 同构的表格，结果 JSON 保存在
`outputs/results/proposed-N_twitter15_cl.json`。

---

## 步骤 4：Proposed-4 / 5（M>0，含微调与 TIES 合并）

```bash
python scripts/joint_align.py --config configs/experiments/proposed-4.yaml --dataset twitter15
python scripts/joint_align.py --config configs/experiments/proposed-5.yaml --dataset twitter15
```

| 检查项 | 论文 Table 7（Twitter15） | 论文 Table 8（Twitter16） |
|---|---|---|
| Proposed-4 ACC | 80.13% | 85.24% |
| Proposed-5 ACC | 80.81% | 85.31% |
| 趋势 | ACC 随微调轮次上升，但**不超过 Proposed-6** | 同左 |

需要重点核查的中间量（都在日志与 `outputs/checkpoints/.../alignment_state.json` 里）：

| 中间量 | 含义 | 期望 |
|---|---|---|
| `lambda_current` | 式(8) 平滑后的 CL 性能 | 稳定在 dev 准确率附近，不剧烈震荡 |
| `omega` | 式(9) 得到的本轮权重 | 随 λ 上升而上升，落在 `[0.05, 0.95]` |
| `merges[].report.sparsity_after_trim` | Algorithm 1 修剪后的稀疏度 | 接近 `trim_percent/100` |
| `merges[].report.merge_rate` | 符号一致被保留的比例 | 0.3~0.8 之间；过低说明任务向量间符号冲突严重 |
| `merges[].scaling` | 写回模型时的缩放 | `(1-α)·λ_m + α·λ_{m-1}`，α=0.5 时≈两者均值 |

**常见问题**

* `omega` 长期贴在下界 0.05 → CL 性能很差，先检查步骤 3 是否达标；
* `merge_rate` 极低 → 多轮任务向量符号冲突严重，可降低 `trim_percent` 观察变化；
* API 后端跑 Proposed-4/5 → 日志会提示"跳过微调"，此时等价于 Proposed-3，请注意区分。

---

## 步骤 5：Proposed-6（13B）

```bash
python scripts/joint_align.py --config configs/experiments/proposed-6.yaml --dataset twitter15
```

| 检查项 | 论文 Table 3（Twitter15） | 论文 Table 4（Twitter16） |
|---|---|---|
| Proposed-6 ACC | 81.81% | 84.56% |
| 与 Proposed-5 的关系 | Twitter15 上更高 | Twitter16 上持平或略低（论文 84.56 vs 85.31） |

注意：Proposed-6 的微调轮次是**推定值**（见 `docs/implementation_notes.md` 第 8 节）。
若只想复现"13B + 3 轮增强"这一确定结论，
把配置里的 `max_finetune_rounds` 设为 0。

---

## 步骤 6：特征分布图（Fg.7–Fg.10）

```bash
python scripts/evaluate.py --config configs/experiments/proposed-1.yaml --dataset twitter15 \
    --checkpoint outputs/checkpoints/proposed-1_twitter15/best.pt --name Proposed-1
# 对 Proposed-2 / 3 / 4 / 5 重复
```

| 检查项 | 论文 |
|---|---|
| 输出路径 | `outputs/figures/tsne_Proposed-N_twitter15_test.png` |
| Fg.7 趋势 | 随增强轮次增加，四类边界更清晰 |
| Fg.9 趋势 | 微调后 FR 类聚集增强，NR 与 UR 出现交叉 |

四类配色是固定的（NR 蓝 / FR 橙 / TR 绿 / UR 红），
跨图对比时颜色一致，便于与论文图直接比对。

---

## 步骤 7：结果汇总

```bash
# 汇总所有结果 JSON（论文 Table 3–8 的数值来源）
ls outputs/results/
```

每个 JSON 含：`acc`、`avg_f1`、`per_class`（逐类 P/R/F1/support）、
`macro_f1`、`weighted_f1`，以及对齐实验的 `alignment_state` 与 `merger` 描述。

---

## 复现偏差排查顺序

若数值与论文差距较大，建议按此顺序排查：

1. **数据**：样本量与四类分布是否与 Table 1 一致（步骤 1）；
2. **增强质量**：成功率与词级重合度分布（步骤 2）；
3. **CL 单独效果**：先用 `--augment-round 0` 跑纯分类，看是否达到 BERT 基线的合理区间；
4. **损失权重**：`training.cl.ce_weight` / `cl_weight` 是否为 1.0；
5. **温度**：`model.contrastive.temperature` 是否为 0.07；
6. **配对模式**：确认是 `paired` 而非 `supervised`；
7. **λ 口径**：`training.alignment.lambda_source` 是否与预期一致；
8. **随机性**：`seed` 是否固定、`deterministic` 是否按需开启。

> 论文没有给出代码，也没有公开超参细节（除学习率/训练周期/批次大小
> "均保持一致"这一句）。因此**复制论文的绝对数值不是复现的唯一目标**——
> 更重要的是复现出论文的**趋势结论**：
> ① 增强轮次增加 → 性能缓慢上升；② 微调轮次增加 → 性能上升但不超过更大的基座；
> ③ Acc/Avg F1 优于 BiGCN/BiMGCL 基线。

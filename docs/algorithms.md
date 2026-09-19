# 算法逐行对照（Algorithm 1 / Algorithm 2）

本文件把论文里的两段伪代码逐行映射到代码，便于审阅"实现是否忠实"。

---

## 一、Algorithm 1 `Merge(τ1, …, τm, ω1, …, ωm, q)`

### 论文伪代码（保留原样）

```
Algorithm 1  Merge( , , , , , , )
Input:  , , , , , ,
Output:
foreach  do
    ;
    ;
    ;
    forall to do
        ;
        ;
        ;
    end for
end for
return ;
```

> 说明：论文 docx 里的公式与下标以 OLE 对象嵌入，纯文本抽取后只剩分号占位符
> （见 `docs/implementation_notes.md` 第 5 节的说明）。因此下面的对照表以
> **论文正文对三个步骤的文字描述**（"修剪（Trim）/ 选举符号（Elect Sign）/
> 基于权重的不相交合并（Weight-based Disjoint Merge）"）以及它明确标注的
> 算法原型文献[35] TIES-Merging 为准。

### 逐步对照

| 论文步骤 | 论文文字描述 | 代码 | 说明 |
|---|---|---|---|
| 修剪 Trim | "为了去除冗余参数，在中，magnitude 的前 q% 被保留，其余被设置为 0（见第 3 行）" | `src/llm/ties_merge.py::trim_task_vector` | 阈值取**全局**分位数（所有参数一起算），与 TIES 原文一致；`q=0` 表示不修剪 |
| 分解 | "进一步被分解为 magnitute 和 sign（见第 4 行）" | `elect_sign` 内部用 `torch.sign` 与算术值 | 实现上不显式拆分，而是"符号比较 + 保留幅值"，数学等价 |
| 选举符号 Elect Sign | "合并 τ1 和 τ2 的先决条件是解决不同向量之间的符号冲突问题（见第 8 行）"；式(6) 给出第 e 个 entry 的计算 | `src/llm/ties_merge.py::elect_sign` | 按**加权**符号和 `sign(Σ ω_m·τ_m[e])`；权重全 1 时退化为 TIES 原文的"支持强度最大者胜" |
| 不相交合并 Disjoint Merge | "对于 τ^m 中的第 e 个参数，我们仅保留来自模型且符号与聚合后选定符号一致的参数值" | `src/llm/ties_merge.py::disjoint_merge` | 符号一致者加权平均；符号选举结果为 0 的 entry 输出 0 |
| 三合一 | — | `src/llm/ties_merge.py::merge_task_vectors` | 依次调用上面三步，并回填 `MergeReport` |
| 最终模型 | "合并后的 LLM 被表示为 … 其中为合并超参数" | `src/llm/ties_merge.py::resolve_scaling` + `hf_backend.apply_task_vector` | `θ = θ_0 + scaling·τ_merged` |

### 关键代码（节选）

```python
# 第 1 步：Trim
threshold = torch.quantile(magnitudes, trim_percent / 100.0)
trimmed[name] = torch.where(value.abs() >= threshold, value, torch.zeros_like(value))

# 第 2 步：Elect Sign（论文式(6)）
accumulator += vector[name] * weight
signs[name] = torch.sign(accumulator)

# 第 3 步：Disjoint Merge
agree = (torch.sign(value) == sign) & (sign != 0) & (value != 0)
merged[name] = Σ(agree * value * weight) / (Σ(agree * weight) + eps)
```

### 与文献[35] TIES-Merging 的差异

| 方面 | TIES-Merging 原文 | 论文 + 本实现 |
|---|---|---|
| 符号选举 | 取绝对值和最大的方向 | 论文式(6) 的**加权**符号和，权重 ω 由 CL 决定 |
| 权重 ω 来源 | 无（所有任务等权） | 论文式(9)，由 CL 性能指标 λ 决定 |
| 合并规模 | 多个任务模型 | 同一任务的多个**微调轮次**（Algorithm 2 的递归结构） |

---

## 二、Algorithm 2 `Alignment(x, D, θ, τ, ω, λ, M, N, q, m, n)`

### 论文伪代码（保留原样）

```
Algorithm 2  Alignment ( , , , , , , , , ,m,n)
Input:  , , , , , , , , ,N,q,m,n
Output:
if
  foreach  do
      ;
      ;
  end for
  ;
  ; //LLM合并
  ;
  //数据增强及CL运行流程
  if
      Alignment（,,,,,,,,,m,q）;
      m++;
  else
     return ;
end if
```

### 论文文字补充（用于确定语义）

* "在联合训练中，每 T 个 epoch 执行一次数据增强操作，即在总计 N 个 epoch 的
  训练周期内将实现 ⌊N/T⌋ 次数据增强。"
* "动量更新策略被用来评估 LLM 的增强效果（见 3-4 行）" + 式(8)
* "通过微调和调用算法 1（见第 6 行），LLM 被微调。微调后通过算法 1 实现合并，
  得到新的 LLM（见第 7 行）。"
* "此外，M 决定了 LLM 的最大微调轮次（见 10-15 行）。当 m 超过 M 时，
  算法将停止微调。"

### 逐步对照

| 论文步骤 | 代码位置 | 说明 |
|---|---|---|
| 第 3-4 行：动量更新 λ | `JointAlignmentTrainer.update_lambda` → `TiesMerger.update_lambda` | 式(8) `λ_m = β·f_m + (1-β)·λ_{m-1}`；`f_m` 为 CL 性能观测值 |
| 第 5 行：由 λ 得权重 ω | `compute_omega` | 式(9)：`ω_m = clip(λ_m, ω_min, ω_max)` |
| 第 6 行：微调 LLM | `JointAlignmentTrainer.finetune_and_merge` → `src.llm.lora.finetune_and_export` | 自举式微调：用 LLM 自己产出的增强数据构造训练对 |
| 第 7 行：调用 Algorithm 1 合并 | `JointAlignmentTrainer.merge_and_apply` → `TiesMerger.merge` | 见上文 Algorithm 1 对照 |
| 第 8-9 行：数据增强 | `JointAlignmentTrainer.maybe_augment` → `Augmentor.augment` | 每 T 个 epoch 一次；结果落盘并累积进增强池 |
| "数据增强及 CL 运行流程" | `JointAlignmentTrainer.on_epoch_end` 全流程 | 增强 → 重建训练集 → 继续 CL 训练 |
| 第 10-15 行：m > M 停止 | `max_finetune_rounds` + `stop_when_max_reached` | 达到 M 后可选择停止训练 |
| 递归调用 Alignment | `CLTrainer.run(on_epoch_end=...)` 的逐 epoch 循环 | 实现为循环而非递归：递归在这里没有任何语义收益，反而会让栈深度随 N 增长，并丢失优化器状态 |

### 关于"递归 → 循环"的说明

论文伪代码里 `Alignment(...)` 递归调用自身（第 125 行）。但从上下文看，
它表达的是"**不断重复**增强→训练→微调→合并的周期"。
实现为循环有三个好处：

1. **优化器状态可以跨周期保留**（递归会让每次调用重建局部状态）；
2. 不会因 N 增大而加深调用栈；
3. 更容易在任意 epoch 中断并保存 checkpoint。

语义上完全等价：循环体 = 递归体的单次执行。

---

## 三、式(6)(7)(8)(9) 与代码的对应

| 公式 | 内容 | 代码 |
|---|---|---|
| 式(6) | 符号选举的加权求和 | `elect_sign` |
| 式(7) | `θ_m^n = (1-α)·θ̃_m^n + α·θ_m^{n-1}` | `resolve_scaling` 的推导（见下） |
| 式(8) | `λ_m = β·f_m + (1-β)·λ_{m-1}` | `TiesMerger.update_lambda` |
| 式(9) | 微调权重 `ω_m` 由 λ 决定 | `compute_omega` |

### 式(7) 的推导（`resolve_scaling` 的 docstring 摘要）

设 `τ` 为本轮合并出的任务向量（由全部历史任务向量经 Algorithm 1 得到），
把两轮参数都写成"基座 + 缩放后的任务向量"：

```
θ̃       = θ_0 + λ_m     · τ
θ^{n-1} = θ_0 + λ_{m-1} · τ
```

代入式(7)：

```
θ_m^n = (1-α)·(θ_0 + λ_m·τ) + α·(θ_0 + λ_{m-1}·τ)
      = θ_0 + [(1-α)·λ_m + α·λ_{m-1}]·τ
```

因此 `scaling = (1-α)·λ_m + α·λ_{m-1}`。这样只需维护 λ 与基座 θ_0，
**不需要保存上一轮完整参数**（13B 模型下这能省掉数十 GB 显存）。

`alpha=None` 时退化为标准 TIES-Merging 的 `θ = θ_0 + λ·τ`。
推导原文也写进了 `TiesMerger.report_note()`，并随结果 JSON 一起落盘，
便于论文复核。

# 数据目录说明

本目录只放**数据**与**读取数据的代码**，不放模型、不放训练脚本。

```
data/
├── raw/                    原始数据（不入库，需自行下载放置）
│   ├── twitter15/
│   └── twitter16/
├── processed/              预处理 / 增强产物（不入库）
├── samples/                仓库自带的小样例（入库，用于跑通流程）
│   └── demo_twitter15.jsonl
├── processors/             原始格式 -> 统一格式的转换代码
│   ├── twitter_rumor.py
│   └── reply_flatten.py
└── dataset.py              PyTorch Dataset
```

---

## 1. 原始数据从哪来

论文使用 **Twitter15** 与 **Twitter16** 两个公开谣言检测数据集
（Ma et al., ACL 2017，文献[20]），类别分布见论文 Table 1：

| statistic | Twitter15 | Twitter16 |
|---|---|---|
| # of source tweets | 1490 | 818 |
| # of non-rumors | 374 | 205 |
| # of false-rumors | 370 | 205 |
| # of true-rumors | 372 | 207 |
| # of unverified rumors | 374 | 201 |

论文另称"利用爬虫技术从对应社交平台上采集了原始帖子及其所有关联回复"。
本仓库**不包含**这两个数据集（受 Twitter/X 服务条款限制，且原始推文大量已被删除）。
请把下载到的数据按下面任一布局放进 `data/raw/<dataset>/`：

### 布局 A：官方发布格式（推荐，优先使用官方划分）

```
data/raw/twitter15/
├── label.txt              # 每行：<tweet_id>\t<label>，label ∈ {true, false, unverified, non-rumor}
├── tree.txt               # 每行：<source_id>\t<parent_id>\t<child_id>（传播树，可选）
└── source_tweets.txt      # 每行：<tweet_id>\t<json>（含 text 字段）
```

解析器同时兼容旧版的 `Twitter15_label.txt` / `Twitter15_tree.txt` / `Twitter15_source_tweets.txt`
命名（`twitter_rumor.py` 里做了文件名候选匹配）。

### 布局 B：已整理好的树结构

```
data/raw/twitter15/
└── twitter15.json         # [{ "uid": "...", "string_value": "...", "label": "...", "replies": [...] }, ...]
```

若只有传播树而没有回复正文，数据处理器会把 `replies` 置空并在日志中告警——
此时建议把 `data.text_mode` 设为 `source_only`，否则回复上下文的增强将无内容可做。

---

## 2. 统一数据实例格式

预处理后的每行是一个 JSON 对象，字段与论文 Fg.2 对齐：

```json
{
  "uid": "699262866205298688",
  "string_value": "an open letter to trump voters from his top strategist-turned-defector",
  "label": "FR",
  "label_id": 1,
  "replies": [
    {
      "uid": "699263000000000001",
      "string_value": "They love him",
      "replies": []
    }
  ],
  "split": "train",
  "source": "twitter15",
  "text": "an open letter to trump voters ... [SEP] They love him ..."
}
```

| 字段 | 来源 | 说明 |
|---|---|---|
| `uid` | 原始数据 | 样本唯一标识；增强样本沿用**同一个 uid**，便于配对 |
| `string_value` | 原始数据 | 原帖正文（**LLM 只改写这个字段**） |
| `replies` | 原始数据 | 关联回复，嵌套结构与输入完全一致（论文"输出格式一致"要求） |
| `label` / `label_id` | **官方 `label.txt`** | 4 类；`label_id` 顺序由 `label_list` 决定（NR/FR/TR/UR → 0/1/2/3） |
| `split` | 划分 | `train` / `dev` / `test` |
| `text` | 派生 | 编码器实际输入（由 `text_mode` 与 `reply_flatten.py` 生成） |

> **重要**：论文 Fg.2 的数据结构里**没有标签字段**。标签必须外挂自数据集官方的
> `label.txt`。LLM 增强时标签不进入 Prompt，也不对标签做任何生成或改写——
> 虚假信息检测是有监督分类任务，标签不能由 LLM 编造。详见
> `docs/implementation_notes.md` 第 1 节。

---

## 3. 增强数据格式

`scripts/augment_data.py` 产出 `data/processed/<dataset>/augmented_round<k>.jsonl`，
每行在原字段基础上追加：

```json
{
  "uid": "699262866205298688",
  "string_value": "A public letter to Trump supporters from his former strategist-turned-critic",
  "replies": [ ... ],
  "label": "FR",
  "label_id": 1,
  "text": "...",
  "augmented": true,
  "augment_round": 1,
  "augment_model": "Qwen/Qwen2.5-7B-Instruct",
  "augment_prompt_hash": "3f1a...",
  "original_uid": "699262866205298688",
  "quality": {
    "char_ngram_overlap": 0.41,
    "structure_ok": true,
    "semantic_ok": true
  }
}
```

`original_uid` 与原始样本的 `uid` 相同，`data/dataset.py` 正是靠这个字段
把"原样本 ↔ 其增强样本"配成对比学习的正样本对。

---

## 4. 样例数据 `samples/demo_twitter15.jsonl`

仓库自带 10 条**手工构造**的样例（不是真实推文，仅用于验证数据流），
使用时通过 `--dataset demo` 指向它：

```bash
python scripts/prepare_data.py --config configs/base.yaml --dataset demo
```

`prepare_data.py` 识别到 `demo` 时会跳过 `raw/` 读取、直接把样例切成
train/dev/test 三份写入 `data/processed/demo/`，因此可以在没有真实数据集、
甚至没有 LLM 的情况下检查整条代码链路是否接通。

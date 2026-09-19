# coding=utf-8
"""Twitter15 / Twitter16 原始数据解析（论文 §4.1 的数据集）。

支持的原始布局
--------------
**布局 A：官方发布格式**（推荐，可用官方划分）::

    data/raw/twitter15/
    ├── label.txt            # 每行 <tweet_id>\t<label>
    ├── tree.txt             # 每行 <source_id>\t<parent_id>\t<child_id>
    ├── source_tweets.txt    # 每行 <tweet_id>\t<python 字面量 dict 或 JSON>
    └── split.txt            # 可选，每行 <tweet_id>\t<train|dev|test>

文件名同时兼容旧命名 ``Twitter15_label.txt`` / ``Twitter15_tree.txt`` /
``Twitter15_source_tweets.txt``，以及带下划线的变体。

**布局 B：已整理好的 JSON**::

    data/raw/twitter15/twitter15.json   # [{uid, string_value, label, replies}, ...]

**布局 C：只有 id → text 的文件**::

    data/raw/twitter15/tweets.txt       # 每行 <tweet_id>\t<text>

解析结果统一为 :class:`data.processors.data_model.DataInstance`。
若数据集中没有回复正文（只有传播树结构而无推文内容），``replies`` 会是空列表，
此时会打日志告警，建议把 ``data.text_mode`` 调成 ``source_only``。
"""

from __future__ import annotations

import ast
import json
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data_model import (
    LABELS,
    LABEL_TO_ID,
    DataInstance,
    Reply,
    normalize_label,
)
from .reply_flatten import build_encoder_text

__all__ = [
    "load_raw_dataset",
    "parse_labels",
    "parse_source_tweets",
    "parse_trees",
    "parse_instances_json",
    "assign_splits",
    "DATASET_DEFAULT_SPLIT",
]

# 论文 §4.1：数据集划分 70% 训练 / 10% 验证 / 20% 测试
DATASET_DEFAULT_SPLIT: Dict[str, float] = {"train": 0.7, "dev": 0.1, "test": 0.2}

_LABEL_FILE_CANDIDATES = ("label.txt", "{name}_label.txt", "{name}_labels.txt")
_TREE_FILE_CANDIDATES = ("tree.txt", "{name}_tree.txt")
_SOURCE_FILE_CANDIDATES = (
    "source_tweets.txt",
    "{name}_source_tweets.txt",
    "tweets.txt",
    "{name}_tweets.txt",
)
_SPLIT_FILE_CANDIDATES = ("split.txt", "{name}_split.txt", "splits.txt")
_JSON_CANDIDATES = ("{name}.json", "dataset.json", "data.json")

_MISSING = object()


# ---------------------------------------------------------------------- #
# 基础 IO
# ---------------------------------------------------------------------- #
def _first_existing(directory: str, candidates: Sequence[str], name: str) -> Optional[str]:
    """在目录里按候选文件名找第一个存在的文件。"""
    for pattern in candidates:
        path = os.path.join(directory, pattern.format(name=name))
        if os.path.isfile(path):
            return path
    return None


def _read_lines(path: str) -> List[str]:
    """读文本文件，自动尝试 utf-8 / utf-8-sig / latin-1 编码。"""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read().splitlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "utf-8", b"", 0, 1, f"无法用 utf-8/utf-8-sig/latin-1 解码 {path}"
    )


def _parse_mapping_payload(payload: str) -> Any:
    """把"python 字面量 dict"或"JSON 字符串"解析成对象。

    Twitter15 官方的 ``source_tweets.txt`` 用的是 python 字面量
    （单引号、``None``/``True`` 等），标准 ``json.loads`` 会失败，
    因此先试 JSON 再退回 :func:`ast.literal_eval`。
    """
    text = payload.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _extract_text(payload: Any) -> str:
    """从各种 tweet 结构中取出正文文本。"""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, Mapping):
        return ""

    for key in ("text", "full_text", "string_value", "content", "tweet", "body"):
        value = payload.get(key, _MISSING)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # 有些镜像把正文放在 retweeted_status / quoted_status 里
    for key in ("retweeted_status", "quoted_status", "extended_tweet"):
        nested = payload.get(key)
        if nested is not None:
            text = _extract_text(nested)
            if text:
                return text

    # 最后尝试 extended_tweet.full_text
    extended = payload.get("extended_tweet")
    if isinstance(extended, Mapping):
        value = extended.get("full_text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ---------------------------------------------------------------------- #
# 各文件解析
# ---------------------------------------------------------------------- #
def parse_labels(path: str) -> Dict[str, str]:
    """解析 ``label.txt``：``<tweet_id>\\t<label>``，返回 ``{tweet_id: NR/FR/TR/UR}``。

    Raises:
        ValueError: 文件中出现无法识别的标签（附行号，便于定位脏数据）。
    """
    labels: Dict[str, str] = {}
    for lineno, line in enumerate(_read_lines(path), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split("\t") if "\t" in text else text.split()
        if len(parts) < 2:
            continue
        uid, raw_label = parts[0], parts[1]
        try:
            labels[uid] = normalize_label(raw_label)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno} {exc}") from exc
    return labels


def parse_source_tweets(path: str) -> Dict[str, str]:
    """解析 ``source_tweets.txt`` / ``tweets.txt``，返回 ``{tweet_id: text}``。

    支持两种行格式：

    * ``<tweet_id>\\t<json 或 python 字面量 dict>``
    * ``<tweet_id>\\t<纯文本>``
    """
    tweets: Dict[str, str] = {}
    for line in _read_lines(path):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "\t" not in text:
            continue
        uid, payload = text.split("\t", 1)
        uid = uid.strip()
        payload = payload.strip()
        if not uid or not payload:
            continue
        parsed = _parse_mapping_payload(payload)
        content = _extract_text(parsed) if parsed is not None else payload
        if content:
            tweets[uid] = content
    return tweets


def parse_trees(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """解析 ``tree.txt``：``<source_id>\\t<parent_id>\\t<child_id>``。

    Returns:
        ``{source_id: [(parent_id, child_id), ...]}``，保持文件中的先后顺序
        （官方文件按时间顺序排列，这个顺序对 BFS 展开有意义）。
    """
    trees: Dict[str, List[Tuple[str, str]]] = {}
    for line in _read_lines(path):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split("\t") if "\t" in text else text.split()
        if len(parts) < 3:
            continue
        source_id, parent_id, child_id = parts[0].strip(), parts[1].strip(), parts[2].strip()
        trees.setdefault(source_id, []).append((parent_id, child_id))
    return trees


def parse_instances_json(path: str) -> List[DataInstance]:
    """解析布局 B 的整份 JSON（list 或 ``{"data": [...]}``）。"""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, Mapping):
        records = payload.get("data") or payload.get("instances") or []
    elif isinstance(payload, list):
        records = payload
    else:
        raise TypeError(f"{path} 的顶层结构必须是 list 或含 data 字段的 mapping")

    instances: List[DataInstance] = []
    for index, record in enumerate(records):
        if isinstance(record, Mapping) and "label" in record:
            instances.append(DataInstance.from_record(record))
            continue
        # 容忍不带 label 的裸结构：标签必须另配，这里直接报错提示
        raise KeyError(
            f"{path} 第 {index} 条记录缺少 label 字段；"
            "若使用论文 Fg.2 的裸结构，请同时提供 label.txt（见 data/README.md）"
        )
    return instances


def _parse_extra_splits(path: str) -> Dict[str, str]:
    """解析可选的 ``split.txt``：``<tweet_id>\\t<train|dev|test>``。"""
    splits: Dict[str, str] = {}
    alias = {"train": "train", "training": "train", "dev": "dev",
             "valid": "dev", "validation": "dev", "test": "test", "testing": "test"}
    for line in _read_lines(path):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split("\t") if "\t" in text else text.split()
        if len(parts) < 2:
            continue
        key = alias.get(parts[1].strip().lower())
        if key:
            splits[parts[0].strip()] = key
    return splits


# ---------------------------------------------------------------------- #
# 回复树重建
# ---------------------------------------------------------------------- #
def _build_reply_forest(
    edges: Sequence[Tuple[str, str]],
    id_to_text: Mapping[str, str],
    source_id: str,
) -> List[Reply]:
    """由 ``(parent, child)`` 边列表重建回复森林。

    官方树的根节点就是源推文本身，因此第一条 ``parent == source_id`` 的边
    才是一级回复。这里以源推文为根做 BFS 展开，不在树中的孤立边会被忽略。
    """
    children_of: Dict[str, List[str]] = {}
    for parent, child in edges:
        children_of.setdefault(parent, []).append(child)

    seen: set = {source_id}
    roots: List[Reply] = []

    def build(node_id: str, depth: int) -> Optional[Reply]:
        # 深度保护：脏数据里可能存在超长链
        if depth > 50:
            return None
        kids: List[Reply] = []
        for child_id in children_of.get(node_id, []):
            if child_id in seen:
                continue
            seen.add(child_id)
            child = build(child_id, depth + 1)
            if child is not None:
                kids.append(child)
        return Reply(
            uid=node_id,
            string_value=id_to_text.get(node_id, ""),
            replies=kids,
        )

    for child_id in children_of.get(source_id, []):
        if child_id in seen:
            continue
        seen.add(child_id)
        node = build(child_id, depth=1)
        if node is not None:
            roots.append(node)
    return roots


# ---------------------------------------------------------------------- #
# 划分
# ---------------------------------------------------------------------- #
def assign_splits(
    instances: Sequence[DataInstance],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    official: Optional[Mapping[str, str]] = None,
) -> Dict[str, int]:
    """给实例分配 ``split`` 字段（分层抽样，按标签均衡）。

    Args:
        instances: 待划分实例；函数会**就地**写入 ``instance.split``。
        train_ratio / dev_ratio / test_ratio: 三者之和应为 1。
        seed: 随机种子，保证同种子下划分完全可复现。
        official: 若提供 ``{uid: split}``，命中官方划分的样本优先使用它，
            其余样本再走分层抽样。

    Returns:
        ``{"train": n, "dev": n, "test": n}`` 计数。
    """
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"划分比例之和必须为 1，当前为 {total_ratio}（{train_ratio}/{dev_ratio}/{test_ratio}）"
        )

    rng = random.Random(seed)
    official = dict(official or {})
    counts = {"train": 0, "dev": 0, "test": 0}

    # 命中官方划分的样本直接采纳
    pending: List[DataInstance] = []
    for instance in instances:
        assigned = official.get(instance.uid)
        if assigned in counts:
            instance.split = assigned
            counts[assigned] += 1
        else:
            pending.append(instance)

    if not pending:
        return counts

    # 按标签分组后各自分层抽样，避免小类别（如 818 条数据集的 UR）被抽空
    by_label: Dict[str, List[DataInstance]] = {label: [] for label in LABELS}
    for instance in pending:
        by_label.setdefault(instance.label, []).append(instance)

    for label, group in by_label.items():
        if not group:
            continue
        rng.shuffle(group)
        n_total = len(group)
        n_train = int(round(n_total * train_ratio))
        n_dev = int(round(n_total * dev_ratio))

        # 保证每个类别在 train/dev 至少各有一条（样本量 > 2 时）
        if n_total > 2:
            n_train = max(1, n_train)
            n_dev = max(1, n_dev) if dev_ratio > 0 else 0
        n_test = n_total - n_train - n_dev
        if n_test < 0:
            # 极端小样本时的兜底：优先保证 test
            n_test = max(0, n_total - n_train)
            n_dev = max(0, n_total - n_train - n_test)

        for index, instance in enumerate(group):
            if index < n_train:
                split = "train"
            elif index < n_train + n_dev:
                split = "dev"
            else:
                split = "test"
            instance.split = split
            counts[split] += 1

    return counts


# ---------------------------------------------------------------------- #
# 对外主函数
# ---------------------------------------------------------------------- #
def load_raw_dataset(
    raw_dir: str,
    name: str,
    text_mode: str = "source_replies",
    max_seq_length: int = 128,
    reply_order: str = "bfs",
    max_replies: int = 20,
    split_ratios: Optional[Mapping[str, float]] = None,
    seed: int = 42,
    prefer_official_split: bool = True,
    logger: Any = None,
) -> List[DataInstance]:
    """加载一个数据集的原始文件并产出统一的 :class:`DataInstance` 列表。

    Args:
        raw_dir: ``data/raw`` 目录（其下按 ``name`` 建子目录），或直接是数据集目录。
        name: ``twitter15`` / ``twitter16``，或演示用 ``demo``。
        text_mode / max_seq_length / reply_order / max_replies: 见
            :func:`data.processors.reply_flatten.build_encoder_text`。
        split_ratios: ``{"train":0.7,"dev":0.1,"test":0.2}``。
        seed: 划分种子。
        prefer_official_split: 找到 ``split.txt`` 时是否优先采用。
        logger: 可选的 logger。

    Returns:
        已回填 ``text`` 与 ``split`` 的实例列表。

    Raises:
        FileNotFoundError: 目录下找不到任何可识别的原始文件。
    """
    def _log(message: str, level: str = "info") -> None:
        if logger is not None:
            getattr(logger, level, logger.info)(message)

    dataset_dir = raw_dir
    if os.path.isdir(os.path.join(raw_dir, name)):
        dataset_dir = os.path.join(raw_dir, name)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

    instances: List[DataInstance] = []
    official_splits: Dict[str, str] = {}

    # ---- 布局 B：单个 JSON ----
    json_path = _first_existing(dataset_dir, _JSON_CANDIDATES, name)
    if json_path is not None:
        _log(f"[{name}] 使用 JSON 布局：{json_path}")
        instances = parse_instances_json(json_path)

    # ---- 布局 A / C：label + tree + source_tweets ----
    if not instances:
        label_path = _first_existing(dataset_dir, _LABEL_FILE_CANDIDATES, name)
        if label_path is None:
            raise FileNotFoundError(
                f"在 {dataset_dir} 中找不到 label.txt（或 {{name}}_label.txt）。"
                "请参考 data/README.md 放置原始数据，或使用 --dataset demo 跑通流程。"
            )
        labels = parse_labels(label_path)
        _log(f"[{name}] 读取 {len(labels)} 条标签：{label_path}")

        source_path = _first_existing(dataset_dir, _SOURCE_FILE_CANDIDATES, name)
        id_to_text: Dict[str, str] = {}
        if source_path is not None:
            id_to_text = parse_source_tweets(source_path)
            _log(f"[{name}] 读取 {len(id_to_text)} 条推文正文：{source_path}")
        else:
            _log(f"[{name}] 未找到推文正文文件，增强与回复上下文将不可用", level="warning")

        tree_path = _first_existing(dataset_dir, _TREE_FILE_CANDIDATES, name)
        trees: Dict[str, List[Tuple[str, str]]] = {}
        if tree_path is not None:
            trees = parse_trees(tree_path)
            _log(f"[{name}] 读取 {len(trees)} 棵传播树：{tree_path}")

        if prefer_official_split:
            split_path = _first_existing(dataset_dir, _SPLIT_FILE_CANDIDATES, name)
            if split_path is not None:
                official_splits = _parse_extra_splits(split_path)
                _log(f"[{name}] 读取官方划分 {len(official_splits)} 条：{split_path}")

        missing_text = 0
        for uid, label in labels.items():
            source_text = id_to_text.get(uid, "")
            if not source_text:
                missing_text += 1
            replies = _build_reply_forest(trees.get(uid, []), id_to_text, uid) if trees else []
            instances.append(
                DataInstance(
                    uid=uid,
                    string_value=source_text,
                    label=label,
                    replies=replies,
                    dataset=name,
                )
            )
        if missing_text:
            _log(
                f"[{name}] {missing_text} 条样本缺少原帖正文（推文可能已被删除）；"
                "这些样本的编码器输入将为空，建议在预处理阶段过滤",
                level="warning",
            )
        replies_total = sum(instance.reply_count for instance in instances)
        if replies_total == 0:
            _log(
                f"[{name}] 未解析出任何回复正文；请把 data.text_mode 设为 source_only，"
                "否则回复上下文为空（论文强调回复中的情绪与观点）",
                level="warning",
            )

    if not instances:
        raise FileNotFoundError(f"{dataset_dir} 中没有解析到任何样本")

    # 过滤空文本样本（推文被删导致）
    before = len(instances)
    instances = [item for item in instances if item.string_value.strip()]
    if len(instances) < before:
        _log(f"[{name}] 过滤掉 {before - len(instances)} 条空正文样本")

    # ---- 划分 ----
    ratios = dict(DATASET_DEFAULT_SPLIT)
    if split_ratios:
        ratios.update({key: float(value) for key, value in split_ratios.items()})
    counts = assign_splits(
        instances,
        train_ratio=ratios.get("train", 0.7),
        dev_ratio=ratios.get("dev", 0.1),
        test_ratio=ratios.get("test", 0.2),
        seed=seed,
        official=official_splits if prefer_official_split else None,
    )
    _log(f"[{name}] 划分完成：{counts}")

    # ---- 生成编码器输入 ----
    for instance in instances:
        instance.text = build_encoder_text(
            instance,
            text_mode=text_mode,
            max_seq_length=max_seq_length,
            reply_order=reply_order,
            max_replies=max_replies,
        )

    return instances

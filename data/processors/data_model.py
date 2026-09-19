# coding=utf-8
"""统一数据实例模型。

论文 Fg.2 给出的每条数据结构是 ``uid / string_value / replies`` 三件套。
本模块把这套结构固化成 :class:`DataInstance`，并作为**唯一**在
"原始数据 → 预处理 → LLM 增强 → 训练"全链路上流转的载体。

设计要点
--------
1. ``replies`` 保持嵌套（回复的回复），因为论文要求增强前后
   "replies 中的每条回复保持了原有的层次结构"；
2. 标签 :attr:`DataInstance.label` 来自数据集官方的 ``label.txt``，
   **不进入 LLM Prompt**，也不允许 LLM 改写（见 ``docs/implementation_notes.md``）；
3. :attr:`DataInstance.text` 是编码器真正看到的字符串，由
   ``data/processors/reply_flatten.py`` 依据配置生成，属于派生字段。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "LABEL_ALIASES",
    "normalize_label",
    "Reply",
    "DataInstance",
    "iter_replies",
    "count_replies",
    "instance_to_record",
    "record_to_instance",
]

# ---------------------------------------------------------------------- #
# 标签体系（论文 Table 1 / §3.2：NR / FR / TR / UR 四分类）
# ---------------------------------------------------------------------- #
LABELS: List[str] = ["NR", "FR", "TR", "UR"]

LABEL_TO_ID: Dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}

# Twitter15/16 官方 label.txt 用的全称，映射到论文的缩写
LABEL_ALIASES: Dict[str, str] = {
    "non-rumor": "NR",
    "nonrumor": "NR",
    "non_rumor": "NR",
    "non rumor": "NR",
    "nr": "NR",
    "false": "FR",
    "false-rumor": "FR",
    "falserumor": "FR",
    "false_rumor": "FR",
    "false rumor": "FR",
    "fr": "FR",
    "true": "TR",
    "true-rumor": "TR",
    "truerumor": "TR",
    "true_rumor": "TR",
    "true rumor": "TR",
    "tr": "TR",
    "unverified": "UR",
    "unverified-rumor": "UR",
    "unverifiedrumor": "UR",
    "unverified_rumor": "UR",
    "unverified rumor": "UR",
    "ur": "UR",
}


def normalize_label(raw: Any) -> str:
    """把数据集里的各种标签写法统一成 ``NR / FR / TR / UR``。

    支持：官方全称（``non-rumor`` / ``false`` / ``true`` / ``unverified``）、
    论文缩写、大小写混写、以及已经是 0..3 的整数下标。

    Raises:
        ValueError: 无法识别的标签。
    """
    if isinstance(raw, bool):
        raise ValueError(f"非法标签：{raw!r}")
    if isinstance(raw, int):
        if raw in ID_TO_LABEL:
            return ID_TO_LABEL[raw]
        raise ValueError(f"标签下标越界：{raw}（合法范围 0..{len(LABELS) - 1}）")

    text = str(raw).strip()
    if text == "":
        raise ValueError("标签为空")

    # 数字字符串按下标处理
    if text.isdigit():
        index = int(text)
        if index in ID_TO_LABEL:
            return ID_TO_LABEL[index]
        raise ValueError(f"标签下标越界：{text}")

    key = text.lower()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    upper = text.upper()
    if upper in LABEL_TO_ID:
        return upper
    raise ValueError(f"无法识别的标签：{raw!r}")


# ---------------------------------------------------------------------- #
# 回复节点
# ---------------------------------------------------------------------- #
@dataclass
class Reply:
    """一条回复（可继续嵌套子回复）。"""

    uid: str
    string_value: str
    replies: List["Reply"] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        """序列化成与论文 Fg.2 一致的字段名。"""
        return {
            "uid": self.uid,
            "string_value": self.string_value,
            "replies": [child.to_record() for child in self.replies],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], uid_fallback: str = "") -> "Reply":
        """从 dict 反序列化。

        容错点：真实数据里回复正文的键名可能是 ``text`` / ``string_value`` /
        ``content``，这里全部接受；缺失时退化为空串而不是抛错，以免个别脏数据
        让整个数据集加载失败。
        """
        if not isinstance(record, Mapping):
            raise TypeError(f"回复节点必须是 mapping，收到 {type(record).__name__}")

        uid = str(
            record.get("uid")
            or record.get("id")
            or record.get("id_str")
            or uid_fallback
        )
        text = (
            record.get("string_value")
            or record.get("text")
            or record.get("content")
            or record.get("full_text")
            or ""
        )
        children_raw = record.get("replies") or record.get("children") or []
        children: List[Reply] = []
        if isinstance(children_raw, Iterable) and not isinstance(children_raw, (str, bytes)):
            for index, child in enumerate(children_raw):
                children.append(cls.from_record(child, uid_fallback=f"{uid}_{index}"))

        return cls(uid=uid, string_value=str(text), replies=children)


def iter_replies(replies: Sequence[Reply], order: str = "bfs") -> Iterable[Reply]:
    """按指定顺序遍历回复树，逐条产出 :class:`Reply`。

    Args:
        replies: 顶层回复列表。
        order: ``bfs``（默认，按信息扩散的层次逐层展开）或 ``dfs``。
    """
    if order not in ("bfs", "dfs"):
        raise ValueError(f"reply_order 只支持 bfs/dfs，收到 {order!r}")

    if order == "dfs":
        stack = list(reversed(list(replies)))
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.replies))
        return

    queue = list(replies)
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        yield node
        queue.extend(node.replies)


def count_replies(replies: Sequence[Reply]) -> int:
    """统计回复树的总节点数（含嵌套层级）。"""
    total = 0
    for node in iter_replies(replies, order="bfs"):
        total += 1
    return total


# ---------------------------------------------------------------------- #
# 数据实例
# ---------------------------------------------------------------------- #
@dataclass
class DataInstance:
    """一条社交网络数据实例（论文 Fg.2 的结构 + 训练所需元信息）。"""

    uid: str
    string_value: str
    label: str
    replies: List[Reply] = field(default_factory=list)
    text: str = ""
    label_id: Optional[int] = None
    split: Optional[str] = None
    dataset: Optional[str] = None
    # 增强相关字段：增强样本沿用同一个 uid，靠 augmented 标记区分
    augmented: bool = False
    augment_round: int = 0
    original_uid: Optional[str] = None
    quality: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.uid = str(self.uid)
        self.string_value = str(self.string_value)
        self.label = normalize_label(self.label)
        if self.label_id is None:
            self.label_id = LABEL_TO_ID[self.label]
        if self.original_uid is None:
            self.original_uid = self.uid

    # ------------------------------------------------------------------ #
    @property
    def is_augmented(self) -> bool:
        """是否为 LLM 增强样本。"""
        return bool(self.augmented)

    @property
    def reply_count(self) -> int:
        """回复总数（含嵌套）。"""
        return count_replies(self.replies)

    def encoder_text(self, fallback_to_source: bool = True) -> str:
        """返回编码器输入文本；``text`` 为空时退回原帖正文。"""
        if self.text:
            return self.text
        return self.string_value if fallback_to_source else ""

    def prompt_payload(self) -> Dict[str, Any]:
        """构造喂给 LLM 的字段集合。

        **只包含** ``uid / string_value / replies`` —— 与论文 Fg.2 完全一致，
        刻意不含 label，避免把监督信号泄漏进增强过程。
        """
        return {
            "uid": self.uid,
            "string_value": self.string_value,
            "replies": [reply.to_record() for reply in self.replies],
        }

    def prompt_hash(self) -> str:
        """对 Prompt 载荷取哈希，用于增强缓存键。"""
        payload = json.dumps(self.prompt_payload(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def to_record(self) -> Dict[str, Any]:
        """序列化为可直接写 JSONL 的普通 dict。"""
        return instance_to_record(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DataInstance":
        """从 dict 构造实例。"""
        return record_to_instance(record)


# ---------------------------------------------------------------------- #
# 序列化
# ---------------------------------------------------------------------- #
def instance_to_record(instance: DataInstance) -> Dict[str, Any]:
    """把 :class:`DataInstance` 转成 JSONL 可写 dict（字段顺序稳定）。"""
    record: Dict[str, Any] = {
        "uid": instance.uid,
        "string_value": instance.string_value,
        "replies": [reply.to_record() for reply in instance.replies],
        "label": instance.label,
        "label_id": instance.label_id,
        "text": instance.text,
    }
    if instance.split is not None:
        record["split"] = instance.split
    if instance.dataset is not None:
        record["source"] = instance.dataset
    if instance.augmented:
        record["augmented"] = True
        record["augment_round"] = instance.augment_round
        record["original_uid"] = instance.original_uid
        if instance.quality is not None:
            record["quality"] = instance.quality
    for key, value in instance.meta.items():
        # meta 不允许覆盖核心字段，避免脏数据破坏结构
        if key not in record:
            record[key] = value
    return record


def record_to_instance(record: Mapping[str, Any]) -> DataInstance:
    """从 JSONL 行构造 :class:`DataInstance`。

    兼容三种输入：
    1. 本仓库 ``prepare_data.py`` 产出的完整记录；
    2. 论文 Fg.2 的裸结构（只有 uid/string_value/replies，无 label）——
       此时要求调用方另行提供标签，故这里会抛错提示；
    3. 增强记录（带 ``augmented`` / ``augment_round``）。
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"数据记录必须是 mapping，收到 {type(record).__name__}")
    if "uid" not in record:
        raise KeyError("数据记录缺少 uid 字段")

    if "label" not in record and "label_id" not in record:
        raise KeyError(
            "数据记录缺少 label 字段。论文 Fg.2 的裸结构不含标签，"
            "标签必须来自数据集官方的 label.txt（见 data/README.md）"
        )

    raw_label = record.get("label", record.get("label_id"))
    label = normalize_label(raw_label)

    replies_raw = record.get("replies") or []
    if not isinstance(replies_raw, Iterable) or isinstance(replies_raw, (str, bytes)):
        raise TypeError("replies 必须是列表")

    replies = [
        Reply.from_record(item, uid_fallback=f"{record['uid']}_{index}")
        for index, item in enumerate(replies_raw)
    ]

    raw_text = record.get("text")
    known_keys = {
        "uid", "string_value", "replies", "label", "label_id", "text",
        "split", "source", "augmented", "augment_round", "original_uid",
        "quality",
    }
    meta = {key: value for key, value in record.items() if key not in known_keys}

    return DataInstance(
        uid=str(record["uid"]),
        string_value=str(record.get("string_value", "")),
        label=label,
        replies=replies,
        text=str(raw_text) if raw_text is not None else "",
        label_id=LABEL_TO_ID[label],
        split=record.get("split"),
        dataset=record.get("source"),
        augmented=bool(record.get("augmented", False)),
        augment_round=int(record.get("augment_round", 0) or 0),
        original_uid=str(record.get("original_uid") or record["uid"]),
        quality=record.get("quality"),
        meta=meta,
    )

# coding=utf-8
"""回复树展平与编码器输入文本构造。

论文强调"社交媒体用户的情绪和观点在评估信息真实性中起着关键作用"，
因此回复不是可丢弃的附属信息，而是分类信号的一部分。

三种 ``text_mode``（配置项 ``data.text_mode``）
------------------------------------------------
``source_only``
    只用原帖正文（即 ``string_value``），对应最保守的设定。
``source_replies``（默认）
    原帖 + 展平后的回复，用 ``[SEP]`` 分隔，整体作为单序列输入 BERT。
    这是与原开源项目 "单序列 BERT 分类" 结构最兼容的方式。
``hierarchical``
    原帖与每条回复各自带角色标记（``Source:`` / ``Reply:`` / ``Reply2:``），
    层级信息显式保留在文本里，仍为单序列（不做多序列分层编码）。

token 预算
----------
BERT 的 ``max_seq_length`` 是硬约束。分配策略：

1. 原帖**优先生成**，最多占 ``sqrt(budget)`` 比例的上限；
2. 剩余预算按 :func:`iter_replies` 的顺序（默认 BFS，贴近传播广度）逐条分配给回复；
3. 单条内容放不下时**保留头部**（社交文本的重要信息通常在开头）；
4. 若连原帖都要截断，则把全部预算给原帖。

这里的 budget 以"词"为单位估算（英文按空格切分）。真正做 WordPiece 分词在
``data/dataset.py`` 里由 tokenizer 完成——本模块刻意不依赖 transformers，
以便预处理脚本可以独立运行。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .data_model import DataInstance, Reply, iter_replies

__all__ = [
    "SOURCE_PREFIX",
    "REPLY_PREFIX",
    "SEPARATOR",
    "ReplySegment",
    "flatten_replies",
    "build_encoder_text",
]

# 角色标记：让单序列 BERT 也能区分"谁说的"
SOURCE_PREFIX = "Source:"
REPLY_PREFIX = "Reply:"
# 与 BERT 的段间分隔符保持一致，便于人工核对
SEPARATOR = "[SEP]"

# 估算 token 数时的粗略系数：英文约 1.3 个 WordPiece / 词
TOKENS_PER_WORD = 1.3


class ReplySegment:
    """展平后的一条回复及其层级信息。"""

    __slots__ = ("uid", "text", "depth", "index")

    def __init__(self, uid: str, text: str, depth: int, index: int):
        self.uid = uid
        self.text = text
        self.depth = depth
        self.index = index

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"ReplySegment(uid={self.uid!r}, depth={self.depth}, index={self.index})"


def _truncate_to_words(text: str, max_words: int) -> str:
    """按词截断，保留头部；``max_words<=0`` 返回空串。"""
    if max_words <= 0:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


def _estimate_words(text: str) -> int:
    """估算一段文本占多少个"词"。"""
    return len(text.split())


def flatten_replies(
    replies: Sequence[Reply],
    order: str = "bfs",
    max_replies: int = 0,
) -> List[ReplySegment]:
    """把回复树展平成带层级信息的列表。

    Args:
        replies: 顶层回复列表。
        order: ``bfs`` 或 ``dfs``，见 :func:`data.processors.data_model.iter_replies`。
        max_replies: 最多返回多少条；``<=0`` 表示不限制。

    Returns:
        :class:`ReplySegment` 列表，``index`` 是展平后的序号，``depth`` 是嵌套深度。
    """
    segments: List[ReplySegment] = []

    if order == "bfs":
        # 广度优先需要显式记录深度，故单独实现而不是复用 iter_replies
        queue: List[Tuple[Reply, int]] = [(node, 1) for node in replies]
        head = 0
        while head < len(queue):
            node, depth = queue[head]
            head += 1
            segments.append(
                ReplySegment(uid=node.uid, text=node.string_value, depth=depth, index=len(segments))
            )
            queue.extend((child, depth + 1) for child in node.replies)
    else:
        # 深度优先：用栈模拟，逐支深入
        stack: List[Tuple[Reply, int]] = [
            (node, 1) for node in reversed(list(replies))
        ]
        while stack:
            node, depth = stack.pop()
            segments.append(
                ReplySegment(uid=node.uid, text=node.string_value, depth=depth, index=len(segments))
            )
            stack.extend((child, depth + 1) for child in reversed(node.replies))

    # 丢掉空回复，避免污染文本
    segments = [seg for seg in segments if seg.text and seg.text.strip()]
    if max_replies and max_replies > 0:
        segments = segments[:max_replies]
    return segments


def _allocate_budget(
    total_words: int,
    source_words: int,
    reply_word_counts: Sequence[int],
) -> Tuple[int, List[int]]:
    """给原帖与回复分配词数预算。

    规则：
    * 原帖上限 = ``min(source_words, ceil(sqrt(total) * 4))``，保证原帖不被回复挤掉；
    * 若原帖比上限还长，则把全部预算给原帖；
    * 剩余预算按顺序贪心分配，分完即止（后面的回复预算为 0，等于丢弃）。
    """
    if total_words <= 0:
        return 0, [0] * len(reply_word_counts)

    source_cap = max(1, int(math.ceil(math.sqrt(total_words) * 4)))
    source_budget = min(source_words, source_cap)
    if source_words > source_cap:
        # 原帖本身超预算：全部给它，回复不再纳入
        return source_cap, [0] * len(reply_word_counts)

    remaining = total_words - source_budget
    reply_budgets: List[int] = []
    for count in reply_word_counts:
        if remaining <= 0:
            reply_budgets.append(0)
            continue
        take = min(count, remaining)
        reply_budgets.append(take)
        remaining -= take
    return source_budget, reply_budgets


def build_encoder_text(
    instance: DataInstance,
    text_mode: str = "source_replies",
    max_seq_length: int = 128,
    reply_order: str = "bfs",
    max_replies: int = 20,
) -> str:
    """按 ``text_mode`` 生成编码器输入文本。

    Args:
        instance: 数据实例（读取 ``string_value`` 与 ``replies``）。
        text_mode: ``source_only`` / ``source_replies`` / ``hierarchical``。
        max_seq_length: BERT 最大序列长度，用于换算词数预算。
        reply_order: 回复遍历顺序。
        max_replies: 最多纳入的回复条数。

    Returns:
        拼好的字符串。``source_only`` 且无回复时即为原帖正文。

    Note:
        预算换算：``word_budget = max_seq_length / TOKENS_PER_WORD``，再为
        ``[CLS]``/``[SEP]`` 与角色标记留 8 个词的余量。
    """
    if text_mode not in ("source_only", "source_replies", "hierarchical"):
        raise ValueError(
            f"text_mode 只支持 source_only/source_replies/hierarchical，收到 {text_mode!r}"
        )

    source = (instance.string_value or "").strip()
    if text_mode == "source_only" or not instance.replies:
        return source

    word_budget = max(16, int(max_seq_length / TOKENS_PER_WORD) - 8)
    segments = flatten_replies(instance.replies, order=reply_order, max_replies=max_replies)
    if not segments:
        return source

    reply_word_counts = [_estimate_words(seg.text) for seg in segments]
    source_budget, reply_budgets = _allocate_budget(
        total_words=word_budget,
        source_words=_estimate_words(source),
        reply_word_counts=reply_word_counts,
    )

    source_text = _truncate_to_words(source, source_budget)

    parts: List[str] = []
    if text_mode == "source_replies":
        parts.append(source_text)
        for seg, budget in zip(segments, reply_budgets):
            if budget <= 0:
                continue
            parts.append(_truncate_to_words(seg.text, budget))
        return f" {SEPARATOR} ".join(part for part in parts if part)

    # hierarchical：显式标注角色与嵌套深度
    parts.append(f"{SOURCE_PREFIX} {source_text}")
    for seg, budget in zip(segments, reply_budgets):
        if budget <= 0:
            continue
        marker = REPLY_PREFIX if seg.depth <= 1 else f"Reply{seg.depth}:"
        parts.append(f"{marker} {_truncate_to_words(seg.text, budget)}")
    return f" {SEPARATOR} ".join(part for part in parts if part)


def attach_encoder_text(
    instances: Sequence[DataInstance],
    text_mode: str = "source_replies",
    max_seq_length: int = 128,
    reply_order: str = "bfs",
    max_replies: int = 20,
) -> None:
    """就地回填 :attr:`DataInstance.text`（对所有实例生效）。

    单独提供这个函数是因为增强样本也需要用**同一套**规则生成 text，
    否则原样本与增强样本的序列长度分布会不一致，破坏对比学习的配对前提。
    """
    for instance in instances:
        instance.text = build_encoder_text(
            instance,
            text_mode=text_mode,
            max_seq_length=max_seq_length,
            reply_order=reply_order,
            max_replies=max_replies,
        )

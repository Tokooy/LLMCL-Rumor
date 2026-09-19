# coding=utf-8
"""增强结果解析与质量校验。

LLM 的输出必须先过这一关，才能进入训练集。校验项与论文 Fg.3 的四条设计目标
一一对应（见 :mod:`src.llm.prompts` 的对照表）：

===================== ==========================================================
校验项                 对应目标 / 依据
===================== ==========================================================
JSON 可解析            "输出格式一致"的前置条件
结构完全一致           约束 C2 + 论文 Fg.4"增强前后数据均包含相同的字段"
uid / 回复数量一致      约束 C2（禁止增删回复）
字符覆盖率           约束 C4（防止 LLM 摘要化 / 大幅扩写）
字符 n-gram 重合度      约束 C3（多样性：重合度越低越多样）
词级重合度             约束 C3（论文"共享内容词不超过 60%"的近似度量）
语义相似度（可选）      约束 C4（需要句向量模型，缺省关闭）
===================== ==========================================================

校验失败**不会**让整批任务崩掉：:func:`merge_augmentation` 会把原样本原样作为
"未增强"返回，并在 ``quality`` 中记录失败原因，由上层决定是重试还是丢弃。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data.processors.data_model import DataInstance, Reply

__all__ = [
    "QualityReport",
    "strip_wrappers",
    "extract_json_object",
    "parse_augmentation_response",
    "char_ngram_overlap",
    "word_overlap",
    "text_similarity",
    "count_reply_nodes",
    "check_structure",
    "build_quality_report",
    "merge_augmentation",
    "DEFAULT_OVERLAP_MAX",
    "DEFAULT_LENGTH_RATIO_RANGE",
]

# 约束 C3：与原样本共享的内容词比例上限（论文写作 "no more than 60%"）
DEFAULT_OVERLAP_MAX = 0.75
# 约束 C4：改写后长度相对原样本的合理区间（防止摘要化或凭空扩写）
DEFAULT_LENGTH_RATIO_RANGE: Tuple[float, float] = (0.5, 2.5)

# 推理型模型可能输出的思考块，必须剥掉
_THINK_TAGS = ("think", "thinking", "reasoning", "analysis")

# 推理块的**闭合**标签在各模型里写法不同，实测至少三种：
#   DeepSeek-R1 等：   thinking ... <｜end▁of▁thinking｜>
#   Qwen3 等：         thinking ... <｜end▁of▁thinking｜>          （闭合标签自带前缀，不是 </think>）
#   Harmony 风格：    <|channel|>analysis<|message|> ... <|end|>
# 只写 `</{tag}>` 只能匹配第一种，对 Qwen 会完全失效——那时抽取只能靠括号扫描，
# 而思考过程里经常出现示例 JSON，扫描可能抽出错误对象。
_THINK_BLOCK_PATTERNS = (
    r"<{tag}>.*?</{tag}\s*>",              # </think>（允许闭合标签内有空白）
    r"<\|{tag}\|>.*?<\|end\|>",            # <|think|> ... <|end|>
    r"<\|channel\|>\s*{tag}\s*<\|message\|>.*?<\|end\|>",   # Harmony 风格
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")


def _think_block_regex(tag: str) -> re.Pattern:
    """把某个思考块标签的所有已知写法编译成一个正则。

    不同写法的闭合标签差异很大（``</think>`` vs ``</think>`` vs ``<|end|>``），
    因此把候选模式用 ``|`` 连起来，一次性替换掉所有形态。
    """
    alternatives = "|".join(
        pattern.format(tag=re.escape(tag)) for pattern in _THINK_BLOCK_PATTERNS
    )
    return re.compile(f"(?:{alternatives})", re.DOTALL | re.IGNORECASE)


_THINK_REGEXES = tuple(_think_block_regex(tag) for tag in _THINK_TAGS)


# ---------------------------------------------------------------------- #
# 文本清洗与 JSON 抽取
# ---------------------------------------------------------------------- #
def strip_wrappers(text: str) -> str:
    """剥掉 markdown 代码围栏与推理模型的思考块。"""
    if not text:
        return ""
    result = text.strip()

    # 剥 ```json ... ```
    if result.startswith("```"):
        result = _FENCE_RE.sub("", result, count=1)
        result = _FENCE_RE.sub("", result, count=1)
        result = result.strip()

    # 剥思考块（含 Qwen 的 </think> 写法）
    for regex in _THINK_REGEXES:
        result = regex.sub("", result).strip()

    # 兜底：闭合标签写成了非标准形态时，"<tag>" 之后到文本末尾的整段都不可信，
    # 但如果剥掉后什么都不剩，说明真正的答案就在里面，此时保留原文交给 JSON 扫描。
    for tag in _THINK_TAGS:
        if f"<{tag}>" in result:
            candidate = result.split(f"<{tag}>")[0].strip()
            if candidate:
                result = candidate

    # 剥常见前缀
    for prefix in ("Output:", "OUTPUT:", "Result:", "Answer:", "JSON:"):
        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    return result


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中抽出第一个完整的 JSON 对象。

    实现方式是按括号配对扫描并逐字符尝试 ``json.loads``，这样即使模型在 JSON
    前后夹了说明文字也能正确抽取（比正则更稳）。

    Raises:
        ValueError: 找不到可解析的 JSON 对象。
    """
    cleaned = strip_wrappers(text)
    if not cleaned:
        raise ValueError("模型输出为空")

    # 快路径：整体就是 JSON
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, Mapping):
            return dict(payload)
    except json.JSONDecodeError:
        pass

    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = cleaned[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(payload, Mapping):
                        return dict(payload)
                    start = None

    raise ValueError("模型输出中找不到合法的 JSON 对象")


def parse_augmentation_response(text: str) -> Dict[str, Any]:
    """解析并做基础字段检查，返回 ``{uid, string_value, replies}``。

    Raises:
        ValueError: JSON 非法、字段缺失或字段类型不对。
    """
    payload = extract_json_object(text)

    if "string_value" not in payload:
        # 容错：有些模型会把结果藏在 data/result/output 里
        for key in ("data", "result", "output", "augmented", "rewritten"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and "string_value" in nested:
                payload = dict(nested)
                break
    if "string_value" not in payload:
        raise ValueError(f"输出缺少 string_value 字段；实际字段：{sorted(payload.keys())}")

    if not isinstance(payload["string_value"], str) or not payload["string_value"].strip():
        raise ValueError("string_value 必须是非空字符串")

    replies = payload.get("replies", [])
    if replies is None:
        replies = []
    if not isinstance(replies, list):
        raise ValueError(f"replies 必须是数组，实际类型 {type(replies).__name__}")

    _validate_reply_nodes(replies)
    return payload


def _validate_reply_nodes(nodes: Sequence[Any], path: str = "replies") -> None:
    """递归检查回复节点结构（必须是带 string_value 的对象数组）。"""
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise ValueError(f"{path} 必须是数组")
    for index, node in enumerate(nodes):
        here = f"{path}[{index}]"
        if not isinstance(node, Mapping):
            raise ValueError(f"{here} 必须是对象，实际类型 {type(node).__name__}")
        if "string_value" not in node:
            raise ValueError(f"{here} 缺少 string_value 字段")
        if not isinstance(node["string_value"], str):
            raise ValueError(f"{here}.string_value 必须是字符串")
        children = node.get("replies") or []
        if children:
            _validate_reply_nodes(children, f"{here}.replies")


# ---------------------------------------------------------------------- #
# 相似度度量
# ---------------------------------------------------------------------- #
def _char_ngrams(text: str, n: int = 2) -> set:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def char_ngram_overlap(source: str, target: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 重合度，取值 ``[0, 1]``；越小越"表达不同"。"""
    left, right = _char_ngrams(source, n), _char_ngrams(target, n)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _content_words(text: str) -> set:
    """取出近似"内容词"的集合（去掉极短的停用词与标点）。"""
    tokens = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return {token for token in tokens if len(token) > 2}


def word_overlap(source: str, target: str) -> float:
    """词级 Jaccard 重合度，是论文约束 C3"共享内容词比例"的近似度量。"""
    left, right = _content_words(source), _content_words(target)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def text_similarity(source: str, target: str, encoder: Optional[Any] = None) -> Optional[float]:
    """语义相似度。

    Args:
        encoder: 任何提供 ``encode(list[str]) -> list[list[float]]`` 或
            ``similarity(a, b) -> float`` 的对象。为 ``None`` 时返回 ``None``，
            表示"未做语义校验"（不阻断流程）。

    Note:
        默认关闭，是因为加载句向量模型会给数据准备阶段引入额外显存开销；
        论文的语义一致约束在实现上主要靠 Prompt 约束 C4 保证，此处提供的是
        "可选的第二道闸门"。
    """
    if encoder is None:
        return None
    if hasattr(encoder, "similarity"):
        return float(encoder.similarity(source, target))
    if hasattr(encoder, "encode"):
        vectors = encoder.encode([source, target])
        if len(vectors) != 2:
            return None
        import math

        left, right = vectors[0], vectors[1]
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        if norm_left == 0 or norm_right == 0:
            return None
        return dot / (norm_left * norm_right)
    return None


# ---------------------------------------------------------------------- #
# 结构校验
# ---------------------------------------------------------------------- #
def count_reply_nodes(nodes: Sequence[Mapping[str, Any]]) -> int:
    """统计增强结果中的回复节点总数（含嵌套层级）。"""
    total = 0
    for node in nodes:
        total += 1
        total += count_reply_nodes(node.get("replies") or [])
    return total


def check_structure(original: DataInstance, payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """校验增强结果与原样本的结构一致性（约束 C2）。

    Returns:
        ``(是否通过, 问题列表)``。
    """
    problems: List[str] = []

    payload_uid = payload.get("uid")
    if payload_uid is not None and str(payload_uid) != original.uid:
        problems.append(f"uid 被改动：{original.uid} -> {payload_uid}")

    expected_keys = {"uid", "string_value", "replies"}
    actual_keys = set(payload.keys())
    # 允许模型多返回 explanation 之类的辅助键，但必需键必须齐全
    missing = expected_keys - actual_keys
    if missing:
        problems.append(f"缺少必需字段 {sorted(missing)}")

    original_ids = _collect_reply_ids(original.replies)
    returned_ids = _collect_reply_ids_from_payload(payload.get("replies") or [])

    if len(returned_ids) != len(original_ids):
        problems.append(
            f"回复数量不一致：原 {len(original_ids)} 条，增强 {len(returned_ids)} 条"
        )
    if returned_ids and set(returned_ids) != set(original_ids):
        missing_ids = set(original_ids) - set(returned_ids)
        extra_ids = set(returned_ids) - set(original_ids)
        if missing_ids:
            problems.append(f"增强结果丢失回复 uid：{sorted(missing_ids)[:5]}")
        if extra_ids:
            problems.append(f"增强结果新增回复 uid：{sorted(extra_ids)[:5]}")

    if not original.string_value.strip():
        problems.append("原样本 string_value 为空，无法增强")
    if not str(payload.get("string_value", "")).strip():
        problems.append("增强结果 string_value 为空")

    return (not problems), problems


def _collect_reply_ids(nodes: Sequence[Reply]) -> List[str]:
    ids: List[str] = []
    for node in nodes:
        ids.append(node.uid)
        ids.extend(_collect_reply_ids(node.replies))
    return ids


def _collect_reply_ids_from_payload(nodes: Sequence[Mapping[str, Any]]) -> List[str]:
    ids: List[str] = []
    for index, node in enumerate(nodes):
        ids.append(str(node.get("uid") or f"__missing_{index}__"))
        ids.extend(_collect_reply_ids_from_payload(node.get("replies") or []))
    return ids


# ---------------------------------------------------------------------- #
# 质量报告
# ---------------------------------------------------------------------- #
@dataclass
class QualityReport:
    """一条增强样本的质量指标（会写入 JSONL 的 ``quality`` 字段）。"""

    structure_ok: bool = False
    semantic_ok: bool = True
    diversity_ok: bool = True
    char_ngram_overlap: float = 1.0
    word_overlap: float = 1.0
    length_ratio: float = 1.0
    semantic_similarity: Optional[float] = None
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """结构必须通过；语义/多样性只作为软约束（论文未给硬阈值）。"""
        return self.structure_ok and self.semantic_ok

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "structure_ok": self.structure_ok,
            "semantic_ok": self.semantic_ok,
            "diversity_ok": self.diversity_ok,
            "char_ngram_overlap": round(self.char_ngram_overlap, 4),
            "word_overlap": round(self.word_overlap, 4),
            "length_ratio": round(self.length_ratio, 4),
        }
        if self.semantic_similarity is not None:
            payload["semantic_similarity"] = round(self.semantic_similarity, 4)
        if self.problems:
            payload["problems"] = list(self.problems)
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def build_quality_report(
    original: DataInstance,
    payload: Mapping[str, Any],
    encoder: Optional[Any] = None,
    overlap_max: float = DEFAULT_OVERLAP_MAX,
    length_ratio_range: Tuple[float, float] = DEFAULT_LENGTH_RATIO_RANGE,
    semantic_threshold: float = 0.5,
) -> QualityReport:
    """对一条增强结果计算完整质量报告。"""
    report = QualityReport()

    structure_ok, problems = check_structure(original, payload)
    report.structure_ok = structure_ok
    report.problems.extend(problems)

    source_text = original.string_value
    target_text = str(payload.get("string_value", ""))

    report.char_ngram_overlap = char_ngram_overlap(source_text, target_text)
    report.word_overlap = word_overlap(source_text, target_text)

    source_len = max(1, len(source_text.strip()))
    report.length_ratio = len(target_text.strip()) / source_len

    low, high = length_ratio_range
    if not (low <= report.length_ratio <= high):
        report.warnings.append(
            f"长度比例 {report.length_ratio:.2f} 超出合理区间 [{low}, {high}]，"
            "可能发生摘要化或凭空扩写（约束 C4）"
        )
        report.semantic_ok = False

    if report.word_overlap > overlap_max:
        report.warnings.append(
            f"词级重合度 {report.word_overlap:.2f} 高于阈值 {overlap_max}，"
            "改写幅度不足（约束 C3）"
        )
        report.diversity_ok = False

    similarity = text_similarity(source_text, target_text, encoder=encoder)
    if similarity is not None:
        report.semantic_similarity = similarity
        if similarity < semantic_threshold:
            report.warnings.append(
                f"语义相似度 {similarity:.3f} 低于阈值 {semantic_threshold}（约束 C4）"
            )
            report.semantic_ok = False

    return report


# ---------------------------------------------------------------------- #
# 合并成新的数据实例
# ---------------------------------------------------------------------- #
def _to_replies(nodes: Sequence[Mapping[str, Any]], fallback: Sequence[Reply]) -> List[Reply]:
    """把增强结果的回复节点转成 :class:`Reply`。

    若模型漏了某条回复的正文，则用原样本的对应回复兜底，保证"不丢信息"。
    """
    fallback_list = list(fallback)
    replies: List[Reply] = []
    for index, node in enumerate(nodes):
        text = str(node.get("string_value") or "").strip()
        if not text and index < len(fallback_list):
            text = fallback_list[index].string_value
        children_payload = node.get("replies") or []
        children_fallback = fallback_list[index].replies if index < len(fallback_list) else []
        replies.append(
            Reply(
                uid=str(node.get("uid") or (fallback_list[index].uid if index < len(fallback_list) else f"r{index}")),
                string_value=text,
                replies=_to_replies(children_payload, children_fallback) if children_payload else [],
            )
        )
    return replies


def merge_augmentation(
    original: DataInstance,
    payload: Optional[Mapping[str, Any]],
    report: Optional[QualityReport] = None,
    augment_round: int = 1,
    model_name: str = "",
    prompt_hash: str = "",
    text_mode: str = "source_replies",
    max_seq_length: int = 128,
    success: bool = True,
) -> DataInstance:
    """把 LLM 的改写结果合并成一个新的 :class:`DataInstance`。

    关键约定（与 :mod:`data.dataset` 的配对逻辑对应）：

    * ``uid`` 与原样本**保持一致**，这样按 uid 就能找回正样本对；
    * ``original_uid`` 指向原样本 uid（二者相同，保留字段是为了语义清晰）；
    * ``label`` / ``label_id`` 直接沿用原样本——**标签绝不由 LLM 产生**；
    * ``augmented=True``、``augment_round=k`` 记录增强轮次；
    * ``text`` 用与原样本完全相同的规则重新生成。

    Args:
        success: 为 False 时输出"未增强"副本（``meta['augment_failed']=True``），
            供上层统计失败率，而不是把失败样本静默丢掉。
    """
    from data.processors.reply_flatten import build_encoder_text

    if payload is not None and success:
        replies = _to_replies(payload.get("replies") or [], original.replies)
        string_value = str(payload.get("string_value") or original.string_value)
    else:
        replies = [
            Reply(uid=node.uid, string_value=node.string_value, replies=_clone_replies(node.replies))
            for node in original.replies
        ]
        string_value = original.string_value

    instance = DataInstance(
        uid=original.uid,
        string_value=string_value,
        label=original.label,
        replies=replies,
        augmented=True,
        augment_round=augment_round,
        original_uid=original.uid,
        quality=report.to_dict() if report is not None else None,
        dataset=original.dataset,
        split=original.split,
        meta={
            "augment_model": model_name,
            "augment_prompt_hash": prompt_hash,
            "augment_failed": not success,
        },
    )
    instance.text = build_encoder_text(
        instance, text_mode=text_mode, max_seq_length=max_seq_length
    )
    return instance


def _clone_replies(nodes: Sequence[Reply]) -> List[Reply]:
    return [
        Reply(uid=node.uid, string_value=node.string_value, replies=_clone_replies(node.replies))
        for node in nodes
    ]

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

# 推理块的闭合标签在各模型/各 API 上写法不同，实测至少三类：
#   1) 标准开闭对：      "<" + "think> 推理过程 " + "</" + "think>"
#   2) **只有闭合标签**（推理 API 常见，开标签被服务端吃掉）：
#      正文里只剩 "<" + "｜end▁of▁thinking｜>"，竖向分隔符有半角 "|" 与全角 "｜" 两种
#   3) Harmony 风格：    "<|channel|>analysis<|message|> ... <|end|>"
# 只写 `</{tag}>` 对第 2 类完全失效——那时抽取只能靠括号扫描，
# 而思考过程里经常出现示例 JSON，扫描会抽出错误对象。
#
# ⚠️ 注意：这些字符串**不能**直接写成字面量的尖括号形式去描述，
# 因为某些文本处理链路会把看似 XML 标签的片段吃掉。规则里一律用转义写法构造。
_THINK_BLOCK_PATTERNS = (
    r"<{tag}>.*?</{tag}\s*>",              # 标准开闭对
    r"<\|{tag}\|>.*?<\|end\|>",            # <|think|> ... <|end|>
    r"<\|channel\|>\s*{tag}\s*<\|message\|>.*?<\|end\|>",   # Harmony 风格
)

#: 只有闭合标签的形态（推理 API 常见：开标签被服务端吃掉，正文里只剩闭合标签）。
#: 竖向分隔符有半角 ``|`` 与全角 ``｜`` 两种写法。
#:
#: **只识别已知的收尾关键字**（end / thinking / analysis / reasoning）——
#: 早期实现把所有 ``<|…|>`` 形状的片段都当标签剥掉，结果会把正文里合法的
#: 尖括号内容（例如 "see the marker <|note|> in text"）一起吃掉。
#: 这个模式**不参与** ``.format(tag=...)``，也不编进 ``_THINK_REGEXES``：
#: 它由 :func:`strip_wrappers` 单独处理（保留标签之后的内容）。
_CLOSE_ONLY_PATTERN = (
    r"<[/\u005c]?[|\uff5c][^<>\n]{0,40}?"
    r"(?:end|thinking|analysis|reasoning)[^<>\n]{0,20}?[|\uff5c]\s*>"
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")
_CLOSE_ONLY_RE = re.compile(_CLOSE_ONLY_PATTERN, re.IGNORECASE)


def _think_block_regex(tag: str) -> re.Pattern:
    """把某个思考块标签的**成对**写法编译成一个正则。

    不同写法的闭合标签差异很大（``</think>`` 与 ``<|end|>``），
    因此把候选模式用 ``|`` 连起来，一次性替换掉所有形态。

    注意：只有闭合标签的形态（:data:`_CLOSE_ONLY_PATTERN`）**不在这里**，
    它需要"保留标签之后的内容"这一处理，由 :func:`strip_wrappers` 单独负责。
    """
    alternatives = [
        pattern.format(tag=re.escape(tag)) for pattern in _THINK_BLOCK_PATTERNS
    ]
    return re.compile("(?:" + "|".join(alternatives) + ")", re.DOTALL | re.IGNORECASE)


_THINK_REGEXES = tuple(_think_block_regex(tag) for tag in _THINK_TAGS)


# ---------------------------------------------------------------------- #
# 文本清洗与 JSON 抽取
# ---------------------------------------------------------------------- #
def _has_json(text: str) -> bool:
    """文本中是否含可解析的 JSON 对象。"""
    try:
        extract_json_object(text)
    except ValueError:
        return False
    return True


def strip_wrappers(text: str) -> str:
    """剥掉 markdown 代码围栏与推理模型的思考块。

    处理顺序（每一步都只在"结果仍然含有 JSON"时才采纳，避免把答案一起丢掉）：

    1. 剥 ``` 围栏；
    2. 若出现**只有闭合标签**的形态（推理 API 常把开标签吃掉，只剩
       ``<｜end▁of▁thinking｜>``）：该标签**之前**是思考过程、**之后**才是答案，
       因此整段替换为之后的部分。这一步必须在第 3 步之前做，
       否则思考过程里的示例 JSON 会排在答案前面；
    3. 剥**成对**的思考块（三种已知写法）；
    4. 兜底：出现未知写法的 ``<think>`` 开标签时，若"标签之前"能解析出 JSON
       就只保留之前的部分。

    注意：只有闭合标签的模式**只匹配已知收尾关键字**（end / thinking /
    analysis / reasoning），因此不会误伤正文里合法的 ``<|...|>`` 内容。
    """
    if not text:
        return ""
    result = text.strip()

    # 1) 剥 ```json ... ```
    if result.startswith("```"):
        result = _FENCE_RE.sub("", result, count=1)
        result = _FENCE_RE.sub("", result, count=1)
        result = result.strip()

    # 2) 只有闭合标签：保留标签之后的内容
    close_match = None
    for match in _CLOSE_ONLY_RE.finditer(result):
        close_match = match  # 取最后一个
    if close_match is not None:
        after = result[close_match.end():].strip()
        if after and _has_json(after):
            result = after
        else:
            # 标签之后没有 JSON，说明标签只是正文里的一个标记，删掉即可
            before = result[: close_match.start()].strip()
            result = (before + " " + after).strip() if before and after else (before or after)

    # 3) 剥成对思考块
    for regex in _THINK_REGEXES:
        result = regex.sub("", result).strip()

    # 4) 兜底：未知写法的开标签 → 若标签之前有 JSON，只保留之前的部分
    for tag in _THINK_TAGS:
        if f"<{tag}>" not in result:
            continue
        candidate = result.split(f"<{tag}>")[0].strip()
        if candidate and _has_json(candidate):
            result = candidate

    # 剥常见前缀
    for prefix in ("Output:", "OUTPUT:", "Result:", "Answer:", "JSON:"):
        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    return result


def _scan_json_objects(text: str) -> List[str]:
    """按括号配对扫描出文本中所有**顶层** JSON 对象字面量（按出现顺序）。"""
    candidates: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
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
                    candidate = text[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(payload, Mapping):
                        candidates.append(candidate)
                    start = None
    return candidates


def _load_json_objects(text: str) -> List[Dict[str, Any]]:
    """把 :func:`_scan_json_objects` 的结果解析成 dict 列表。"""
    objects: List[Dict[str, Any]] = []
    for candidate in _scan_json_objects(text):
        try:
            objects.append(json.loads(candidate))
        except json.JSONDecodeError:  # pragma: no cover - 扫描已保证可解析
            continue
    return objects


def extract_json_object(text: str, prefer_last: bool = False) -> Dict[str, Any]:
    """从模型输出中抽出第一个（或最后一个）完整的 JSON 对象。

    实现方式是按括号配对扫描并逐字符尝试 ``json.loads``，这样即使模型在 JSON
    前后夹了说明文字也能正确抽取（比正则更稳）。

    Args:
        text: 模型原始输出。
        prefer_last: 为 True 时返回**最后一个**候选对象。

    Note:
        推理型模型的输出结构是"思考过程 → 答案"，而思考过程里经常出现示例 JSON。
        此时"最后一个对象"才是答案，因此 :func:`parse_augmentation_response`
        会在第一个对象缺少必需字段时改用最后一个。

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

    candidates = _load_json_objects(cleaned)
    if not candidates:
        raise ValueError("模型输出中找不到合法的 JSON 对象")
    return candidates[-1] if prefer_last else candidates[0]


def parse_augmentation_response(
    text: str,
    expected_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """解析并做基础字段检查，返回 ``{uid, string_value, replies}``。

    **JSON 抽取策略（两步）**：

    1. **按 uid 匹配**（``expected_uid`` 非空时）：只接受 ``uid`` 与预期一致、
       且能通过结构校验的候选对象，取**最后一个**满足条件的；
    2. **按位置回退**：没有传 ``expected_uid``、或没有候选 uid 匹配时，
       取**最后一个**能通过校验的候选。

    为什么需要第 1 步：推理型模型的输出是"思考过程 → 答案"，思考过程里经常出现
    **格式完全合法的示例 JSON**；而模型有时还会在答案**后面**再回显一次输入
    （"For reference, the input was: {...}"）。这两种诱饵与真答案在字段上无法
    区分（都有 ``uid``/``string_value``/``replies``，都能过结构校验），
    位置规则（"取最后一个"）只能覆盖前者。**uid 是唯一可靠的判据**：
    :class:`~src.llm.augmentor.Augmentor` 知道它请求的是哪条样本，
    因此把 uid 传进来即可同时挡掉"前面的示例"和"后面的回显"。

    Args:
        text: 模型原始输出。
        expected_uid: 期望的样本 uid；``None`` 表示不做 uid 过滤。

    Raises:
        ValueError: JSON 非法、字段缺失或字段类型不对。
    """
    cleaned = strip_wrappers(text)
    candidates = _load_json_objects(cleaned)

    # 整体就是 JSON 的情况（清洗后可能只剩一个对象，扫描不到时兜底再试一次）
    if not candidates:
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型输出中找不到合法的 JSON 对象：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("模型输出的顶层结构不是 JSON 对象")
        candidates = [dict(payload)]

    if not candidates:
        raise ValueError("模型输出中找不到合法的 JSON 对象")

    # ---- 第 1 步：优先取"uid 匹配 + 校验通过"的最后一个候选 ----
    if expected_uid is not None:
        target = str(expected_uid)
        for payload in reversed(candidates):
            if str(payload.get("uid", "")) != target:
                continue
            try:
                return _validate_augmentation_payload(payload)
            except ValueError:
                continue

    # ---- 第 2 步：不看 uid，从后往前取第一个校验通过的 ----
    last_error: Optional[Exception] = None
    for payload in reversed(candidates):
        try:
            return _validate_augmentation_payload(payload)
        except ValueError as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError("模型输出无有效 JSON 对象")


def _validate_augmentation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """校验单个候选对象是否是合格的增强结果。"""
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

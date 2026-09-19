# coding=utf-8
"""Prompt 编排器：把论文 Fg.3 的四条设计目标落成可执行的 Prompt。

论文 §3.1 要求 Prompt 满足四个目标，本模块逐条对应到模板与代码：

============ ============================================== ==========================
论文目标      含义                                           落地方式
============ ============================================== ==========================
全视角增强    从全局而非局部改写                             约束 C1：整句重构、禁止逐词同义替换
输出格式一致  与原始数据结构保持完全一致                      约束 C2 + :mod:`src.llm.parser` 结构校验
多样性提升    增加表达的多样性与信息丰富度                     约束 C3 + 生成温度/抖动（:class:`PromptBuilder`）
语义一致      语义严格与原始数据一致                          约束 C4 + 语义相似度校验
============ ============================================== ==========================

模板文件（默认 ``configs/prompts/augment_default.txt``）使用 ``{instance_json}``、
``{target_field}``、``{label_descriptions}`` 三个占位符。其中 ``instance_json``
只包含 ``uid / string_value / replies``，与论文 Fg.2 一致——**标签不进入 Prompt**。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from data.processors.data_model import DataInstance

__all__ = [
    "PLACEHOLDER_INSTANCE",
    "PLACEHOLDER_TARGET_FIELD",
    "PLACEHOLDER_LABELS",
    "DEFAULT_TARGET_FIELD",
    "PROMPT_VERSION",
    "PromptSpec",
    "PromptBuilder",
    "load_prompt_template",
]

PLACEHOLDER_INSTANCE = "{instance_json}"
PLACEHOLDER_TARGET_FIELD = "{target_field}"
PLACEHOLDER_LABELS = "{label_descriptions}"

# 论文只改写原帖正文（string_value），回复正文按同样风格改写但结构必须保留
DEFAULT_TARGET_FIELD = "string_value"

# Prompt 版本号：修改模板时同步 +1，使增强缓存键自动失效
PROMPT_VERSION = "v1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PromptSpec:
    """一次 LLM 调用的完整输入。

    Attributes:
        system: system message（角色与硬约束）。
        user: user message（具体实例与指令）。
        target_field: 需要改写的字段名。
        prompt_hash: 模板 + 实例内容 + 版本号的联合哈希，用作增强缓存键。
        template_path: 模板来源，便于追溯"这批数据是用哪个 Prompt 生成的"。
    """

    system: str
    user: str
    target_field: str = DEFAULT_TARGET_FIELD
    prompt_hash: str = ""
    template_path: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_messages(self) -> List[Dict[str, str]]:
        """转成 chat 模板所需的 messages 列表。"""
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user})
        return messages

    def as_prompt_text(self) -> str:
        """转成单段纯文本（供不区分 role 的后端使用）。"""
        parts = []
        if self.system:
            parts.append(self.system)
        parts.append(self.user)
        return "\n\n".join(parts)


def load_prompt_template(path: str) -> str:
    """读取 Prompt 模板文件。

    路径按"绝对路径 → 仓库根相对路径 → 当前工作目录相对路径"依次尝试，
    这样从任意目录运行脚本都能找到 ``configs/prompts/...``。
    """
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(_REPO_ROOT, path))
        candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                return handle.read()
    raise FileNotFoundError(
        f"Prompt 模板不存在：{path}（已尝试 {candidates}）"
    )


class PromptBuilder:
    """把数据实例 + 模板渲染成 :class:`PromptSpec`。

    Args:
        template: 模板正文；也可传 ``template_file`` 由本类读取。
        label_descriptions: 四类标签的自然语言描述（配置项
            ``llm.prompt.label_descriptions``）。**用于让 LLM 判断"哪些内容必须保留"，
            而不是让 LLM 预测标签**——标签始终来自官方 ``label.txt``。
        target_field: 需要改写的字段，默认 ``string_value``。
        max_replies_in_prompt: 送入 Prompt 的最大回复条数，防止超长实例挤爆上下文。
        pretty: 是否把实例 JSON 缩进美化（可读性更好，token 略多）。
    """

    def __init__(
        self,
        template: Optional[str] = None,
        template_file: Optional[str] = None,
        label_descriptions: Optional[Mapping[str, str]] = None,
        target_field: str = DEFAULT_TARGET_FIELD,
        max_replies_in_prompt: int = 30,
        pretty: bool = False,
    ):
        if template is None:
            if template_file is None:
                raise ValueError("必须提供 template 或 template_file 之一")
            template = load_prompt_template(template_file)
        self.template = template
        self.template_file = template_file or ""
        self.label_descriptions = dict(label_descriptions or {})
        self.target_field = target_field
        self.max_replies_in_prompt = max_replies_in_prompt
        self.pretty = pretty

        missing = [
            placeholder
            for placeholder in (PLACEHOLDER_INSTANCE, PLACEHOLDER_LABELS)
            if placeholder not in template
        ]
        if missing:
            raise ValueError(
                f"Prompt 模板缺少必需占位符 {missing}；"
                f"模板需包含 {PLACEHOLDER_INSTANCE} 与 {PLACEHOLDER_LABELS}"
            )

    # ------------------------------------------------------------------ #
    def _format_labels(self) -> str:
        if not self.label_descriptions:
            return "(no class definitions provided)"
        return "\n".join(
            f"- {label}: {desc}" for label, desc in self.label_descriptions.items()
        )

    def _instance_payload(self, instance: DataInstance) -> Dict[str, Any]:
        """构造送进 Prompt 的实例：截断回复数量，不带标签。"""
        payload = instance.prompt_payload()
        replies = payload.get("replies") or []
        if self.max_replies_in_prompt and len(replies) > self.max_replies_in_prompt:
            payload["replies"] = replies[: self.max_replies_in_prompt]
            payload["replies_truncated"] = len(replies) - self.max_replies_in_prompt
        return payload

    # ------------------------------------------------------------------ #
    def build(self, instance: DataInstance, temperature_hint: float = 0.0) -> PromptSpec:
        """渲染出一条 Prompt。

        Args:
            instance: 数据实例。
            temperature_hint: 该条样本的采样温度（用于"多样性提升"目标）。
                仅写入 meta 供后端读取，模板本身不感知温度。
        """
        payload = self._instance_payload(instance)
        instance_json = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if self.pretty else None,
            sort_keys=False,
        )
        rendered = (
            self.template.replace(PLACEHOLDER_INSTANCE, instance_json)
            .replace(PLACEHOLDER_TARGET_FIELD, self.target_field)
            .replace(PLACEHOLDER_LABELS, self._format_labels())
        )

        # 模板首行若以 [ROLE] 开头，则把这一段拆成 system message
        system, user = self._split_roles(rendered)

        digest = hashlib.sha1(
            "\u0000".join(
                [
                    PROMPT_VERSION,
                    self.template,
                    instance_json,
                    self.target_field,
                    f"{temperature_hint:.3f}",
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]

        return PromptSpec(
            system=system,
            user=user,
            target_field=self.target_field,
            prompt_hash=digest,
            template_path=self.template_file,
            meta={"temperature_hint": temperature_hint, "uid": instance.uid},
        )

    @staticmethod
    def _split_roles(rendered: str) -> Tuple[str, str]:
        """把渲染结果切成 ``(system, user)``。

        约定：``[ROLE] ... `` 到 ``[INPUT DATA]`` 之前属于 system（角色 + 约束），
        其余属于 user（实例 + 输出要求）。找不到标记时整段作为 user。
        """
        marker = "[INPUT DATA]"
        role_marker = "[ROLE]"
        if marker in rendered and role_marker in rendered:
            index = rendered.index(marker)
            return rendered[:index].strip(), rendered[index:].strip()
        return "", rendered.strip()

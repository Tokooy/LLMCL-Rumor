# coding=utf-8
"""LLMCL-Rumor: LLM 增强对比学习的社交网络虚假信息检测。

复现自《基于LLM增强对比学习的社交网络虚假信息检测方法》(2025-01-22)。

本包按职责划分为四层，依赖方向自上而下单向流动：

    src.models   -> 只依赖 src.utils
    src.llm      -> 只依赖 src.utils
    src.training -> 依赖 src.models / src.llm / src.utils
    (scripts)    -> 依赖以上全部

各子包用途见 docs/project_structure.md。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

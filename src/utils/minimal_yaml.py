# coding=utf-8
"""极简 YAML 子集解析器：用于在**没有 pyyaml** 的环境下校验配置文件。

为什么需要它
------------
``scripts/verify_pipeline.py`` 想在只装了标准库的机器上验证"六个 Proposed 变体的
配置是否与论文 Table 2 一致"。虽然配置本身是 YAML，但本项目用到的语法非常窄，
完全可以不依赖 pyyaml 解析：

* 嵌套映射（靠缩进）
* 空列表项 ``- ../base.yaml``（字符串列表）
* 行内流式列表 ``[a, b, c]`` 与行内流式映射 ``{}`` / ``{a: 1}``
* 标量：整数、浮点、布尔、``null``、``""``、引号字符串、普通字符串
* 以 ``#`` 开头的注释与行尾注释

**它不支持的**（本项目配置里也没用）：多行字符串 ``|``/``>``、
锚点与别名 ``&``/``*``、复杂键 ``?``、重复键合并 ``<<``。

> 流式映射必须支持：``configs/base.yaml`` 里就有 ``extra_body: {}``，
> 早期版本把 ``{}`` 当普通字符串返回，导致 ``dict(cfg.get_path("llm.api.extra_body"))``
> 抛 ``ValueError``——而这只在**没有 pyyaml** 的机器上出现（pyyaml 会正确解析成空字典），
> 属于典型的"回退实现悄悄跑偏"。

正确的用法是"有 pyyaml 就用 pyyaml"
------------------------------------
:func:`load_yaml` 会优先使用 pyyaml；只有在 pyyaml 缺失时才回退到本解析器，
并通过返回值告诉调用方用的是哪条路径。这样在有 pyyaml 的机器上，
本解析器的行为可以被真实 pyyaml 直接比对验证，而不会成为"悄悄跑偏的第二实现"。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["load_yaml", "simple_yaml_load", "pyyaml_available"]


def pyyaml_available() -> bool:
    """返回环境中是否安装了 pyyaml。"""
    try:
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


def _strip_comment(line: str) -> str:
    """去掉行尾注释；引号内的 ``#`` 不算注释。"""
    result: List[str] = []
    quote: Optional[str] = None
    for index, char in enumerate(line):
        if quote:
            result.append(char)
            if char == quote and (index == 0 or line[index - 1] != "\\"):
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            result.append(char)
            continue
        if char == "#":
            break
        result.append(char)
    return "".join(result).rstrip()


def _parse_scalar(text: str) -> Any:
    """把标量文本转成 Python 对象。"""
    value = text.strip()
    if value == "":
        return ""
    if value in ("null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE", "yes", "Yes"):
        return True
    if value in ("false", "False", "FALSE", "no", "No"):
        return False

    # 引号字符串
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]

    # 流式列表
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in _split_flow(inner)]

    # 流式映射（如 `{}` 或 `{a: 1, b: 2}`）
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        mapping: Dict[str, Any] = {}
        for item in _split_flow(inner):
            key, _, item_value = item.partition(":")
            mapping[key.strip().strip("'\"")] = _parse_scalar(item_value)
        return mapping

    # 数字
    try:
        if any(char in value for char in (".", "e", "E")) and not value.isdigit():
            return float(value)
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_flow(text: str) -> List[str]:
    """按逗号切分流式列表，注意不要切到引号内部。"""
    items: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    depth = 0
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())
    return [item for item in items if item != ""]


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def simple_yaml_load(path: str) -> Dict[str, Any]:
    """用极简解析器读取 YAML 文件，返回普通 dict。

    仅支持本项目的配置语法（见模块 docstring）。遇到无法理解的缩进结构时
    不会静默给出错误结果，而是抛出 :class:`ValueError` 说明行号。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw_lines = handle.read().splitlines()

    # 预处理：去掉注释与空行，但保留缩进信息
    entries: List[Tuple[int, str, int]] = []   # (缩进, 内容, 原始行号)
    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = _strip_comment(line)
        if not stripped.strip():
            continue
        entries.append((_indent_of(stripped), stripped.strip(), lineno))

    if not entries:
        return {}

    root: Dict[str, Any] = {}
    # 栈：[(缩进, 对应的容器)]
    stack: List[Tuple[int, Any]] = [(-1, root)]

    index = 0
    while index < len(entries):
        indent, content, lineno = entries[index]

        # 弹出比当前缩进更深的容器
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{path}:{lineno} 缩进结构异常，无法定位父级容器")
        container = stack[-1][1]

        if content.startswith("- "):
            # 列表项
            if not isinstance(container, list):
                raise ValueError(f"{path}:{lineno} 列表项出现在非列表容器中")
            item_text = content[2:].strip()
            if item_text == "":
                # 嵌套列表/映射的父节点：用占位容器承接后续更深缩进的行
                child: Any = {}
                container.append(child)
                stack.append((indent, child))
            elif ":" in item_text and not item_text.startswith(("'", '"')):
                # 列表中的行内映射，例如 "- title: xxx"
                key, _, value_text = item_text.partition(":")
                item: Dict[str, Any] = {key.strip(): _parse_scalar(value_text)}
                container.append(item)
                stack.append((indent, item))
            else:
                container.append(_parse_scalar(item_text))
            index += 1
            continue

        # 映射项
        if ":" not in content:
            raise ValueError(f"{path}:{lineno} 既不是列表项也没有 ':'，无法解析：{content!r}")
        key, _, value_text = content.partition(":")
        key = key.strip().strip("'\"")
        value_text = value_text.strip()
        if not isinstance(container, dict):
            raise ValueError(f"{path}:{lineno} 映射项出现在列表容器中")

        if value_text == "":
            # 可能是空标量，也可能是下一层容器的父节点——看向后一行的缩进
            next_is_child = False
            if index + 1 < len(entries):
                next_indent, next_content, _ = entries[index + 1]
                if next_indent > indent:
                    next_is_child = True
                    child = [] if next_content.startswith("- ") else {}
            if next_is_child:
                container[key] = child
                stack.append((indent, child))
            else:
                container[key] = None
        else:
            container[key] = _parse_scalar(value_text)
        index += 1

    return root


def load_yaml(path: str) -> Tuple[Dict[str, Any], str]:
    """读取 YAML 文件。

    Returns:
        ``(解析结果, 使用的解析器名)``，解析器名为 ``"pyyaml"`` 或 ``"minimal"``。
        调用方可以据此在日志里说明"这次校验用的是哪一个实现"。
    """
    if pyyaml_available():
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return (payload if isinstance(payload, dict) else {}), "pyyaml"
    return simple_yaml_load(path), "minimal"

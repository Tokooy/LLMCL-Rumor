# coding=utf-8
"""配置系统：YAML -> 属性可访问的嵌套对象，支持实验配置覆盖基础配置。

设计目标（与常见深度学习项目一致）：

1. ``configs/base.yaml`` 描述"默认超参"；
2. ``configs/experiments/proposed*.yaml`` 只写"与默认值的差异"；
3. ``configs/llm_qwen7b.yaml`` 之类描述"换基座要改的部分"，一个实验配置可以
   通过 ``defaults`` 字段同时继承多份配置，后写的覆盖先写的。

用法::

    cfg = load_config("configs/base.yaml", "configs/experiments/proposed-3.yaml")
    cfg.training.cl.epochs          # 属性访问
    cfg["training"]["cl"]["epochs"] # 字典访问
    to_dict(cfg)                    # 还原成普通 dict（便于写日志 / json）

本模块只依赖标准库与 ``yaml``，不依赖 torch。
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["Config", "load_config", "merge_dict", "to_dict", "deep_update"]


class Config(dict):
    """嵌套 dict 的薄封装，把嵌套的 dict 自动包装成 Config 以支持属性访问。

    ``Config`` 是 ``dict`` 的子类，因此可以被 ``json.dumps`` 直接序列化，
    也可以和普通 dict 混用。
    """

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #
    def __init__(self, data: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__()
        merged: Dict[str, Any] = {}
        if data is not None:
            if not isinstance(data, Mapping):
                raise TypeError(
                    f"Config 只接受 Mapping，收到 {type(data).__name__}"
                )
            merged.update(data)
        merged.update(kwargs)
        for key, value in merged.items():
            super().__setitem__(key, self._wrap(value))

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @classmethod
    def _wrap(cls, value: Any) -> Any:
        """递归把内层 Mapping 包装成 Config，其余类型原样返回。"""
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        """递归把 Config 还原成普通 dict（用于写入 yaml/json）。"""
        if isinstance(value, Config):
            return {key: Config._unwrap(item) for key, item in value.items()}
        if isinstance(value, Mapping):
            return {key: Config._unwrap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Config._unwrap(item) for item in value]
        return value

    # ------------------------------------------------------------------ #
    # 访问
    # ------------------------------------------------------------------ #
    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - 防御式分支
            raise AttributeError(
                f"配置项 {item!r} 不存在；当前可用键：{sorted(self.keys())}"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #
    def get_path(self, dotted: str, default: Any = None) -> Any:
        """按 ``a.b.c`` 取嵌套值，缺失时返回 ``default``。"""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """按 ``a.b.c`` 写嵌套值，中间层缺失时自动创建 Config。"""
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if not isinstance(node.get(part), Config):
                node[part] = Config()
            node = node[part]  # type: ignore[assignment]
        node[parts[-1]] = value

    def update_from(self, other: Optional[Mapping[str, Any]]) -> "Config":
        """就地递归合并另一个映射，返回 self（便于链式调用）。"""
        if other:
            deep_update(self, other)
        return self

    def clone(self) -> "Config":
        """深拷贝，避免实验之间互相污染。"""
        return Config(copy.deepcopy(self._unwrap(self)))

    def to_dict(self) -> Dict[str, Any]:
        """还原成普通 dict。"""
        return self._unwrap(self)  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return f"Config({self.to_dict()!r})"


# ---------------------------------------------------------------------- #
# 合并逻辑
# ---------------------------------------------------------------------- #
def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并：``override`` 中的 dict 与 ``base`` 同键 dict 深度合并，
    其余类型直接覆盖；list 视为整体替换（不逐元素合并）。"""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = Config._unwrap(value)
    return base


def merge_dict(*dicts: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """按顺序深度合并多个映射，返回新的普通 dict。"""
    result: Dict[str, Any] = {}
    for item in dicts:
        if item:
            deep_update(result, item)
    return result


def to_dict(config: Any) -> Any:
    """把 Config / 嵌套结构转成纯 Python 内建类型。"""
    if isinstance(config, Config):
        return config.to_dict()
    return Config._unwrap(config)


# ---------------------------------------------------------------------- #
# 加载逻辑
# ---------------------------------------------------------------------- #
def _read_yaml(path: str) -> Dict[str, Any]:
    """读取单个 YAML 文件；文件不存在或为空时返回空 dict。

    优先使用 pyyaml；未安装时回退到 :mod:`src.utils.minimal_yaml` 的极简解析器
    （本项目配置用到的语法很窄，回退实现足以覆盖）。回退时会在 stderr 打印一行
    提示，避免"用了另一个解析器"这件事被悄悄忽略。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")

    try:
        import yaml  # 延迟导入：仅此函数需要 pyyaml
    except ImportError:
        yaml = None

    if yaml is not None:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"{path} 的顶层结构必须是映射（mapping）")
        return Config._unwrap(raw)  # type: ignore[return-value]

    from .minimal_yaml import simple_yaml_load
    from .logger import get_logger

    get_logger("config").warning(
        f"未安装 pyyaml，改用内置极简解析器读取 {path}；"
        "如需与官方 YAML 语义完全一致，请执行 pip install pyyaml"
    )
    raw = simple_yaml_load(path)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} 的顶层结构必须是映射（mapping）")
    return Config._unwrap(raw)  # type: ignore[return-value]


def _resolve_defaults(
    path: str,
    _seen: Optional[Sequence[str]] = None,
    _stack: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """递归解析 ``defaults:`` 字段。

    约定：``defaults`` 中的路径相对当前文件所在目录解析；
    最终合并顺序为「先 defaults（按书写顺序），后文件自身」，即文件自身优先级最高。

    ``_seen`` 用于检测循环继承；``_stack`` 用于记录"这份配置由哪几个文件合成"，
    结果会被写进 ``_config_sources``（排查"这个值到底来自哪份配置"时很有用）。
    """
    seen = list(_seen or [])
    real = os.path.abspath(path)
    if real in seen:
        chain = " -> ".join(seen + [real])
        raise ValueError(f"检测到配置的循环 defaults 继承：{chain}")
    seen.append(real)
    if _stack is not None:
        _stack.append(real)

    raw = _read_yaml(real)
    defaults = raw.pop("defaults", None)

    base: Dict[str, Any] = {}
    if defaults:
        if isinstance(defaults, str):
            defaults = [defaults]
        if not isinstance(defaults, Iterable):
            raise TypeError(f"{path} 的 defaults 必须是字符串或字符串列表")
        here = os.path.dirname(real)
        for entry in defaults:
            entry_path = entry if os.path.isabs(entry) else os.path.join(here, entry)
            deep_update(base, _resolve_defaults(entry_path, seen, _stack))

    deep_update(base, raw)
    return base


def load_config(*paths: str, strict: bool = True) -> Config:
    """按顺序加载并深度合并若干 YAML 配置文件。

    Args:
        *paths: 配置文件路径，相对路径按当前工作目录解析。后一个覆盖前一个。
        strict: 为 True 时，任一文件缺失即抛错；为 False 时跳过缺失文件。

    Returns:
        属性可访问的 :class:`Config`。

    每份配置的 ``defaults`` 字段都会被先展开，因此可以写成
    ``defaults: [../base.yaml]`` 来继承公共超参。
    """
    if not paths:
        return Config()

    merged: Dict[str, Any] = {}
    sources: List[str] = []
    for path in paths:
        if not os.path.isfile(path):
            if strict:
                raise FileNotFoundError(f"配置文件不存在：{path}")
            continue
        stack: List[str] = []
        part = _resolve_defaults(path, _seen=None, _stack=stack)
        part.pop("_config_sources", None)
        deep_update(merged, part)
        sources.extend(stack)

    # 去重并保持顺序，便于排查"这个值到底来自哪份配置"
    unique_sources: List[str] = []
    for item in sources:
        if item not in unique_sources:
            unique_sources.append(item)
    merged["_config_sources"] = unique_sources
    return Config(merged)

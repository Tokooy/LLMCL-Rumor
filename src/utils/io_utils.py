# coding=utf-8
"""JSONL / JSON 读写与目录管理。

论文的数据流水线涉及四个不断累积的文件，全部采用 **JSONL**（每行一个 JSON 对象）：

* ``data/processed/<dataset>/{train,dev,test}.jsonl``  原始数据
* ``data/processed/<dataset>/augmented_round{k}.jsonl`` 第 k 轮 LLM 增强数据
* ``outputs/results/...``                                评估指标

之所以选 JSONL 而不是一个大 JSON：追加增强样本时不必重写整个文件，
单行损坏也不会污染整个数据集。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Optional

__all__ = [
    "ensure_dir",
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "iter_jsonl",
    "load_json",
    "save_json",
    "count_lines",
]


def ensure_dir(path: str) -> str:
    """确保目录存在（递归创建），返回该目录路径。"""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def iter_jsonl(path: str, skip_empty: bool = True) -> Iterator[Dict[str, Any]]:
    """逐行流式读取 JSONL，适合大文件（增强后的数据集可能几万行）。

    Args:
        path: 文件路径。
        skip_empty: 跳过空行；为 False 时空行会触发解析错误。

    Yields:
        每行的 dict。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL 文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                if skip_empty:
                    continue
                raise ValueError(f"{path}:{lineno} 是空行")
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败：{exc}") from exc


def read_jsonl(path: str, skip_empty: bool = True) -> List[Dict[str, Any]]:
    """一次性读入 JSONL，返回 list。"""
    return list(iter_jsonl(path, skip_empty=skip_empty))


def write_jsonl(
    path: str,
    records: Iterable[Dict[str, Any]],
    append: bool = False,
    ensure_ascii: bool = False,
) -> int:
    """写入 JSONL。

    Args:
        path: 目标文件。
        records: 可迭代的 dict 序列。
        append: True 时追加写；False 时覆盖写（且为原子写）。
        ensure_ascii: 默认 False（保留中文原文，便于人工核对增强质量）。

    Returns:
        写入的行数。
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    mode = "a" if append else "w"
    count = 0

    if append:
        with open(path, mode, encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
                count += 1
        return count

    # 原子写：先写临时文件再替换，避免中断产生半个数据集
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
                count += 1
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return count


def append_jsonl(path: str, records: Iterable[Dict[str, Any]], ensure_ascii: bool = False) -> int:
    """追加写入 JSONL，返回写入行数。"""
    return write_jsonl(path, records, append=True, ensure_ascii=ensure_ascii)


def load_json(path: str, default: Optional[Any] = None) -> Any:
    """读取单个 JSON 文件；文件不存在时返回 ``default``。"""
    if not os.path.isfile(path):
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON 文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
    """原子写入单个 JSON 文件，返回文件路径。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def count_lines(path: str) -> int:
    """统计文件行数（不解析内容），用于日志里报告数据集规模。"""
    if not os.path.isfile(path):
        return 0
    total = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total

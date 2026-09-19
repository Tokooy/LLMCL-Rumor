# coding=utf-8
"""LoRA 微调与任务向量导出（论文 §3.3 的"自举式数据增强微调"）。

论文的做法
----------
""考虑到公开数据集的数据数量难以满足 LLM 微调的需求，我们设计了自举式数据增强
微调方法。该方法通过收集 LLM 在训练过程中生成的增强数据，将其作为新的微调样本
来微调 LLM，从而持续提升 LLM 在数据增强任务上的表现。""

因此微调样本**不是**外部标注数据，而是 LLM 自己产出的增强数据（bootstrap）。
本模块负责把增强数据组装成"指令 → JSON 输出"的训练对：

* ``prompt``：与增强时**完全相同**的 Prompt（复用 :class:`PromptBuilder`），
  保证微调目标与推理时的输入分布一致；
* ``completion``：原始样本（而非增强结果）的规范化 JSON。

为什么目标用"原始样本"而不是"增强结果"？
    若让模型回归自己的增强输出，等于自我蒸馏、会把已有的多样性抹平；
    而让模型学习"给定实例 → 稳定复现该实例的结构化表示"，
    它学到的是"如何按约束输出结构正确的 JSON"，这正是增强成败的关键
    （论文 §4.3 把 FR 类别的退化归因于"LLM 无法独立判断真实性"，
    而不是格式问题）。这一选择在 ``docs/implementation_notes.md`` 中有完整论证。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from data.processors.data_model import DataInstance
from src.utils.io_utils import ensure_dir, read_jsonl, write_jsonl

from .prompts import PromptBuilder
from .task_vector import TaskVector, subtract_parameters

__all__ = [
    "build_finetune_record",
    "build_finetune_records",
    "export_task_vector",
    "describe_task_vector",
    "finetune_and_export",
    "save_task_vector",
    "load_task_vector",
]


# ---------------------------------------------------------------------- #
# 微调样本构造
# ---------------------------------------------------------------------- #
def build_finetune_record(
    instance: DataInstance,
    prompt_builder: PromptBuilder,
    target: str = "original",
) -> Dict[str, Any]:
    """构造一条微调样本 ``{"prompt", "completion", "uid"}``。

    Args:
        instance: 数据实例（原样本）。
        prompt_builder: 与增强阶段共用的 Prompt 编排器。
        target: ``original`` 时 completion 为**原样本**的规范化 JSON
            （默认，见模块 docstring 的论证）；``self`` 时用增强样本自身
            （自我蒸馏对照）。

    Returns:
        ``{"prompt": str, "completion": str, "uid": str}``。
    """
    spec = prompt_builder.build(instance)
    payload = {
        "uid": instance.uid,
        "string_value": instance.string_value,
        "replies": [reply.to_record() for reply in instance.replies],
    }
    return {
        "uid": instance.uid,
        "prompt": spec.as_prompt_text(),
        "completion": json.dumps(payload, ensure_ascii=False),
    }


def build_finetune_records(
    originals: Sequence[DataInstance],
    augmented: Sequence[DataInstance],
    prompt_builder: PromptBuilder,
    target: str = "original",
    max_records: int = 0,
) -> List[Dict[str, Any]]:
    """组装一轮微调所需的全部样本。

    Args:
        originals: 原样本（用于取 Prompt 与标签）。
        augmented: 同一批样本的增强结果；只用于确认"这些样本确实完成过增强"。
        prompt_builder: Prompt 编排器。
        target: 见 :func:`build_finetune_record`。
        max_records: ``>0`` 时截断到该数量。

    Returns:
        微调样本列表。若 ``augmented`` 为空则返回空列表
        （没有增强数据就没有可自举微调的样本，论文亦如此）。
    """
    if not augmented:
        return []

    augmented_uids = {item.original_uid for item in augmented}
    candidates = [item for item in originals if item.uid in augmented_uids]
    if not candidates:
        candidates = list(originals)

    if target == "self":
        # 自我蒸馏对照：completion 用增强样本自身的 JSON
        augmented_by_uid: Dict[str, DataInstance] = {}
        for item in augmented:
            augmented_by_uid.setdefault(item.original_uid, item)
        records = []
        for instance in candidates:
            reference = augmented_by_uid.get(instance.uid, instance)
            spec = prompt_builder.build(instance)
            payload = {
                "uid": reference.uid,
                "string_value": reference.string_value,
                "replies": [reply.to_record() for reply in reference.replies],
            }
            records.append(
                {
                    "uid": instance.uid,
                    "prompt": spec.as_prompt_text(),
                    "completion": json.dumps(payload, ensure_ascii=False),
                }
            )
    else:
        records = [build_finetune_record(item, prompt_builder, target="original") for item in candidates]

    if max_records and max_records > 0:
        records = records[:max_records]
    return records


# ---------------------------------------------------------------------- #
# 任务向量
# ---------------------------------------------------------------------- #
def export_task_vector(backend: Any) -> Optional[Dict[str, Any]]:
    """从后端导出 ``τ = θ_ft - θ_base``。

    ``transformers`` 后端在注入 LoRA 时会快照初始参数，因此这里直接调用其
    :meth:`export_task_vector`；对不支持的后端返回 ``None`` 并给出明确日志。
    """
    if not getattr(backend, "supports_task_vector", False):
        return None
    vector = backend.export_task_vector()
    return dict(vector) if vector else None


def describe_task_vector(vector: Mapping[str, Any]) -> Dict[str, Any]:
    """统计任务向量的规模与稀疏度，便于日志核对。"""
    total = 0
    nonzero = 0
    for value in vector.values():
        total += int(value.numel())
        nonzero += int((value != 0).sum().item())
    return {
        "num_parameters": len(vector),
        "total_entries": total,
        "nonzero_entries": nonzero,
        "sparsity": round(1.0 - (nonzero / total), 6) if total else 0.0,
    }


def save_task_vector(path: str, vector: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> str:
    """保存任务向量（``torch.save``，附带元信息）。"""
    import torch

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    payload = {
        "vector": {name: value.detach().cpu() for name, value in vector.items()},
        "meta": dict(meta or {}),
        "stats": describe_task_vector(vector),
    }
    torch.save(payload, path)
    return path


def load_task_vector(path: str) -> Dict[str, Any]:
    """读取任务向量，返回 ``{"vector", "meta", "stats"}``。"""
    import torch

    payload = torch.load(path, map_location="cpu")
    if "vector" not in payload:
        # 兼容直接保存 {name: tensor} 的情况
        return {"vector": payload, "meta": {}, "stats": describe_task_vector(payload)}
    return payload


# ---------------------------------------------------------------------- #
# 一步完成：微调 + 导出
# ---------------------------------------------------------------------- #
def finetune_and_export(
    backend: Any,
    records: Sequence[Mapping[str, Any]],
    output_dir: str,
    base_vector: Optional[Mapping[str, Any]] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """执行一次微调并导出任务向量。

    ``base_vector`` 用于"累计差分"：TIES-Merging 要求每个任务向量都相对**同一基座**
    ``θ_0``。若后端每次微调前不会重置参数，就必须减去上一次的累计结果，
    否则任务向量会被重复累加。

    Args:
        backend: 支持微调的后端。
        records: 微调样本（来自 :func:`build_finetune_records`）。
        output_dir: LoRA 适配器保存目录。
        base_vector: 上一次导出的任务向量；非空时本次导出结果会减去它。
        logger: 可选 logger。

    Returns:
        ``{"path", "vector", "stats", "num_records"}``；后端不支持微调时
        ``vector`` 为 ``None``。
    """
    def _log(message: str, level: str = "info") -> None:
        if logger is not None:
            getattr(logger, level, logger.info)(message)

    if not getattr(backend, "supports_finetuning", False):
        _log(
            f"{backend.name} 后端不支持梯度微调，跳过本轮 LLM 微调"
            "（论文 Proposed-1/2/3 的 M=0 设置不受影响）",
            level="warning",
        )
        return {"path": None, "vector": None, "stats": {}, "num_records": 0}

    if not records:
        _log("没有可用的自举微调样本（增强数据为空），跳过本轮微调", level="warning")
        return {"path": None, "vector": None, "stats": {}, "num_records": 0}

    adapter_dir = backend.finetune(records, output_dir)
    _log(f"LoRA 微调完成：{len(records)} 条自举样本，适配器保存于 {adapter_dir}")

    vector = export_task_vector(backend)
    if vector is None:
        return {"path": adapter_dir, "vector": None, "stats": {}, "num_records": len(records)}

    if base_vector:
        # 累计差分：确保所有任务向量都相对同一个 θ_0
        vector = subtract_parameters(vector, base_vector)
        _log("已扣除上一次的累计任务向量，保证所有 τ 都相对同一基座 θ_0")

    stats = describe_task_vector(vector)
    _log(
        f"任务向量：{stats['num_parameters']} 个参数、{stats['total_entries']} 项、"
        f"稀疏度 {stats['sparsity']:.4f}"
    )
    return {"path": adapter_dir, "vector": vector, "stats": stats, "num_records": len(records)}


def register_task_vector(
    task_vector: TaskVector,
    vector: Mapping[str, Any],
    weight: float,
    round_index: int,
) -> TaskVector:
    """把导出的任务向量登记进 :class:`TaskVector` 容器。"""
    task_vector.append(vector, weight=weight, round_index=round_index)
    return task_vector


def dump_records(path: str, records: Iterable[Mapping[str, Any]]) -> int:
    """把微调样本落盘（便于核查自举数据），返回条数。"""
    return write_jsonl(path, records)


def read_records(path: str) -> List[Dict[str, Any]]:
    """读取微调样本。"""
    return read_jsonl(path)

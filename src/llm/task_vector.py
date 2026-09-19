# coding=utf-8
"""任务向量（论文 Algorithm 1 的输入 ``τ``，即文献[35] 的 task vector）。

定义（TIES-Merging, Yadav et al. 2024）::

    τ_t = θ_t - θ_0

其中 ``θ_0`` 是基座参数，``θ_t`` 是在某个任务/轮次上微调后的参数。
论文 Algorithm 1 的输入里有 ``τ1 … τm`` 多个任务向量，正是"多轮微调各得到一个 τ"。

本仓库的实现选择
----------------
微调走 LoRA（论文 §3.3 提到 PEFT/LoRA 是主流微调方式之一），因此任务向量
**以 LoRA 参数为粒度**：``τ = {name: A_ft - A_init, ...}``，只覆盖 ``lora_A`` /
``lora_B`` 这些可训练矩阵。好处：

* 13B 模型的任务向量只有几十 MB，合并时不需要把两份 13B 权重同时读进显存；
* 基座权重始终冻结，天然满足 TIES-Merging"从同一个 θ_0 出发"的前提。

对"全参数微调"路径同样适用：只要传入的是 ``θ_ft - θ_0`` 的逐参数差值即可，
本模块的算子对键名与形状不做任何假设。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "TaskVector",
    "subtract_parameters",
    "add_task_vectors",
    "scale_task_vector",
]


class TaskVector:
    """一批任务向量的容器，按"轮次"组织。

    之所以做成容器而不是裸 dict：论文 Algorithm 1 需要在**多个**任务向量之间
    做逐参数的符号选举与不相交合并，容器负责保证所有向量键集合一致、
    形状一致，避免合并时出现"某个参数缺一个向量"的静默错误。

    Args:
        vectors: ``[{参数名: 张量}, ...]``，按轮次从早到晚排列。
        weights: 与 ``vectors`` 等长的权重列表（论文式(6)的 ``ω_m``）；
            默认全 1。
        rounds: 与 ``vectors`` 等长的轮次编号；默认 1..M。
    """

    def __init__(
        self,
        vectors: Optional[Sequence[Mapping[str, Any]]] = None,
        weights: Optional[Sequence[float]] = None,
        rounds: Optional[Sequence[int]] = None,
    ):
        self.vectors: List[Dict[str, Any]] = [dict(item) for item in (vectors or [])]
        count = len(self.vectors)
        self.weights: List[float] = (
            [float(value) for value in weights] if weights is not None else [1.0] * count
        )
        self.rounds: List[int] = (
            [int(value) for value in rounds] if rounds is not None
            else list(range(1, count + 1))
        )
        if len(self.weights) != count:
            raise ValueError(
                f"weights 长度 {len(self.weights)} 与向量数量 {count} 不一致"
            )
        if len(self.rounds) != count:
            raise ValueError(
                f"rounds 长度 {len(self.rounds)} 与向量数量 {count} 不一致"
            )
        if count:
            self._check_consistency()

    # ------------------------------------------------------------------ #
    def _check_consistency(self) -> None:
        """校验所有任务向量的键集合与形状一致。"""
        reference = self.vectors[0]
        reference_keys = set(reference.keys())
        for index, vector in enumerate(self.vectors[1:], start=1):
            keys = set(vector.keys())
            if keys != reference_keys:
                missing = sorted(reference_keys - keys)
                extra = sorted(keys - reference_keys)
                raise ValueError(
                    f"第 {index + 1} 个任务向量的参数集合与第 1 个不一致："
                    f"缺少 {missing[:5]}，多出 {extra[:5]}"
                )
            for name in reference_keys:
                if tuple(vector[name].shape) != tuple(reference[name].shape):
                    raise ValueError(
                        f"参数 {name} 的形状不一致："
                        f"{tuple(reference[name].shape)} vs {tuple(vector[name].shape)}"
                    )

    # ------------------------------------------------------------------ #
    @property
    def num_vectors(self) -> int:
        return len(self.vectors)

    @property
    def parameter_names(self) -> List[str]:
        return list(self.vectors[0].keys()) if self.vectors else []

    def append(self, vector: Mapping[str, Any], weight: float = 1.0, round_index: Optional[int] = None) -> None:
        """追加一个任务向量（新一轮微调的结果）。

        Args:
            vector: 参数名到张量的映射。
            weight: 该向量的权重 ω（由 CL 损失决定，见论文式(9)）。
            round_index: 轮次编号；默认取"上一个轮次 + 1"。
        """
        if self.vectors:
            reference_keys = set(self.vectors[0].keys())
            keys = set(vector.keys())
            if keys != reference_keys:
                raise ValueError(
                    f"追加的任务向量参数集合不一致：缺少 {sorted(reference_keys - keys)[:5]}，"
                    f"多出 {sorted(keys - reference_keys)[:5]}"
                )
        self.vectors.append(dict(vector))
        self.weights.append(float(weight))
        self.rounds.append(
            int(round_index) if round_index is not None else (self.rounds[-1] + 1 if self.rounds else 1)
        )

    def latest(self) -> Optional[Dict[str, Any]]:
        """最近一个任务向量。"""
        return self.vectors[-1] if self.vectors else None

    def subset(self, indices: Iterable[int]) -> "TaskVector":
        """取子集（用于"只合并最近 k 轮"这类消融）。"""
        indices = list(indices)
        return TaskVector(
            vectors=[self.vectors[index] for index in indices],
            weights=[self.weights[index] for index in indices],
            rounds=[self.rounds[index] for index in indices],
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化元信息（不含张量本身，张量请用 torch.save）。"""
        return {
            "num_vectors": self.num_vectors,
            "weights": list(self.weights),
            "rounds": list(self.rounds),
            "parameter_names": self.parameter_names,
        }

    def __len__(self) -> int:
        return self.num_vectors

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"TaskVector(vectors={self.num_vectors}, weights={self.weights}, "
            f"rounds={self.rounds})"
        )


# ---------------------------------------------------------------------- #
# 逐参数算子（对 torch 无强依赖，只要求张量支持 - + * 与逐元素比较）
# ---------------------------------------------------------------------- #
def subtract_parameters(finetuned: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    """计算 ``τ = θ_ft - θ_0``。

    Args:
        finetuned: 微调后参数。
        base: 基座参数。

    Raises:
        KeyError: ``finetuned`` 中存在基座没有的参数（例如新增了模块）。
    """
    vector: Dict[str, Any] = {}
    for name, value in finetuned.items():
        if name not in base:
            raise KeyError(
                f"参数 {name} 在基座中不存在，无法计算任务向量；"
                "请确认微调前后结构一致（LoRA 场景下应只比较可训练参数）"
            )
        vector[name] = value - base[name]
    return vector


def add_task_vectors(left: Mapping[str, Any], right: Mapping[str, Any], scale: float = 1.0) -> Dict[str, Any]:
    """``left + scale * right``（逐参数）。"""
    result: Dict[str, Any] = {}
    for name in left:
        if name in right:
            result[name] = left[name] + right[name] * scale
        else:
            result[name] = left[name]
    return result


def scale_task_vector(vector: Mapping[str, Any], scale: float) -> Dict[str, Any]:
    """``scale * vector``（逐参数）。"""
    return {name: value * scale for name, value in vector.items()}

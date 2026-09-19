# coding=utf-8
"""LLM 后端抽象接口。

论文里 LLM 承担两个角色：

1. **数据增强器**（§3.1）：由 Prompt 编排器驱动，生成语义一致、表达多样的样本；
2. **被微调对象**（§3.3）：用 CL 的对比损失指导 LoRA 微调，再用 TIES-Merging 合并。

因此接口分成三组能力，不同后端按能力实现：

====================== ========== ============ ============
能力                    transformers  api         demo
====================== ========== ============ ============
``generate``            ✅          ✅           ✅（伪增强）
``finetune``            ✅          ❌           ❌
``export_task_vector``  ✅          ❌           ❌
====================== ========== ============ ============

不具备的能力默认抛出 :class:`NotImplementedError`，并由
:meth:`LLMBackend.supports_finetuning` 提前声明，调用方（如
:class:`src.training.joint_trainer.JointAlignmentTrainer`）据此**跳过**微调阶段
而不是崩溃——这样 API 后端也能跑通 Proposed-1/2/3（M=0）的实验。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .prompts import PromptSpec

__all__ = ["GenerationResult", "LLMBackend", "TaskVectorLike"]


class GenerationResult:
    """一条生成结果。

    Attributes:
        text: 模型原始输出文本。
        uid: 对应样本 uid（便于回溯）。
        prompt_hash: Prompt 哈希（缓存键）。
        error: 生成失败时的错误信息；成功为 ``None``。
        meta: 后端附加信息（耗时、token 数等）。
    """

    __slots__ = ("text", "uid", "prompt_hash", "error", "meta")

    def __init__(
        self,
        text: str = "",
        uid: str = "",
        prompt_hash: str = "",
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.uid = uid
        self.prompt_hash = prompt_hash
        self.error = error
        self.meta = meta or {}

    @property
    def ok(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        status = "ok" if self.ok else f"error={self.error!r}"
        return f"GenerationResult(uid={self.uid!r}, {status}, len={len(self.text)})"


# 任务向量用任何 Mapping[str, Any] 表示（键为参数名），避免本模块依赖 torch
TaskVectorLike = Mapping[str, Any]


class LLMBackend(ABC):
    """LLM 后端基类。

    子类必须实现 :meth:`generate`；微调相关方法按需实现。
    """

    #: 后端名字，用于日志与结果记录
    name: str = "base"

    #: 该后端是否支持梯度微调（LoRA）
    supports_finetuning: bool = False

    #: 该后端是否支持导出任务向量（TIES-Merging 的前提）
    supports_task_vector: bool = False

    def __init__(self, model_name: str = "", **kwargs: Any):
        self.model_name = model_name
        self.config: Dict[str, Any] = dict(kwargs)

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #
    @abstractmethod
    def generate(
        self,
        prompts: Sequence[PromptSpec],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        """批量生成，返回与输入等长的结果列表（顺序一一对应）。

        实现约定：**单条失败不得抛异常中断整批**，而应返回 ``error`` 非空的结果，
        由上层决定重试或跳过。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 微调（可选）
    # ------------------------------------------------------------------ #
    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        output_dir: str,
        **kwargs: Any,
    ) -> Optional[str]:
        """用增强数据自举微调（论文 §3.3 的"自举式数据增强微调"）。

        Args:
            records: 微调样本，通常就是 LLM 自己产出的增强数据。
            output_dir: 微调产物（LoRA 适配器）保存目录。

        Returns:
            适配器目录路径；不支持时返回 ``None``。
        """
        raise NotImplementedError(
            f"{self.name} 后端不支持微调（supports_finetuning=False）"
        )

    def export_task_vector(self) -> Optional[TaskVectorLike]:
        """导出当前适配器的任务向量 ``τ = θ_ft - θ_base``。

        Returns:
            参数名到张量的映射；不支持时返回 ``None``。
        """
        raise NotImplementedError(
            f"{self.name} 后端不支持导出任务向量（supports_task_vector=False）"
        )

    def apply_task_vector(self, task_vector: TaskVectorLike, scaling: float = 1.0) -> None:
        """把（合并后的）任务向量写回模型：``θ ← θ_base + scaling · τ``。

        Args:
            task_vector: 参数名到张量的映射。
            scaling: 缩放系数（论文式(7)中的 ``(1-α)·λ_m/λ_{m-1}``）。
        """
        raise NotImplementedError(
            f"{self.name} 后端不支持写回任务向量（supports_task_vector=False）"
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def reset_to_base(self) -> None:
        """把模型恢复到基座状态（丢弃当前适配器）。

        论文 Algorithm 2 每次都从基座 θ_0 出发计算任务向量，因此每个微调周期
        开始前都应确保模型处于基座状态。
        """
        raise NotImplementedError(f"{self.name} 后端不支持 reset_to_base()")

    def close(self) -> None:
        """释放显存 / 关闭连接。默认无操作。"""

    def describe(self) -> Dict[str, Any]:
        """返回后端描述，写入增强结果的元信息。"""
        return {
            "backend": self.name,
            "model_name": self.model_name,
            "supports_finetuning": self.supports_finetuning,
            "supports_task_vector": self.supports_task_vector,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.model_name!r}>"

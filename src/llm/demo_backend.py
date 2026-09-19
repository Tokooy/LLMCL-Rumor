# coding=utf-8
"""``demo`` 后端：不加载任何模型，用规则改写产出"伪增强"。

**它的输出不是论文的 LLM 增强结果**，只用于：

* 在装完依赖但还没有 Qwen 权重的机器上验证"增强 → 训练 → 评测"链路是否接通；
* 单元测试里作为可复现的确定性 stub。

规则改写在 :mod:`data.samples.make_demo` 中实现（与演示数据集共用同一套词表），
因此 demo 后端与 ``data/samples/demo_twitter15.jsonl`` 的行为完全一致。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from data.processors.data_model import DataInstance

from .base import GenerationResult, LLMBackend
from .prompts import PromptSpec

__all__ = ["DemoBackend"]


class DemoBackend(LLMBackend):
    """确定性的规则后端，输出合法 JSON，形状与真实 LLM 输出一致。"""

    name = "demo"
    supports_finetuning = False
    supports_task_vector = False

    def __init__(self, model_name: str = "demo-rule-based", **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        # 延迟导入，避免 src.llm 无故依赖 data.samples
        from data.samples.make_demo import fake_augment

        self._fake_augment = fake_augment

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_instance(spec: PromptSpec) -> Optional[Dict[str, Any]]:
        """从渲染好的 Prompt 里把 ``<instance>...</instance>`` 段抽回来。

        这样 demo 后端无需反向依赖数据加载流程——它只处理"给定 Prompt，
        产出符合结构的 JSON"这一件事，与真实后端职责一致。
        """
        text = spec.user
        start_marker = "<instance>"
        end_marker = "</instance>"
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start == -1 or end == -1 or end <= start:
            return None
        payload = text[start + len(start_marker) : end].strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def generate(
        self,
        prompts: Sequence[PromptSpec],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        results: List[GenerationResult] = []
        for spec in prompts:
            uid = str(spec.meta.get("uid", ""))
            instance = self._extract_instance(spec)
            if instance is None:
                results.append(
                    GenerationResult(
                        uid=uid,
                        prompt_hash=spec.prompt_hash,
                        error="demo 后端无法从 Prompt 中解析 <instance> JSON",
                    )
                )
                continue

            augmented = self._fake_augment(
                {
                    "uid": instance.get("uid", uid),
                    "string_value": instance.get("string_value", ""),
                    "replies": instance.get("replies", []),
                }
            )
            results.append(
                GenerationResult(
                    text=json.dumps(augmented, ensure_ascii=False),
                    uid=uid,
                    prompt_hash=spec.prompt_hash,
                    meta={
                        "backend": self.name,
                        "model_name": self.model_name,
                        "note": "规则伪增强，非论文的 LLM 增强结果",
                    },
                )
            )
        return results

    # ------------------------------------------------------------------ #
    def finetune(self, records, output_dir, **kwargs):  # type: ignore[override]
        """demo 后端不支持微调：返回 ``None`` 而不是抛错，便于上层统一处理。"""
        return None

    def export_task_vector(self):  # type: ignore[override]
        """demo 后端没有参数，任务向量为 ``None``。"""
        return None


def describe_instance(instance: DataInstance) -> Dict[str, Any]:
    """辅助函数：打印一条实例的 prompt 载荷（调试用）。"""
    return instance.prompt_payload()

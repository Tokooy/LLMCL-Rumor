# coding=utf-8
"""后端工厂：按配置构造 LLM 后端与 Prompt 编排器。

把所有"从配置到对象"的装配逻辑收在这一个文件里，好处是
``scripts/augment_data.py`` 与 :class:`src.training.joint_trainer.JointAlignmentTrainer`
共用同一套构造规则，不会出现"脚本能跑、训练器跑不起来"的偏差。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from src.utils.config import Config

from .api_backend import APIBackend
from .augmentor import Augmentor
from .base import LLMBackend
from .demo_backend import DemoBackend
from .hf_backend import HFBackend
from .prompts import PromptBuilder

__all__ = [
    "BACKENDS",
    "build_backend",
    "build_prompt_builder",
    "build_augmentor",
    "resolve_label_descriptions",
]

#: 已注册的后端名 -> 实现类
BACKENDS = {
    "transformers": HFBackend,
    "hf": HFBackend,
    "local": HFBackend,
    "api": APIBackend,
    "openai": APIBackend,
    "demo": DemoBackend,
}


def resolve_label_descriptions(config: Config) -> dict:
    """读取标签描述；缺失时给出论文默认的英文描述。

    注意：这些描述**只用于让 LLM 判断"哪些内容必须保留"**，
    不作为标签预测依据，也不会写进任何标签字段。
    """
    defaults = {
        "NR": "non-rumor (verified truthful information)",
        "FR": "false rumor (verified as false information)",
        "TR": "true rumor (verified as true information)",
        "UR": "unverified rumor (truthfulness not yet determined)",
    }
    configured = config.get_path("llm.prompt.label_descriptions", {}) or {}
    defaults.update({str(key): str(value) for key, value in configured.items()})
    return defaults


def build_backend(config: Config, backend_name: Optional[str] = None) -> LLMBackend:
    """按配置构造 LLM 后端。

    Args:
        config: 全局配置对象。
        backend_name: 覆盖 ``llm.backend``；可选 ``transformers`` / ``api`` / ``demo``。

    Raises:
        ValueError: 后端名未注册。
    """
    name = (backend_name or config.get_path("llm.backend", "transformers")).lower()
    if name not in BACKENDS:
        raise ValueError(
            f"未知的 LLM 后端 {name!r}；可选 {sorted(set(BACKENDS))}"
        )
    backend_cls = BACKENDS[name]

    generation = {
        "temperature": config.get_path("llm.generation.temperature", 0.9),
        "top_p": config.get_path("llm.generation.top_p", 0.9),
        "top_k": config.get_path("llm.generation.top_k", 50),
        "repetition_penalty": config.get_path("llm.generation.repetition_penalty", 1.05),
        "do_sample": config.get_path("llm.generation.do_sample", True),
        "seed": config.get_path("llm.generation.seed", 42),
        "temperature_jitter": config.get_path("llm.augmentation.temperature_jitter", 0.0),
        "max_new_tokens": config.get_path("llm.max_new_tokens", 1024),
        "response_format_json": config.get_path("llm.api.response_format_json", False),
    }
    lora = dict(config.get_path("llm.lora", {}) or {})

    if backend_cls is APIBackend:
        return APIBackend(
            base_url=config.get_path("llm.api.base_url", "https://api.openai.com/v1"),
            api_key_env=config.get_path("llm.api.api_key_env", "OPENAI_API_KEY"),
            model=config.get_path("llm.api.model", config.get_path("llm.model_name", "")),
            timeout=float(config.get_path("llm.api.timeout", 120)),
            max_retries=int(config.get_path("llm.api.max_retries", 3)),
            generation=generation,
            extra_body=dict(config.get_path("llm.api.extra_body", {}) or {}),
        )

    if backend_cls is DemoBackend:
        return DemoBackend(model_name="demo-rule-based")

    return HFBackend(
        model_name=config.get_path("llm.model_name", "Qwen/Qwen2.5-7B-Instruct"),
        local_dir=config.get_path("llm.local_dir", "") or "",
        torch_dtype=config.get_path("llm.torch_dtype", "bfloat16"),
        device_map=config.get_path("llm.device_map", "auto"),
        load_in_8bit=bool(config.get_path("llm.load_in_8bit", False)),
        max_new_tokens=int(config.get_path("llm.max_new_tokens", 1024)),
        generation=generation,
        lora=lora,
        trust_remote_code=bool(config.get_path("llm.trust_remote_code", True)),
    )


def build_prompt_builder(config: Config, template_file: Optional[str] = None) -> PromptBuilder:
    """按配置构造 Prompt 编排器。"""
    return PromptBuilder(
        template_file=template_file
        or config.get_path("llm.prompt.template_file", "configs/prompts/augment_default.txt"),
        label_descriptions=resolve_label_descriptions(config),
        target_field=config.get_path("llm.prompt.target_field", "string_value"),
        max_replies_in_prompt=int(config.get_path("llm.prompt.max_replies_in_prompt", 30)),
        pretty=bool(config.get_path("llm.prompt.pretty", False)),
    )


def build_augmentor(
    config: Config,
    backend: Optional[LLMBackend] = None,
    template_file: Optional[str] = None,
    logger: Optional[Any] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Augmentor:
    """装配一个可直接使用的 :class:`Augmentor`。

    Args:
        config: 全局配置。
        backend: 复用已有后端（例如训练循环里已经加载好的模型）；
            为 ``None`` 时按配置新建。
        template_file: 覆盖 Prompt 模板路径。
        logger: 可选 logger。
        overrides: 覆盖增强参数（如 ``{"concurrency": 8, "max_retries": 1}``）。
    """
    overrides = dict(overrides or {})
    backend = backend or build_backend(config, overrides.pop("backend", None))
    prompt_builder = build_prompt_builder(config, template_file=template_file)

    return Augmentor(
        backend=backend,
        prompt_builder=prompt_builder,
        cache_dir=str(config.get_path("llm.augmentation.cache_dir", "")) or None,
        max_retries=int(overrides.get(
            "max_retries", config.get_path("llm.augmentation.max_retries", 3)
        )),
        copies_per_sample=int(overrides.get(
            "copies_per_sample", config.get_path("llm.augmentation.copies_per_sample", 1)
        )),
        strict_format=bool(overrides.get(
            "strict_format", config.get_path("llm.augmentation.strict_format", True)
        )),
        concurrency=int(overrides.get(
            "concurrency", config.get_path("llm.augmentation.batch_size", 4)
        )),
        overlap_max=float(overrides.get(
            "overlap_max", config.get_path("llm.augmentation.overlap_max", 0.75)
        )),
        temperature_jitter=float(config.get_path("llm.augmentation.temperature_jitter", 0.0)),
        text_mode=config.get_path("data.text_mode", "source_replies"),
        max_seq_length=int(config.get_path("data.max_seq_length", 128)),
        logger=logger,
    )

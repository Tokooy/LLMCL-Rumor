# coding=utf-8
"""LLM 层：数据增强（论文 §3.1）与 LLM/CL 对齐（论文 §3.3）。

子模块
------
* :mod:`src.llm.base`       后端抽象接口
* :mod:`src.llm.hf_backend` ``transformers`` 本地后端（支持 LoRA 微调与任务向量导出）
* :mod:`src.llm.api_backend` OpenAI 兼容接口后端
* :mod:`src.llm.demo_backend` 规则伪增强后端（无需模型权重）
* :mod:`src.llm.prompts`    Prompt 编排（论文 Fg.3 的四条设计目标）
* :mod:`src.llm.parser`     输出解析与质量校验
* :mod:`src.llm.augmentor`  增强流水线（缓存 / 重试 / 并发 / 落盘）
* :mod:`src.llm.factory`    按配置装配以上组件

本包的模块在导入时**不加载任何模型**：torch / transformers / peft 全部延迟到
真正构造后端时才导入，因此 ``src.llm`` 可以被静态导入与单元测试。
"""

from .augmentor import AugmentationStats, Augmentor, dump_augmented, load_original_split
from .base import GenerationResult, LLMBackend
from .lora import (
    build_finetune_record,
    build_finetune_records,
    describe_task_vector,
    export_task_vector,
    finetune_and_export,
    load_task_vector,
    save_task_vector,
)
from .parser import (
    QualityReport,
    build_quality_report,
    char_ngram_overlap,
    extract_json_object,
    merge_augmentation,
    parse_augmentation_response,
    word_overlap,
)
from .prompts import PromptBuilder, PromptSpec
from .task_vector import TaskVector, subtract_parameters
from .ties_merge import (
    MergeReport,
    TiesMerger,
    disjoint_merge,
    elect_sign,
    merge_task_vectors,
    resolve_scaling,
    trim_task_vector,
)
from .factory import (
    BACKENDS,
    build_augmentor,
    build_backend,
    build_prompt_builder,
    resolve_label_descriptions,
)

__all__ = [
    "LLMBackend",
    "GenerationResult",
    "Augmentor",
    "AugmentationStats",
    "dump_augmented",
    "load_original_split",
    "PromptBuilder",
    "PromptSpec",
    "QualityReport",
    "parse_augmentation_response",
    "extract_json_object",
    "build_quality_report",
    "merge_augmentation",
    "char_ngram_overlap",
    "word_overlap",
    "TaskVector",
    "subtract_parameters",
    "trim_task_vector",
    "elect_sign",
    "disjoint_merge",
    "merge_task_vectors",
    "resolve_scaling",
    "TiesMerger",
    "MergeReport",
    "build_finetune_record",
    "build_finetune_records",
    "export_task_vector",
    "describe_task_vector",
    "finetune_and_export",
    "save_task_vector",
    "load_task_vector",
    "BACKENDS",
    "build_backend",
    "build_prompt_builder",
    "build_augmentor",
    "resolve_label_descriptions",
]

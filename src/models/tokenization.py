# coding=utf-8
"""文本编码工具：把字符串转成 BERT 所需的三个张量。

为什么要单独一个模块而不是直接用 ``BertTokenizer``：

1. **统一契约**：数据集（``data.dataset.PairDataset``）只要求传入对象可调用并返回
   ``{"input_ids", "attention_mask", "token_type_ids"}`` 三个等长列表。
   把契约固定在这里，好处是训练脚本可以先建好 tokenizer 再复用给多个 Dataset，
   避免同一次运行里加载多份分词器；
2. **独立于模型**：数据准备/调试阶段只需要分词，不需要加载 BERT 权重，
   :class:`BertTextEncoder` 因此提供 :meth:`release` 以便及时释放内存；
3. **可替换**：想换 RoBERTa 之类的分词器时只改这一个类。

数据集默认的 ``[CLS] ... [SEP]`` 结构由 ``BertTokenizer`` 的
``build_inputs_with_special_tokens`` 自动处理，无需手写。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["BertTextEncoder"]


class BertTextEncoder:
    """BERT 分词与张量构造。

    Args:
        model_name: HuggingFace 模型名（用于定位词表）。
        local_dir: 本地目录（含 ``vocab.txt``）；非空时优先。
        max_seq_length: 默认最大序列长度。
        do_lower_case: 是否小写（``bert-base-uncased`` 为 True）。
        cache_dir: 词表缓存目录。
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        local_dir: str = "",
        max_seq_length: int = 128,
        do_lower_case: bool = True,
        cache_dir: Optional[str] = None,
    ):
        from transformers import BertTokenizer

        self.model_name = model_name
        self.model_path = local_dir or model_name
        self.max_seq_length = int(max_seq_length)
        self.do_lower_case = bool(do_lower_case)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_path, cache_dir=cache_dir, do_lower_case=self.do_lower_case
        )
        self.pad_token_id = int(self.tokenizer.pad_token_id or 0)

    # ------------------------------------------------------------------ #
    def __call__(self, text: str, max_length: Optional[int] = None) -> Dict[str, List[int]]:
        """编码单条文本，返回等长的三个列表。

        这是 ``PairDataset`` 依赖的调用契约：返回**等长**的
        ``input_ids`` / ``attention_mask`` / ``token_type_ids``。
        """
        length = int(max_length or self.max_seq_length)
        encoded = self.tokenizer(
            text,
            max_length=length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "token_type_ids": list(encoded.get("token_type_ids", [0] * length)),
        }

    # ------------------------------------------------------------------ #
    def batch(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
        return_tensors: bool = False,
    ) -> Dict[str, Any]:
        """批量编码。

        Args:
            texts: 文本列表。
            max_length: 最大长度。
            return_tensors: True 时返回 ``torch.Tensor``（需要 torch）。
        """
        length = int(max_length or self.max_seq_length)
        encoded = self.tokenizer(
            list(texts),
            max_length=length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if return_tensors else None,
        )
        return dict(encoded)

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def release(self) -> None:
        """释放分词器引用（训练开始前调用可回收一部分内存）。"""
        self.tokenizer = None  # type: ignore[assignment]

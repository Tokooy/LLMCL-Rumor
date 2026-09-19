# coding=utf-8
"""BERT 句向量编码器（论文 §3.2 的"基于 BERT 的特征提取网络"[27]）。

实现选择说明
------------
原开源项目 ``Bert-TextClassification-master`` 用的是 ``pytorch_pretrained_bert``，
该库已停止维护且在新版 PyTorch/CUDA 上无法安装。本仓库改用官方 ``transformers``，
语义完全等价：同样是"BERT 编码 → 取 ``[CLS]`` 作为句表示"。

区别只有两点，都写在这里以免后续对照时困惑：

1. ``transformers`` 的 ``BertModel`` 返回 ``(last_hidden_state, pooler_output)``，
   而旧库是 4 元组 ``(all_encoded_layers, pooled_output, ...)``；
   本模块显式取 ``last_hidden_state[:, 0]``，**不用** ``pooler_output``——
   因为池化头带一层 tanh 的随机初始化权重，对"句向量"这种通用表示没有必要，
   且论文提到的"CL 提取的 d 维特征"来自编码器输出而非池化头。
2. ``output_all_encoded_layers`` 参数取消了，统一用 ``output_hidden_states``。

**重要**：:class:`BertEncoder` 必须是 ``nn.Module`` 的子类。若把它写成普通对象，
``model.to(device)`` / ``state_dict()`` / ``parameters()`` 都不会递归到内部的
BERT 权重——结果是编码器留在 CPU、checkpoint 里没有 BERT、优化器收不到它的参数，
而训练表面上"跑得通"。因此这里继承 ``nn.Module`` 并把 ``BertModel`` 注册为子模块。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["BertEncoder"]

try:  # pragma: no cover - 环境探测：允许在无 torch 环境下静态导入本模块
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

    class _Placeholder:  # type: ignore[no-redef]
        """torch 缺失时的占位基类，保证模块可被静态导入。"""

    nn = type("nn", (), {"Module": _Placeholder})  # type: ignore[assignment]


class BertEncoder(nn.Module):  # type: ignore[misc]
    """``transformers`` BERT 编码器的 ``nn.Module`` 封装。

    Args:
        model_name: HuggingFace 模型名（如 ``bert-base-uncased``）。
        local_dir: 本地权重目录；非空时优先使用。
        freeze_layers: 冻结底部 N 层（0 表示全量微调，论文默认）。
        gradient_checkpointing: 是否开启梯度检查点（长序列省显存，速度换空间）。
        cache_dir: 模型缓存目录。

    Note:
        transformers 延迟到实例化时才 import，使本模块可被静态导入；
        但 ``nn.Module`` 基类本身需要 torch，因此无 torch 时会在构造阶段报错
        （这是可接受的：没有 torch 就用不到这个类）。
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        local_dir: str = "",
        freeze_layers: int = 0,
        gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 BertEncoder 需要先安装 torch 与 transformers")
        super().__init__()

        from transformers import BertModel, BertTokenizer

        self.model_name = model_name
        self.model_path = local_dir or model_name
        self.freeze_layers = max(0, int(freeze_layers))
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_path, cache_dir=cache_dir, do_lower_case=True
        )
        # add_pooling_layer=False：不使用带随机初始化 tanh 的池化头
        self.model = BertModel.from_pretrained(
            self.model_path, cache_dir=cache_dir, add_pooling_layer=False
        )
        self.hidden_size = int(self.model.config.hidden_size)

        if self.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        if self.freeze_layers:
            self._freeze_bottom_layers(self.freeze_layers)

    # ------------------------------------------------------------------ #
    def _freeze_bottom_layers(self, count: int) -> None:
        """冻结 embedding 层与底部 ``count`` 个编码层的参数。"""
        modules = []
        embeddings = getattr(self.model, "embeddings", None)
        if embeddings is not None:
            modules.append(embeddings)
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layer", None) if encoder is not None else None
        if layers is not None:
            modules.extend(list(layers)[:count])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Any,
        max_length: int = 128,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """把文本编码成模型输入张量（不跑前向）。

        Args:
            texts: 单个字符串或字符串列表。
            max_length: 最大序列长度。
            device: 目标设备；``None`` 表示跟随模型参数所在设备。

        Returns:
            含 ``input_ids`` / ``attention_mask`` / ``token_type_ids`` 的字典。
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = device if device is not None else next(self.model.parameters()).device
        return {key: value.to(target) for key, value in encoded.items()}

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any] = None,
        return_hidden_states: bool = False,
    ) -> Any:
        """前向计算，返回 ``[CLS]`` 句向量（可选返回全部隐状态）。"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_hidden_states,
        )
        cls_vector = outputs.last_hidden_state[:, 0, :]
        if return_hidden_states:
            return cls_vector, outputs.hidden_states
        return cls_vector

    # ------------------------------------------------------------------ #
    @property
    def encoder_parameters(self):
        """暴露 BERT 参数迭代器（``nn.Module.parameters()`` 也会递归到同一批参数）。

        保留这个属性是为了让上层代码显式表达"我要的是编码器的参数"，
        同时避免与 ``nn.Module.parameters`` 方法重名。
        """
        return self.model.parameters()

    def named_encoder_parameters(self, prefix: str = "encoder"):
        """暴露具名参数（带前缀），供优化器做分层学习率分组。"""
        for name, param in self.model.named_parameters():
            yield f"{prefix}.{name}", param

# coding=utf-8
"""分类器（论文 §3.2 式(4)）。

论文原文：

    CL 网络中的分类器被描述为 ……，其中 C 为标签类别数目。
    （NR 表示非谣言、FR 为虚假谣言、TR 为真实谣言、UR 为未经证实的谣言）。
    该分类器由一个全连接层组成，后接 softmax 激活函数。

实现约定
--------
* 网络里**只保留全连接层**，softmax 交给 ``nn.CrossEntropyLoss``
  （它内部做 log-sum-exp，数值更稳）。``predict`` 方法提供显式的 softmax 概率，
  用于评测与可视化；
* 输入可以是投影特征 ``z``（``on_projection=True``，论文的"被送入分类器中"的
  d 维特征），也可以是编码器句向量 ``h``（消融对照）。
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["LabelClassifier"]

try:  # pragma: no cover - 环境探测
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

    class _Placeholder:  # type: ignore[no-redef]
        """torch 缺失时的占位基类。"""

    nn = type("nn", (), {"Module": _Placeholder})  # type: ignore[assignment]


class LabelClassifier(nn.Module):  # type: ignore[misc]
    """全连接 + softmax 分类器。

    Args:
        input_size: 输入特征维度（投影维度或 BERT hidden size）。
        num_labels: 类别数 C（Twitter15/16 为 4）。
        dropout: 输入 dropout 概率。
    """

    def __init__(self, input_size: int, num_labels: int = 4, dropout: float = 0.1):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 LabelClassifier 需要先安装 torch")
        super().__init__()
        if num_labels < 2:
            raise ValueError(f"num_labels 至少为 2，收到 {num_labels}")

        self.input_size = int(input_size)
        self.num_labels = int(num_labels)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.input_size, self.num_labels)

    # ------------------------------------------------------------------ #
    def forward(self, features: Any) -> Any:
        """返回 logits ``[B, C]``（不含 softmax）。"""
        return self.classifier(self.dropout(features))

    def predict(self, features: Any) -> Any:
        """返回 softmax 概率 ``[B, C]``。"""
        return torch.softmax(self.forward(features), dim=-1)

    def predict_label(self, features: Any) -> Any:
        """返回预测类别下标 ``[B]``。"""
        return torch.argmax(self.forward(features), dim=-1)

    def extra_repr(self) -> str:  # pragma: no cover - 调试用
        return f"input={self.input_size}, num_labels={self.num_labels}"


def build_classifier(
    input_size: int,
    num_labels: int = 4,
    dropout: float = 0.1,
    init_range: Optional[float] = None,
) -> LabelClassifier:
    """构造分类器，可选地按 BERT 的风格初始化权重。

    Args:
        init_range: 非空时用 ``normal_(0, init_range)`` 初始化权重、零初始化 bias
            （与 ``BertPreTrainedModel.init_bert_weights`` 一致）。
    """
    classifier = LabelClassifier(input_size=input_size, num_labels=num_labels, dropout=dropout)
    if init_range is not None:
        classifier.classifier.weight.data.normal_(mean=0.0, std=init_range)
        classifier.classifier.bias.data.zero_()
    return classifier

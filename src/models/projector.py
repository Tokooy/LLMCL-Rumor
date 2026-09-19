# coding=utf-8
"""MLP 投影头（论文 §3.2 与式(1)）。

论文原文：

    CL 网络由一个基于 BERT 的特征提取网络和一个基于多层感知机（MLP）投影头构成，
    用来提取数据特征。…… 这一过程被一个非线性变换表示。

即 ``z = W2 · σ(W1 · h)``，其中 ``h`` 是 BERT 的 ``[CLS]`` 句向量（768 维），
``σ`` 是非线性激活函数，``z`` 是对比学习所在的投影空间表示。

设计要点
--------
* **两层结构可配置**：``num_layers=1`` 时退化为线性投影（消融用）；
* **L2 归一化默认开启**：归一化后内积等价于余弦相似度，与论文"余弦相似度被用于
  量化特征向量的相似性"一致，也让温度超参 τ 的尺度稳定；
* **温度不是本模块的参数**：τ 属于损失函数（见 :mod:`src.training.losses`），
  放在这里会让"投影头"和"对比损失"的职责混淆。
"""

from __future__ import annotations

from typing import Any, List, Optional

__all__ = ["MLPProjector", "resolve_activation"]

try:  # pragma: no cover - 环境探测
    import torch
    from torch import nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

    class _Placeholder:  # type: ignore[no-redef]
        """torch 缺失时的占位基类，保证模块可被静态导入。"""

    nn = type("nn", (), {"Module": _Placeholder})  # type: ignore[assignment]


def resolve_activation(name: str):
    """把配置里的激活函数名解析成可调用对象。"""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("使用 resolve_activation 需要先安装 torch")
    mapping = {
        "relu": F.relu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "silu": F.silu,
        "leaky_relu": F.leaky_relu,
        "identity": lambda tensor: tensor,
    }
    key = (name or "relu").lower()
    if key not in mapping:
        raise ValueError(f"不支持的激活函数 {name!r}；可选 {sorted(mapping)}")
    return mapping[key]


class MLPProjector(nn.Module):  # type: ignore[misc]
    """MLP 投影头：``h -> z``。

    Args:
        input_size: 输入维度（BERT hidden size，通常 768）。
        hidden_size: 隐藏层维度；``num_layers == 1`` 时忽略。
        output_size: 投影空间维度（论文写作 d 维）。
        num_layers: 层数。1 = 单层线性；2 = ``Linear->Act->Linear``；>=3 时按
            ``input -> hidden -> ... -> output`` 搭建。
        activation: 激活函数名。
        dropout: dropout 概率。作用在**两处**：输入句向量之后（进入 MLP 之前），
            以及每个隐藏层之后；``num_layers == 1`` 时只有输入那一处。
        normalize: 输出是否做 L2 归一化（默认 True）。

    Note:
        投影头是"用 dropout 正则化的 MLP"这一常见写法（SimCLR 系列同款），
        因此输入处也有 dropout——这一处容易被 docstring 忽略，特意写明。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 768,
        output_size: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 MLPProjector 需要先安装 torch")
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers 至少为 1，收到 {num_layers}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.normalize = bool(normalize)
        self.activation_name = activation
        self.activation = resolve_activation(activation)

        layers: List[Any] = []
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if num_layers == 1:
            # 单层：退化为线性投影（消融设置，论文写的是 MLP）
            layers.append(nn.Linear(self.input_size, self.output_size))
        else:
            # 第一层：input -> hidden，接激活
            layers.append(nn.Linear(self.input_size, self.hidden_size))
            layers.append(_FunctionalActivation(self.activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            # 中间层：hidden -> hidden，每层接激活与 dropout
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(_FunctionalActivation(self.activation))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
            # 最后一层：hidden -> output，接激活（无数值稳定性问题）
            layers.append(nn.Linear(self.hidden_size, self.output_size))

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    def forward(self, hidden: Any) -> Any:
        """前向：``[B, input_size] -> [B, output_size]``。"""
        projected = self.dropout(hidden)
        projected = self.net(projected)
        if self.normalize:
            projected = F.normalize(projected, dim=-1)
        return projected

    # ------------------------------------------------------------------ #
    @property
    def output_dim(self) -> int:
        return self.output_size

    def extra_repr(self) -> str:  # pragma: no cover - 调试用
        return (
            f"input={self.input_size}, hidden={self.hidden_size}, output={self.output_size}, "
            f"layers={self.num_layers}, act={self.activation_name}, normalize={self.normalize}"
        )


class _FunctionalActivation(nn.Module):  # type: ignore[misc]
    """把函数式激活（gelu/tanh/silu...）包装成 ``nn.Module``。"""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, tensor: Any) -> Any:  # pragma: no cover - 一行转发
        return self.func(tensor)

# coding=utf-8
"""对比学习网络总装（论文 §3.2 的 CL 分类网络）。

结构（与论文 Fig.1 的 "CL classification network" 对应）::

    text ──► BERT 编码器 ──► h ([CLS], 768)
                              │
                              ├──► MLP 投影头 ──► z (d 维, 默认 128)
                              │                      │
                              │                      └──► 分类器 ──► logits [C]
                              │
                              └──（消融：分类器直接接 h）

一次前向返回 ``logits`` 与 ``projection``，因为训练目标同时需要两者：

* 交叉熵用 ``logits``（论文式(4)的标签预测）；
* InfoNCE 用 ``projection``（论文式(3)的投影空间对比）。

这样设计避免了"跑两次前向"，也是论文"联合微调/联合优化"最直接的实现方式。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .classifier import LabelClassifier
from .projector import MLPProjector

__all__ = ["ContrastiveModel", "build_contrastive_model"]

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


class ContrastiveModel(nn.Module):  # type: ignore[misc]
    """BERT + MLP 投影头 + 分类器。

    Args:
        encoder: :class:`src.models.encoder.BertEncoder` 实例。
        projector: :class:`src.models.projector.MLPProjector` 实例。
        classifier: :class:`src.models.classifier.LabelClassifier` 实例。
        freeze_encoder: 是否冻结编码器（论文在"训练完成后冻结"，
            这里提供开关以便先冻结编码器热启动投影头）。
    """

    def __init__(
        self,
        encoder: Any,
        projector: Any,
        classifier: Any,
        freeze_encoder: bool = False,
    ):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 ContrastiveModel 需要先安装 torch")
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.classifier = classifier
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.set_encoder_grad(False)

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any] = None,
        return_hidden: bool = False,
    ) -> Dict[str, Any]:
        """一次前向返回所有需要的表示。

        Returns:
            ``{"logits": [B, C], "projection": [B, d], "hidden": [B, H]?}``。
        """
        hidden = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        projection = self.projector(hidden)
        # 分类器输入取决于配置：论文用投影特征 z，消融可换成句向量 h
        classifier_input = projection if self.classifier_on_projection else hidden
        logits = self.classifier(classifier_input)

        outputs: Dict[str, Any] = {"logits": logits, "projection": projection}
        if return_hidden:
            outputs["hidden"] = hidden
        return outputs

    # ------------------------------------------------------------------ #
    @property
    def classifier_on_projection(self) -> bool:
        """分类器是否接在投影特征上（由 ``model.classifier.on_projection`` 决定）。

        通过比较维度推断：分类器输入维度等于投影维度时视为接在投影上。
        这样无需额外传参，也避免了配置与结构不一致的风险。
        """
        return int(self.classifier.input_size) == int(self.projector.output_dim)

    @property
    def num_labels(self) -> int:
        return int(self.classifier.num_labels)

    @property
    def projection_dim(self) -> int:
        return int(self.projector.output_dim)

    # ------------------------------------------------------------------ #
    # 参数管理
    # ------------------------------------------------------------------ #
    def set_encoder_grad(self, requires_grad: bool) -> None:
        """开关编码器梯度（论文测试阶段"CL 模型的参数被冻结"）。"""
        for param in self.encoder.parameters():
            param.requires_grad = bool(requires_grad)
        self.freeze_encoder = not requires_grad

    def trainable_parameters(self):
        """返回需要梯度的参数（用于只对可训练部分建优化器）。"""
        return [param for param in self.parameters() if param.requires_grad]

    def encoder_parameters(self):
        """返回编号器参数（供优化器做分层学习率分组）。"""
        return self.encoder.parameters()

    def named_encoder_parameters(self, prefix: str = "encoder.model"):
        """返回带前缀的编码器具名参数。

        前缀默认 ``encoder.model``：因为 :class:`ContrastiveModel` 把
        ``encoder``（:class:`~src.models.encoder.BertEncoder`）注册为子模块，
        而它内部又把 HuggingFace 的 ``BertModel`` 注册为 ``model``，
        所以完整的参数名是 ``encoder.model.encoder.layer.<i>...``。
        """
        return self.encoder.named_encoder_parameters(prefix=prefix)


def build_contrastive_model(
    encoder_config: Any,
    projector_config: Any,
    classifier_config: Any,
    num_labels: int = 4,
    freeze_encoder: bool = False,
) -> ContrastiveModel:
    """按配置节构造完整的 CL 网络。

    Args:
        encoder_config: ``model.encoder`` 配置节。
        projector_config: ``model.projector`` 配置节。
        classifier_config: ``model.classifier`` 配置节。
        num_labels: 类别数（Twitter15/16 为 4）。
        freeze_encoder: 是否冻结编码器。

    Returns:
        组装好的 :class:`ContrastiveModel`。
    """
    from .classifier import build_classifier
    from .encoder import BertEncoder
    from .projector import MLPProjector

    encoder = BertEncoder(
        model_name=encoder_config.get("name", "bert-base-uncased"),
        local_dir=encoder_config.get("local_dir", "") or "",
        freeze_layers=int(encoder_config.get("freeze_layers", 0) or 0),
        gradient_checkpointing=bool(encoder_config.get("gradient_checkpointing", False)),
    )

    projector = MLPProjector(
        input_size=encoder.hidden_size,
        hidden_size=int(projector_config.get("hidden_size", 768)),
        output_size=int(projector_config.get("output_size", 128)),
        num_layers=int(projector_config.get("num_layers", 2)),
        activation=projector_config.get("activation", "relu"),
        dropout=float(projector_config.get("dropout", 0.1)),
        normalize=bool(projector_config.get("normalize", True)),
    )

    on_projection = bool(classifier_config.get("on_projection", True))
    classifier_input = projector.output_dim if on_projection else encoder.hidden_size
    classifier = build_classifier(
        input_size=classifier_input,
        num_labels=num_labels,
        dropout=float(classifier_config.get("dropout", 0.1)),
    )

    return ContrastiveModel(
        encoder=encoder,
        projector=projector,
        classifier=classifier,
        freeze_encoder=freeze_encoder,
    )

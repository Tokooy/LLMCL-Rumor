# coding=utf-8
"""模型层：CL 分类网络的各个组件（论文 §3.2）。

结构层次（与论文 Fig.1 的 "CL classification network" 一致）::

    src/models/encoder.py     BERT 句向量编码器（[CLS]）
    src/models/projector.py   MLP 投影头，论文式(1)
    src/models/classifier.py  全连接 + softmax 分类器，论文式(4)
    src/models/cl_model.py    总装，一次前向返回 logits 与投影特征

``src/models/backbones/`` 保留原开源项目 ``Bert-TextClassification-master`` 中的
其它文本分类骨干（BertCNN / BertLSTM / BertATT / BertRCNN / BertCNNPlus /
BertDPCNN / BertHAN）。**论文没有使用它们**，保留是为了：

1. 便于做消融（"换成 CNN 池化会不会更好"）；
2. 便于对照原开源代码，确认改造范围。

与 :mod:`src.models.backbones` 的依赖关系：backbones 既不被 CL 网络引用，
也不依赖 CL 网络，可独立删除而不影响主流程。
"""

from .classifier import LabelClassifier, build_classifier
from .cl_model import ContrastiveModel, build_contrastive_model

__all__ = [
    "ContrastiveModel",
    "build_contrastive_model",
    "LabelClassifier",
    "build_classifier",
]

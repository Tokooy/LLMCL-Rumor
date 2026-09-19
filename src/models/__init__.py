# coding=utf-8
"""模型层：CL 分类网络的各个组件（论文 §3.2）。

结构层次（与论文 Fig.1 的 "CL classification network" 一致）::

    src/models/encoder.py     BERT 句向量编码器（[CLS]）
    src/models/projector.py   MLP 投影头，论文式(1)
    src/models/classifier.py  全连接 + softmax 分类器，论文式(4)
    src/models/cl_model.py    总装，一次前向返回 logits 与投影特征
    src/models/tokenization.py BERT 分词（数据集契约）

论文明说 CL 网络由"BERT 特征提取网络 + MLP 投影头 + 分类器"三部分组成，
**没有使用**原开源项目里的 CNN/LSTM/注意力池化骨干。那些骨干的完整代码保存在
``reference/baseline/``（原项目快照），主流程不引用它们；
如需做"换池化方式"的消融实验，可把对应的 ``BertXXX.py`` 复制进本目录后自行接入，
并在 ``cl_model.ContrastiveModel`` 里替换编码器的输出池化方式。
"""

from .classifier import LabelClassifier, build_classifier
from .cl_model import ContrastiveModel, build_contrastive_model

__all__ = [
    "ContrastiveModel",
    "build_contrastive_model",
    "LabelClassifier",
    "build_classifier",
]

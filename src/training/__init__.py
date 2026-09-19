# coding=utf-8
"""训练层：损失函数、对比学习训练器、LLM/CL 联合对齐、评估与可视化。

子模块
------
* :mod:`src.training.losses`        论文式(2)(3)(4)的对比损失与联合损失
* :mod:`src.training.cl_trainer`    一个增强轮次内的 CL 训练（含优化器/调度器）
* :mod:`src.training.joint_trainer` 论文 Algorithm 2 的联合对齐主循环
* :mod:`src.training.evaluate`      指标计算与模型评测（论文 Table 3–8）
* :mod:`src.training.visualize`     t-SNE 特征分布图（论文 Fg.7–Fg.10）

导入策略：本包顶层**不导入 torch**，各子模块按需导入，
这样 ``src.training.evaluate`` 之类的纯指标代码可以在无 torch 环境下被复用。
"""

from .losses import (
    CombinedLoss,
    ContrastiveLoss,
    PairingMode,
    cosine_similarity_matrix,
    paired_infonce_loss,
    supervised_contrastive_loss,
)

__all__ = [
    "CombinedLoss",
    "ContrastiveLoss",
    "PairingMode",
    "paired_infonce_loss",
    "supervised_contrastive_loss",
    "cosine_similarity_matrix",
]

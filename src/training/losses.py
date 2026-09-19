# coding=utf-8
"""对比学习损失（论文式(2)(3)(4)）。

论文原文要点
------------
* 式(2)：用来衡量两个投影表示相似度的函数（实现为**余弦相似度**，
  论文明确指出"余弦相似度被用于量化特征向量的相似性"）；
* 式(3)：把式(2)代入 **InfoNCE Loss**[33]，得到第 n 个 epoch 的 CL 损失；
  其中 ``τ`` 是温度超参数；
* "参与损失计算的样本包括原始样本及其增强样本，数量为 2B"；
* "在每个 batch 中，(x_i, x'_i) 被配对为正样本，而 (x_i, x'_j) 被视为负样本"。

因此本模块提供两种配对策略（配置项 ``model.contrastive.pairing``）：

``paired``（默认，严格照论文）
    只有 **(原样本, 它自己的增强样本)** 互为正样本；同一 batch 内其它所有样本
    ——**包括同标签的样本**——在计算 InfoNCE 时都被屏蔽为"既非正也非负"。
    屏蔽是必要的：同一 batch 里不同原样本的副本互为"锚点-正样本"关系，
    若不屏蔽，配对模式就退化成有监督对比学习，与论文描述不符。

``supervised``（消融备选）
    同标签样本互为正样本（SupCon 形式），用于回答"论文为何不直接做有监督对比"。

数值实现
--------
* 相似度用 ``F.normalize`` 后的内积，等价于余弦相似度除以温度；
* 用 ``logsumexp`` 手工算 InfoNCE 而不是 ``CrossEntropyLoss``，
  是因为需要自定义正/负样本掩码；
* 全部在 ``float32`` 下计算相似度矩阵——混合精度训练时 bf16 的 logsumexp 会掉精度。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "PairingMode",
    "normalize_embeddings",
    "cosine_similarity_matrix",
    "paired_infonce_loss",
    "supervised_contrastive_loss",
    "ContrastiveLoss",
    "CombinedLoss",
]

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
        """torch 缺失时的占位基类。"""

    nn = type("nn", (), {"Module": _Placeholder})  # type: ignore[assignment]


class PairingMode:
    """正样本对构造方式（常量集合，避免到处写裸字符串）。"""

    PAIRED = "paired"
    SUPERVISED = "supervised"

    ALL = (PAIRED, SUPERVISED)


# ---------------------------------------------------------------------- #
# 相似度
# ---------------------------------------------------------------------- #
def normalize_embeddings(embeddings: Any) -> Any:
    """L2 归一化（若投影头已归一化，这里再归一化是幂等的）。"""
    return F.normalize(embeddings.float(), dim=-1)


def cosine_similarity_matrix(left: Any, right: Optional[Any] = None) -> Any:
    """余弦相似度矩阵 ``[N, M]``；``right`` 为空时计算 ``left`` 与自身的相似度。"""
    left_norm = normalize_embeddings(left)
    right_norm = left_norm if right is None else normalize_embeddings(right)
    return left_norm @ right_norm.t()


# ---------------------------------------------------------------------- #
# 论文式(3)：严格配对 InfoNCE
# ---------------------------------------------------------------------- #
def paired_infonce_loss(
    anchor: Any,
    positive: Any,
    temperature: float = 0.07,
    negative_mask: Optional[Any] = None,
    symmetric: bool = True,
) -> Any:
    """论文式(3)的实现：配对样本做正样本、batch 内其余样本做负样本。

    以 batch 大小 B 为例：``anchor`` 是原样本投影 ``z``，``positive`` 是其增强样本
    投影 ``z'``，二者形状均为 ``[B, d]``。第 i 行的正样本就是第 i 列，
    负样本是其余 ``B-1`` 列——这正是论文"数量为 2B 的样本参与损失计算"的等价写法
    （把 2B 个样本拼起来时，z_i 的正样本只有 z'_i）。

    Args:
        anchor: ``[B, d]``
        positive: ``[B, d]``
        temperature: 温度超参数 τ。
        negative_mask: ``[B, B]`` 布尔张量，``True`` 表示该位置**允许**作为负样本；
            ``None`` 表示全部允许。用于屏蔽"同一原样本的其它副本"。
        symmetric: 是否同时计算 positive→anchor 方向（默认 True，
            论文的损失对两个方向对称，等价于把 2B 个样本都当锚点各算一次）。

    Returns:
        标量损失。
    """
    if temperature <= 0:
        raise ValueError(f"温度 τ 必须为正数，收到 {temperature}")

    similarity = cosine_similarity_matrix(anchor, positive) / float(temperature)
    batch_size = similarity.shape[0]
    if batch_size < 2:
        # batch 内没有负样本，InfoNCE 无法定义；返回 0 而不是 NaN
        return similarity.sum() * 0.0

    # 屏蔽无效负样本：-inf 使其在 logsumexp 中权重为 0。
    # 用 where 而非 masked_fill，保证被屏蔽位置的反向传播梯度严格为 0（不会出现 NaN）。
    if negative_mask is not None:
        mask = negative_mask.to(similarity.device, dtype=torch.bool)
        similarity = torch.where(
            mask, similarity, torch.full_like(similarity, float("-inf"))
        )

    labels = torch.arange(batch_size, device=similarity.device)
    loss_anchor = F.cross_entropy(similarity, labels)
    if not symmetric:
        return loss_anchor

    loss_positive = F.cross_entropy(similarity.t(), labels)
    return 0.5 * (loss_anchor + loss_positive)


def _build_negative_mask(group_ids: Sequence[Any], device: Any) -> Any:
    """构造负样本掩码：同一 ``group_id``（同一原样本的不同副本）互不为负样本。

    返回值形状 ``[B, B]``，``True`` 表示该位置允许参与 logsumexp（即允许作为负样本）。
    对角线（正样本列）恒为 ``True``。
    """
    size = len(group_ids)
    mask = torch.ones((size, size), dtype=torch.bool, device=device)
    for i in range(size):
        for j in range(size):
            if i != j and group_ids[i] == group_ids[j]:
                mask[i, j] = False
    return mask


# ---------------------------------------------------------------------- #
# 消融：有监督对比（SupCon）
# ---------------------------------------------------------------------- #
def supervised_contrastive_loss(
    embeddings: Any,
    labels: Any,
    temperature: float = 0.07,
    eps: float = 1e-12,
) -> Any:
    """SupCon 形式的对比损失（Khosla et al. 2020 的实现风格）。

    同一标签的样本互为正样本；分母排除自身。仅作为消融对照，
    论文正文描述的是 :func:`paired_infonce_loss`。
    """
    if temperature <= 0:
        raise ValueError(f"温度 τ 必须为正数，收到 {temperature}")

    features = normalize_embeddings(embeddings)
    similarity = (features @ features.t()) / float(temperature)

    size = similarity.shape[0]
    if size < 2:
        return similarity.sum() * 0.0

    device = similarity.device
    size = similarity.shape[0]
    self_mask = torch.eye(size, dtype=torch.bool, device=device)

    # 数值稳定：先减去每行最大值（detach 避免梯度经过常数项）
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    # 屏蔽自身：用 where 而不是 masked_fill(-inf)，否则 masked 位置在
    # ``0 * (-inf)`` 形式的反向传播里会产生 NaN 梯度
    logits = torch.where(self_mask, torch.full_like(logits, float("-inf")), logits)

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.t()).to(device) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return similarity.sum() * 0.0

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # 只对"确有正样本"的行求均值，避免 0/0
    mean_log_prob = (positive_mask * log_prob).sum(dim=1)[valid] / (
        positive_count[valid].float() + eps
    )
    return -mean_log_prob.mean()


# ---------------------------------------------------------------------- #
# 组合损失
# ---------------------------------------------------------------------- #
class ContrastiveLoss(nn.Module):  # type: ignore[misc]
    """把 batch 内多份增强样本的对比损失聚合成一个标量。

    当一条样本有 K 份增强样本时（``copies_per_sample > 1`` 或多轮增强样本一起用），
    对每一份分别计算配对 InfoNCE 后取平均。这样每条样本的贡献权重恒为 1，
    不会因为"某条样本增强成功、另一条失败"而让梯度失衡。
    """

    def __init__(self, temperature: float = 0.07, pairing: str = PairingMode.PAIRED):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 ContrastiveLoss 需要先安装 torch")
        super().__init__()
        if pairing not in PairingMode.ALL:
            raise ValueError(
                f"pairing 只支持 {PairingMode.ALL}，收到 {pairing!r}"
            )
        self.temperature = float(temperature)
        self.pairing = pairing
        # 需要跨卡同步吗？论文为单机多卡 DataParallel 场景，默认不同步
        self.last_components: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def forward(
        self,
        anchor_embeddings: Any,
        augmented_embeddings: Any,
        labels: Optional[Any] = None,
        group_ids: Optional[Sequence[Any]] = None,
    ) -> Any:
        """计算对比损失。

        Args:
            anchor_embeddings: ``[B, d]`` 原样本投影。
            augmented_embeddings: ``[B, K, d]`` 或 ``[B, d]``（K=1）增强样本投影。
            labels: ``[B]`` 标签，仅 ``pairing='supervised'`` 时需要。
            group_ids: 长度 B 的分组标识（默认用样本在 batch 内的下标），
                用于在 K>1 时屏蔽同一原样本的不同副本。

        Returns:
            标量损失。
        """
        if augmented_embeddings.dim() == 2:
            augmented_embeddings = augmented_embeddings.unsqueeze(1)
        batch_size, copies = augmented_embeddings.shape[0], augmented_embeddings.shape[1]
        if batch_size == 0 or copies == 0:
            return anchor_embeddings.sum() * 0.0

        if group_ids is None:
            group_ids = list(range(batch_size))

        if self.pairing == PairingMode.SUPERVISED:
            if labels is None:
                raise ValueError("pairing='supervised' 时必须提供 labels")
            stacked = torch.cat(
                [anchor_embeddings.unsqueeze(1), augmented_embeddings], dim=1
            )  # [B, K+1, d]
            flat = stacked.reshape(-1, stacked.shape[-1])
            flat_labels = labels.unsqueeze(1).expand(-1, copies + 1).reshape(-1)
            loss = supervised_contrastive_loss(flat, flat_labels, self.temperature)
            self.last_components = {
                "cl_loss": float(loss.detach().cpu()),
                "pairing": PairingMode.SUPERVISED,
            }
            return loss

        # ---- paired：逐份增强样本分别计算 ----
        losses: List[Any] = []
        mask = _build_negative_mask(group_ids, anchor_embeddings.device)
        for copy_index in range(copies):
            losses.append(
                paired_infonce_loss(
                    anchor_embeddings,
                    augmented_embeddings[:, copy_index, :],
                    temperature=self.temperature,
                    negative_mask=mask,
                )
            )
        loss = torch.stack(losses).mean()
        self.last_components = {
            "cl_loss": float(loss.detach().cpu()),
            "copies": copies,
            "pairing": PairingMode.PAIRED,
        }
        return loss


class CombinedLoss(nn.Module):  # type: ignore[misc]
    """联合目标：``ce_weight * 交叉熵 + cl_weight * InfoNCE``。

    论文把"标签预测"和"对比特征提取"放在同一个 CL 网络里联合优化，
    因此训练损失取两者加权和。权重可在配置 ``training.cl.ce_weight`` /
    ``training.cl.cl_weight`` 中调整（默认各 1.0）。
    """

    def __init__(
        self,
        temperature: float = 0.07,
        pairing: str = PairingMode.PAIRED,
        ce_weight: float = 1.0,
        cl_weight: float = 1.0,
        label_smoothing: float = 0.0,
        joint_objective: bool = True,
    ):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 CombinedLoss 需要先安装 torch")
        super().__init__()
        self.contrastive = ContrastiveLoss(temperature=temperature, pairing=pairing)
        self.ce_weight = float(ce_weight)
        self.cl_weight = float(cl_weight)
        self.joint_objective = bool(joint_objective)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.last_components: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def forward(
        self,
        logits: Any,
        labels: Any,
        anchor_embeddings: Optional[Any] = None,
        augmented_embeddings: Optional[Any] = None,
        group_ids: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """计算总损失。

        Returns:
            ``{"loss": 标量, "ce_loss": float, "cl_loss": float}``。
            ``augmented_embeddings`` 为 ``None`` 时（例如评测阶段）只算交叉熵；
            ``joint_objective=False`` 时也退化为纯分类损失（消融对照）。
        """
        ce_loss = self.cross_entropy(logits, labels)
        components: Dict[str, Any] = {
            "loss": self.ce_weight * ce_loss,
            "ce_loss": float(ce_loss.detach().cpu()),
            "cl_loss": 0.0,
        }

        if (
            not self.joint_objective
            or anchor_embeddings is None
            or augmented_embeddings is None
        ):
            self.last_components = {
                "ce_loss": components["ce_loss"],
                "cl_loss": 0.0,
            }
            return components

        cl_loss = self.contrastive(
            anchor_embeddings,
            augmented_embeddings,
            labels=labels,
            group_ids=group_ids,
        )
        total = components["loss"] + self.cl_weight * cl_loss
        components["loss"] = total
        components["cl_loss"] = float(cl_loss.detach().cpu())
        self.last_components = {
            "ce_loss": components["ce_loss"],
            "cl_loss": components["cl_loss"],
            "total_loss": float(total.detach().cpu()),
        }
        return components

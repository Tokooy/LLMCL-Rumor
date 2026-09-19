# coding=utf-8
"""论文式(2)(3)(4) 的单元测试：余弦相似度、配对 InfoNCE、联合损失。

需要 torch；未安装时整个模块会被跳过。
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import require_torch

torch = pytest.importorskip("torch", reason="对比损失测试需要 PyTorch")


# ===================================================================== #
# 式(2)：余弦相似度
# ===================================================================== #
class TestSimilarity:
    def test_cosine_similarity_matches_manual_computation(self):
        from src.training.losses import cosine_similarity_matrix

        left = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        right = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        matrix = cosine_similarity_matrix(left, right)
        # 第一行：[1, 1/sqrt(2)]；第二行：[0, 1/sqrt(2)]
        assert matrix[0, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert matrix[0, 1].item() == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert matrix[1, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert matrix[1, 1].item() == pytest.approx(1 / math.sqrt(2), abs=1e-6)

    def test_normalize_embeddings_unit_norm(self):
        from src.training.losses import normalize_embeddings

        embeddings = torch.randn(5, 16)
        normalized = normalize_embeddings(embeddings)
        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)


# ===================================================================== #
# 式(3)：配对 InfoNCE
# ===================================================================== #
class TestPairedInfoNCE:
    def test_perfect_alignment_gives_low_loss(self):
        """正样本完全对齐、负样本正交时，损失应显著低于随机情形。"""
        from src.training.losses import paired_infonce_loss

        batch = 4
        # 每个锚点与其正样本完全相同，不同样本之间正交
        base = torch.eye(batch)
        anchor = torch.cat([base, torch.zeros(batch, 4)], dim=1)
        positive = anchor.clone()
        loss = paired_infonce_loss(anchor, positive, temperature=0.07)
        # 正样本相似度 1、负样本 0 → loss = log(1 + (B-1)·exp(-1/τ))
        expected = math.log(1 + (batch - 1) * math.exp(-1 / 0.07))
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_misaligned_positives_raise_loss(self):
        from src.training.losses import paired_infonce_loss

        batch = 4
        anchor = torch.eye(batch)
        same = paired_infonce_loss(anchor, anchor.clone(), temperature=0.1)
        shuffled = paired_infonce_loss(anchor, anchor.roll(1, dims=0), temperature=0.1)
        assert shuffled.item() > same.item()

    def test_lower_temperature_sharpens_loss(self):
        from src.training.losses import paired_infonce_loss

        anchor = torch.randn(8, 16)
        positive = anchor + 0.5 * torch.randn(8, 16)
        high_temp = paired_infonce_loss(anchor, positive, temperature=1.0)
        low_temp = paired_infonce_loss(anchor, positive, temperature=0.05)
        assert low_temp.item() > high_temp.item()

    def test_single_sample_returns_zero_instead_of_nan(self):
        """batch 内没有负样本时 InfoNCE 无法定义，必须返回 0 而不是 NaN。"""
        from src.training.losses import paired_infonce_loss

        anchor = torch.randn(1, 8)
        positive = torch.randn(1, 8)
        loss = paired_infonce_loss(anchor, positive, temperature=0.07)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)

    def test_invalid_temperature_raises(self):
        from src.training.losses import paired_infonce_loss

        with pytest.raises(ValueError, match="温度"):
            paired_infonce_loss(torch.randn(2, 4), torch.randn(2, 4), temperature=0.0)

    def test_negative_mask_blocks_positions(self):
        """被屏蔽的位置不参与 logsumexp：屏蔽掉"最难的负样本"后损失必须下降。"""
        from src.training.losses import paired_infonce_loss

        torch.manual_seed(0)
        batch = 3
        anchor = torch.randn(batch, 8)
        positive = anchor + 0.1 * torch.randn(batch, 8)

        full_mask = torch.ones(batch, batch, dtype=torch.bool)
        # 找出最相似的"非对角"位置并屏蔽
        from src.training.losses import cosine_similarity_matrix

        similarity = cosine_similarity_matrix(anchor)
        similarity.fill_diagonal_(-float("inf"))
        hardest = int(torch.argmax(similarity.reshape(-1)).item())
        row, column = divmod(hardest, batch)
        if row != column:
            full_mask[row, column] = False

        with_hard = paired_infonce_loss(anchor, positive, 0.07, negative_mask=torch.ones_like(full_mask))
        without_hard = paired_infonce_loss(anchor, positive, 0.07, negative_mask=full_mask)
        assert without_hard.item() <= with_hard.item() + 1e-6

    def test_gradients_are_finite_with_masking(self):
        """屏蔽用 -inf 时曾出现过 NaN 梯度，这里做回归测试。"""
        from src.training.losses import paired_infonce_loss

        anchor = torch.randn(3, 8, requires_grad=True)
        positive = torch.randn(3, 8, requires_grad=True)
        mask = torch.ones(3, 3, dtype=torch.bool)
        mask[0, 1] = False
        loss = paired_infonce_loss(anchor, positive, 0.07, negative_mask=mask)
        loss.backward()
        assert torch.isfinite(anchor.grad).all()
        assert torch.isfinite(positive.grad).all()


# ===================================================================== #
# 消融：有监督对比
# ===================================================================== #
class TestSupervisedContrastive:
    def test_same_label_are_positives(self):
        from src.training.losses import supervised_contrastive_loss

        embeddings = torch.tensor(
            [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, -0.1]], dtype=torch.float32
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(embeddings, labels, temperature=0.1)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_no_positive_pairs_returns_zero(self):
        from src.training.losses import supervised_contrastive_loss

        embeddings = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])
        loss = supervised_contrastive_loss(embeddings, labels)
        assert loss.item() == 0.0


# ===================================================================== #
# 组合：ContrastiveLoss / CombinedLoss
# ===================================================================== #
class TestCombinedLoss:
    def test_contrastive_loss_accepts_2d_augmented(self):
        from src.training.losses import ContrastiveLoss

        criterion = ContrastiveLoss(temperature=0.1, pairing="paired")
        anchor = torch.randn(4, 8)
        augmented = torch.randn(4, 8)
        loss = criterion(anchor, augmented)
        assert torch.isfinite(loss)

    def test_contrastive_loss_averages_multiple_copies(self):
        from src.training.losses import ContrastiveLoss

        criterion = ContrastiveLoss(temperature=0.1)
        anchor = torch.randn(4, 8)
        copies = torch.randn(4, 3, 8)
        loss = criterion(anchor, copies)
        assert torch.isfinite(loss)

    def test_supervised_pairing_requires_labels(self):
        from src.training.losses import ContrastiveLoss

        criterion = ContrastiveLoss(temperature=0.1, pairing="supervised")
        with pytest.raises(ValueError, match="labels"):
            criterion(torch.randn(4, 8), torch.randn(4, 1, 8))

    def test_invalid_pairing_raises(self):
        from src.training.losses import ContrastiveLoss

        with pytest.raises(ValueError, match="pairing"):
            ContrastiveLoss(pairing="unknown")

    def test_combined_loss_weights(self):
        from src.training.losses import CombinedLoss

        criterion = CombinedLoss(
            temperature=0.1, ce_weight=1.0, cl_weight=2.0, joint_objective=True
        )
        logits = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 3])
        anchor = torch.randn(4, 8)
        augmented = torch.randn(4, 1, 8)
        components = criterion(
            logits, labels, anchor_embeddings=anchor, augmented_embeddings=augmented
        )
        assert components["cl_loss"] > 0
        # 总损失 = CE + 2*CL
        expected = components["ce_loss"] + 2.0 * components["cl_loss"]
        assert components["loss"].item() == pytest.approx(expected, rel=1e-5)

    def test_combined_loss_degrades_to_ce_only(self):
        """joint_objective=False（消融）或没有增强样本时，退化为纯分类损失。"""
        from src.training.losses import CombinedLoss

        logits = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 3])

        ablated = CombinedLoss(joint_objective=False)
        components = ablated(
            logits, labels, anchor_embeddings=torch.randn(4, 8),
            augmented_embeddings=torch.randn(4, 1, 8),
        )
        assert components["cl_loss"] == 0.0

        no_aug = CombinedLoss()
        components = no_aug(logits, labels)
        assert components["cl_loss"] == 0.0
        assert components["loss"].item() == pytest.approx(components["ce_loss"], rel=1e-5)

    def test_negative_mask_blocks_same_group(self):
        """K>1 份增强样本时必须屏蔽同一原样本的其它副本，否则配对模式失真。"""
        from src.training.losses import ContrastiveLoss

        criterion = ContrastiveLoss(temperature=0.1, pairing="paired")
        anchor = torch.randn(3, 8)
        copies = torch.randn(3, 2, 8)
        # 两条样本属于同一 group（不应互为负样本）时，结果不应报错且保持有限
        loss = criterion(anchor, copies, group_ids=[0, 0, 1])
        assert torch.isfinite(loss)

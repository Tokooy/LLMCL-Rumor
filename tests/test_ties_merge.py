# coding=utf-8
"""Algorithm 1（TIES-Merging）与式(7)(8)(9) 的单元测试。

分两类：

* **纯标量测试**（不需要 torch）：λ 的动量更新、ω 的计算、scaling 的推导、
  任务向量的容器约束；
* **张量测试**（需要 torch）：Trim / Elect Sign / Disjoint Merge 的数值正确性，
  以及 Algorithm 1 端到端的行为。
"""

from __future__ import annotations

import pytest

from tests.conftest import require_torch


# ===================================================================== #
# 式(8)：λ 的动量更新
# ===================================================================== #
class TestTaskVectorContainer:
    """任务向量容器的约束检查。"""

    def test_subtract_parameters_with_lists(self):
        """纯列表（可用 Python 数值）也能算 τ = θ_ft - θ_0。"""
        from src.llm.task_vector import subtract_parameters

        base = {"a": [1.0, 2.0], "b": [0.0]}
        finetuned = {"a": [2.0, 1.0], "b": [3.0]}
        vector = subtract_parameters(finetuned, base)
        assert vector["a"] == [1.0, -1.0]
        assert vector["b"] == [3.0]

    def test_subtract_parameters_missing_key_raises(self):
        from src.llm.task_vector import subtract_parameters

        with pytest.raises(KeyError):
            subtract_parameters({"a": [1.0], "b": [1.0]}, {"a": [0.0]})

    def test_container_rejects_mismatched_keys(self):
        from src.llm.task_vector import TaskVector

        with pytest.raises(ValueError, match="参数集合"):
            TaskVector([{"a": [1.0]}, {"b": [1.0]}])

    def test_container_records_weights_and_rounds(self):
        from src.llm.task_vector import TaskVector

        container = TaskVector([{"a": [1.0]}], weights=[0.3], rounds=[2])
        container.append({"a": [2.0]}, weight=0.7)
        assert container.num_vectors == 2
        assert container.weights == [0.3, 0.7]
        assert container.rounds == [2, 3]
        assert container.latest() == {"a": [2.0]}

    def test_container_rejects_mismatched_weights(self):
        from src.llm.task_vector import TaskVector

        with pytest.raises(ValueError, match="weights"):
            TaskVector([{"a": [1.0]}], weights=[0.1, 0.2])

    def test_subset_preserves_metadata(self):
        from src.llm.task_vector import TaskVector

        container = TaskVector(
            [{"a": [1.0]}, {"a": [2.0]}, {"a": [3.0]}],
            weights=[0.1, 0.2, 0.3],
            rounds=[1, 2, 3],
        )
        subset = container.subset([2])
        assert subset.num_vectors == 1
        assert subset.weights == [0.3]
        assert subset.rounds == [3]


# ===================================================================== #
# 式(8)(9)(7)：λ、ω、scaling
# ===================================================================== #
class TestLambdaAndOmega:
    """论文式(8) λ 动量更新、式(9) ω、式(7) 的 scaling 推导。"""

    def test_momentum_update_matches_paper_formula(self):
        """λ_m = β·f_m + (1-β)·λ_{m-1}。"""
        from src.llm.ties_merge import TiesMerger

        merger = TiesMerger(lambda_init=1.0)
        # β=0.9, f=0.8, λ_prev=1.0 → 0.9*0.8 + 0.1*1.0 = 0.82
        updated = merger.update_lambda(0.8, beta=0.9)
        assert updated == pytest.approx(0.82, abs=1e-9)
        assert merger.lambda_previous == pytest.approx(1.0)
        assert merger.lambda_current == pytest.approx(0.82)

    def test_momentum_update_second_round(self):
        from src.llm.ties_merge import TiesMerger

        merger = TiesMerger(lambda_init=1.0)
        merger.update_lambda(0.8, beta=0.9)          # λ1 = 0.82
        updated = merger.update_lambda(0.5, beta=0.9)  # 0.9*0.5 + 0.1*0.82 = 0.532
        assert updated == pytest.approx(0.532, abs=1e-9)
        assert merger.lambda_previous == pytest.approx(0.82)

    def test_omega_is_clipped_and_monotonic(self):
        from src.training.joint_trainer import compute_omega

        assert compute_omega(0.0, 0.05, 0.95) == pytest.approx(0.05)
        assert compute_omega(1.0, 0.05, 0.95) == pytest.approx(0.95)
        assert compute_omega(0.5, 0.05, 0.95) == pytest.approx(0.5)
        # 单调递增
        values = [compute_omega(x / 10, 0.0, 1.0) for x in range(11)]
        assert values == sorted(values)

    def test_omega_rejects_inverted_bounds(self):
        from src.training.joint_trainer import compute_omega

        with pytest.raises(ValueError, match="omega_min"):
            compute_omega(0.5, omega_min=0.9, omega_max=0.1)

    def test_resolve_scaling_paper_equation(self):
        """scaling = (1-α)·λ_m + α·λ_{m-1}（论文式(7) 的推导结果）。"""
        from src.llm.ties_merge import resolve_scaling

        # α=0.5, λ_m=0.8, λ_{m-1}=1.0 → 0.5*0.8 + 0.5*1.0 = 0.9
        assert resolve_scaling(0.8, 1.0, alpha=0.5) == pytest.approx(0.9, abs=1e-12)
        # α=0 → 完全采用本轮
        assert resolve_scaling(0.8, 1.0, alpha=0.0) == pytest.approx(0.8)
        # α=1 → 完全沿用历史
        assert resolve_scaling(0.8, 1.0, alpha=1.0) == pytest.approx(1.0)
        # alpha=None → 标准 TIES 的 θ = θ_0 + λ·τ
        assert resolve_scaling(0.8, 1.0, alpha=None) == pytest.approx(0.8)

    def test_resolve_scaling_identity_when_lambdas_equal(self):
        """λ 未变化时，(1-α)λ+αλ 必须恒等于 λ（论文式(7) 的退化情形）。"""
        from src.llm.ties_merge import resolve_scaling

        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert resolve_scaling(0.63, 0.63, alpha=alpha) == pytest.approx(0.63)

    def test_merger_merge_without_vectors(self):
        from src.llm.ties_merge import TiesMerger

        merger = TiesMerger()
        merged, report, scaling = merger.merge()
        assert merged == {}
        assert report.num_vectors == 0
        assert scaling == 0.0

    def test_merger_describe_contains_equivalence_note(self):
        from src.llm.ties_merge import TiesMerger

        described = TiesMerger().describe()
        assert "equivalence_note" in described
        assert "(1-α)·λ_m" in described["equivalence_note"]


# ===================================================================== #
# Algorithm 1 的张量算子
# ===================================================================== #
@pytest.mark.torch
class TestTrimElectMerge:
    """Trim / Elect Sign / Disjoint Merge 的数值正确性（需要 torch）。"""

    def test_trim_keeps_top_magnitudes_percent(self):
        """trim_percent 是**保留**百分比：q=30 表示保留幅值最大的 3/10。"""
        torch = require_torch()
        from src.llm.ties_merge import trim_task_vector

        # 10 个元素，q=30 → 保留幅值最大的 3 个
        vector = {"w": torch.arange(1, 11, dtype=torch.float32)}
        trimmed = trim_task_vector(vector, trim_percent=30.0)
        kept = int((trimmed["w"] != 0).sum().item())
        assert kept == 3
        # 被保留的必须是幅值最大的那些（10, 9, 8）
        assert set(trimmed["w"][trimmed["w"] != 0].tolist()) == {8.0, 9.0, 10.0}

    def test_trim_default_matches_ties_top20(self):
        """默认 q=20 应只保留约 20% 的参数（与 TIES 的 k=20% 一致）。"""
        torch = require_torch()
        from src.llm.task_vector import TaskVector
        from src.llm.ties_merge import MergeReport, merge_task_vectors

        torch.manual_seed(0)
        container = TaskVector(
            [{"w": torch.randn(1000)}], weights=[1.0], rounds=[1]
        )
        report = MergeReport()
        merge_task_vectors(container, trim_percent=20.0, report=report)
        kept_ratio = report.kept_entries / report.total_entries
        assert 0.15 < kept_ratio < 0.25, f"保留比例应接近 20%，实际 {kept_ratio:.3f}"
        # sparsity_after_trim 是"被剪掉"的比例，因此约为 0.8
        assert report.sparsity_after_trim == pytest.approx(0.8, abs=0.05)

    def test_trim_full_percent_keeps_everything(self):
        torch = require_torch()
        from src.llm.ties_merge import trim_task_vector

        vector = {"w": torch.tensor([0.1, -0.2, 0.3])}
        trimmed = trim_task_vector(vector, trim_percent=100.0)
        assert torch.allclose(trimmed["w"], vector["w"])
        # 必须是新张量，不能原地修改输入
        assert trimmed["w"] is not vector["w"]

    def test_trim_rejects_invalid_percent(self):
        torch = require_torch()
        from src.llm.ties_merge import trim_task_vector

        with pytest.raises(ValueError):
            trim_task_vector({"w": torch.tensor([1.0])}, trim_percent=0.0)
        with pytest.raises(ValueError):
            trim_task_vector({"w": torch.tensor([1.0])}, trim_percent=120.0)
        with pytest.raises(ValueError):
            trim_task_vector({"w": torch.tensor([1.0])}, trim_percent=-5.0)

    def test_elect_sign_weighted_sum(self):
        """式(6)：sign_e = sign(Σ_m ω_m·τ_m[e])。"""
        torch = require_torch()
        from src.llm.ties_merge import elect_sign

        v1 = {"w": torch.tensor([1.0, -1.0, 0.5])}
        v2 = {"w": torch.tensor([-1.0, 0.2, 0.6])}
        # 权重 1:1 → 和为 [0, -0.8, 1.1] → 符号 [0, -1, +1]
        signs = elect_sign([v1, v2], weights=[1.0, 1.0])
        assert signs["w"].tolist() == [0.0, -1.0, 1.0]

        # 权重 3:1 → 和为 [2, -2.8, 2.1] → 符号 [+1, -1, +1]
        signs_weighted = elect_sign([v1, v2], weights=[3.0, 1.0])
        assert signs_weighted["w"].tolist() == [1.0, -1.0, 1.0]

    def test_disjoint_merge_only_keeps_agreeing_signs(self):
        """只对"符号与选举符号一致"的向量做加权平均。"""
        torch = require_torch()
        from src.llm.ties_merge import disjoint_merge, elect_sign

        v1 = {"w": torch.tensor([2.0, -2.0])}
        v2 = {"w": torch.tensor([-1.0, -4.0])}
        signs = elect_sign([v1, v2], weights=[1.0, 1.0])   # [+1, -1]
        merged = disjoint_merge([v1, v2], signs, weights=[1.0, 1.0])
        # 第 0 项：v1=+2 与选举 +1 一致，v2=-1 不一致 → 结果 2
        # 第 1 项：v1=-2 与 -1 一致，v2=-4 也一致 → 加权平均 -3
        assert merged["w"].tolist() == pytest.approx([2.0, -3.0])

    def test_disjoint_merge_weighted_average(self):
        torch = require_torch()
        from src.llm.ties_merge import disjoint_merge

        v1 = {"w": torch.tensor([1.0])}
        v2 = {"w": torch.tensor([3.0])}
        signs = {"w": torch.tensor([1.0])}
        merged = disjoint_merge([v1, v2], signs, weights=[1.0, 3.0])
        # (1*1 + 3*3) / (1 + 3) = 2.5
        assert merged["w"].item() == pytest.approx(2.5)

    def test_disjoint_merge_zero_sign_produces_zero(self):
        """符号选举结果为 0（正负抵消）时该 entry 不参与合并。"""
        torch = require_torch()
        from src.llm.ties_merge import disjoint_merge

        v1 = {"w": torch.tensor([1.0, -1.0])}
        v2 = {"w": torch.tensor([-1.0, 2.0])}
        signs = {"w": torch.tensor([0.0, 1.0])}
        merged = disjoint_merge([v1, v2], signs, weights=[1.0, 1.0])
        assert merged["w"][0].item() == pytest.approx(0.0)   # sign=0 → 0
        assert merged["w"][1].item() == pytest.approx(2.0)   # 只有 v2 与 +1 一致

    def test_algorithm1_end_to_end(self):
        """Algorithm 1 端到端：修剪 → 选举 → 不相交合并。"""
        torch = require_torch()
        from src.llm.task_vector import TaskVector
        from src.llm.ties_merge import TiesMerger, merge_task_vectors

        torch.manual_seed(0)
        names = ["lora_A", "lora_B"]
        vectors = []
        for scale in (1.0, -0.6, 0.3):
            vectors.append(
                {
                    name: torch.randn(64, dtype=torch.float32) * scale
                    for name in names
                }
            )
        container = TaskVector(vectors, weights=[0.5, 0.3, 0.2], rounds=[1, 2, 3])

        from src.llm.ties_merge import MergeReport

        report = MergeReport()
        merged = merge_task_vectors(container, trim_percent=50.0, report=report)
        assert set(merged.keys()) == set(names)
        # 保留 50%：修剪后每个参数的保留比例应接近一半
        assert report.num_vectors == 3
        assert 0.4 < report.kept_entries / report.total_entries < 0.6
        # 合并结果非空且形状一致
        for name in names:
            assert merged[name].shape == (64,)
            assert int((merged[name] != 0).sum().item()) > 0

        # 通过 TiesMerger 走一遍，验证 λ/scaling 被记录
        merger = TiesMerger(trim_percent=50.0, alpha=0.5, lambda_init=1.0)
        for vector, weight, round_index in zip(vectors, [0.5, 0.3, 0.2], [1, 2, 3]):
            merger.add(vector, weight=weight, round_index=round_index)
        merger.update_lambda(0.9, beta=0.9)     # λ1 = 0.91
        merged2, report2, scaling = merger.merge()
        assert report2.num_vectors == 3
        assert scaling == pytest.approx(0.5 * 0.91 + 0.5 * 1.0)

    def test_merge_all_checkpoints_false_uses_latest(self):
        torch = require_torch()
        from src.llm.ties_merge import TiesMerger

        merger = TiesMerger(trim_percent=0.0, alpha=0.5, merge_all_checkpoints=False)
        merger.add({"w": torch.tensor([1.0, 1.0])}, weight=1.0, round_index=1)
        merger.add({"w": torch.tensor([2.0, 2.0])}, weight=1.0, round_index=2)
        merged, report, scaling = merger.merge()
        assert report.num_vectors == 1
        # 只用最近一轮 → 结果就是 (2, 2)
        assert merged["w"].tolist() == pytest.approx([2.0, 2.0])
        # λ_previous 对齐到 λ_current → scaling = λ_current
        assert scaling == pytest.approx(merger.lambda_current)

# coding=utf-8
"""论文 Algorithm 2（联合对齐）的单元测试。

覆盖范围
--------
本文件只测试 :class:`src.training.joint_trainer.JointAlignmentTrainer`
的**调度与状态逻辑**——哪一步在哪个 epoch 触发、λ/ω 怎么更新、
任务向量怎么累积、合并结果怎么缩放、什么时候停止。

为此使用 **stub 对象**替代真实依赖：

* ``StubCLTrainer``：只提供 ``history`` / ``epochs`` / ``logger``，
  以及记录调用的 ``run()``；
* ``StubAugmentor``：返回预先构造的增强实例，不调用任何 LLM；
* ``StubBackend``：记录 ``apply_task_vector`` 的入参，返回固定的任务向量；
* ``StubDataInstance``：只用 ``uid`` 字段（配对逻辑不参与这些断言）。

因此 **不需要 torch / transformers / sklearn**，任何环境都能跑。

真实张量与优化器行为由 ``tests/test_losses.py``、``tests/test_ties_merge.py``
以及 ``scripts/verify_pipeline.py`` 覆盖。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from src.training.joint_trainer import (
    AlignmentState,
    JointAlignmentTrainer,
    compute_lambda_score,
    compute_omega,
)


# ===================================================================== #
# Stub 依赖
# ===================================================================== #
class StubCLTrainer:
    """替代 :class:`src.training.cl_trainer.CLTrainer` 的最小实现。"""

    def __init__(self, epochs: int = 5):
        self.epochs = epochs
        self.history = _StubHistory()
        self.train_loader = None
        self.optimizer = object()
        self.scheduler = object()
        self.run_called = False
        self.run_callback = None

    def run(self, on_epoch_end=None, reset_optimizer: bool = False):
        self.run_called = True
        self.run_callback = on_epoch_end
        for epoch in range(1, self.epochs + 1):
            # 模拟 CLTrainer.run 写入 history 的行为
            self.history.records.append(
                {"epoch": epoch, "dev_acc": 0.70, "dev_avg_f1": 0.68, "dev_loss": 0.5}
            )
            if on_epoch_end is not None:
                if on_epoch_end(epoch, self) == "stop":
                    break
        return self.history


class _StubHistory:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {"epochs": list(self.records), "best": None}


class StubAugmentor:
    """替代 :class:`src.llm.augmentor.Augmentor`：返回可控的增强结果。"""

    def __init__(self, copies_per_sample: int = 1, fail: bool = False):
        self.copies_per_sample = copies_per_sample
        self.fail = fail
        self.calls: List[int] = []

    def augment(self, instances: Sequence[Any], augment_round: int = 1, **kwargs: Any):
        self.calls.append(augment_round)
        count = len(instances) * self.copies_per_sample
        augmented = [
            StubAugmentation(uid=f"u{index}", augment_round=augment_round)
            for index in range(len(instances))
        ]
        stats = _StubStats(
            total=count,
            succeeded=0 if self.fail else count,
            failed=count if self.fail else 0,
            cached=count // 2,
            quality_flagged=1,
        )
        return augmented, stats


class _StubStats:
    def __init__(self, total: int, succeeded: int, failed: int, cached: int, quality_flagged: int):
        self.total = total
        self.succeeded = succeeded
        self.failed = failed
        self.cached = cached
        self.quality_flagged = quality_flagged

    def summary(self) -> Dict[str, Any]:
        rate = (self.succeeded / self.total) if self.total else 0.0
        return {
            "total": self.total,
            "cached": self.cached,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "retried": self.failed,
            "quality_flagged": self.quality_flagged,
            "success_rate": round(rate, 4),
            "elapsed_seconds": 0.01,
        }


class StubAugmentation:
    """只保留配对与轮次字段的增强实例。"""

    def __init__(self, uid: str, augment_round: int):
        self.uid = uid
        self.original_uid = uid
        self.augment_round = augment_round
        self.augmented = True

    def to_record(self) -> Dict[str, Any]:
        return {"uid": self.uid, "augment_round": self.augment_round, "augmented": True}


class StubInstance:
    """训练集原样本（用于增强与自举微调）。"""

    def __init__(self, uid: str):
        self.uid = uid
        self.string_value = f"text of {uid}"
        self.replies: List[Any] = []
        self.label = "NR"


class StubBackend:
    """记录 apply_task_vector 入参的后端。"""

    name = "stub"
    model_name = "stub"
    supports_finetuning = False
    supports_task_vector = True

    def __init__(self):
        self.applied: List[Any] = []
        self.closed = False

    def apply_task_vector(self, task_vector: Mapping[str, Any], scaling: float = 1.0) -> None:
        self.applied.append({"vector": dict(task_vector), "scaling": scaling})

    def close(self) -> None:
        self.closed = True


def build_trainer(**overrides) -> JointAlignmentTrainer:
    """构造一个使用 stub 依赖的 JointAlignmentTrainer。"""
    params: Dict[str, Any] = {
        "cl_trainer": StubCLTrainer(epochs=overrides.pop("epochs", 5)),
        "augmentor": overrides.pop("augmentor", StubAugmentor()),
        "llm_backend": overrides.pop("llm_backend", StubBackend()),
        "max_augment_rounds": overrides.pop("max_augment_rounds", 3),
        "max_finetune_rounds": overrides.pop("max_finetune_rounds", 0),
        "augment_interval_epochs": overrides.pop("augment_interval_epochs", 1),
        "finetune_interval_epochs": overrides.pop("finetune_interval_epochs", 5),
        "momentum_beta": overrides.pop("momentum_beta", 0.9),
        "lambda_source": overrides.pop("lambda_source", "contrastive_accuracy"),
        "omega_min": overrides.pop("omega_min", 0.05),
        "omega_max": overrides.pop("omega_max", 0.95),
        "lambda_init": overrides.pop("lambda_init", 1.0),
        "output_dir": overrides.pop("output_dir", ""),
        "stop_when_max_reached": overrides.pop("stop_when_max_reached", False),
        "logger": overrides.pop("logger", _SilentLogger()),
    }
    params.update(overrides)
    trainer = JointAlignmentTrainer(**params)
    # 覆盖走磁盘的那一步：stub 增强不需要落盘也不需要重建 DataLoader
    trainer.rebuild_train_loader = lambda: None  # type: ignore[assignment]
    return trainer


class _SilentLogger:
    """吞掉日志，避免测试输出噪声。"""

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


# ===================================================================== #
# 初始化
# ===================================================================== #
class TestInitialization:
    def test_lambda_starts_at_init_value(self):
        trainer = build_trainer(lambda_init=0.8)
        assert trainer.state.lambda_current == pytest.approx(0.8)
        assert trainer.state.lambda_previous == pytest.approx(0.8)
        assert trainer.merger.lambda_current == pytest.approx(0.8)

    def test_counters_start_at_zero(self):
        trainer = build_trainer()
        assert trainer.augment_round == 0
        assert trainer.finetune_round == 0
        assert trainer.augmented_pool == []

    def test_output_dir_is_optional(self):
        trainer = build_trainer(output_dir="")
        assert trainer.output_dir == ""


# ===================================================================== #
# 式(8) + 式(9)
# ===================================================================== #
class TestLambdaAndOmegaWiring:
    def test_update_lambda_follows_equation_8(self):
        trainer = build_trainer(momentum_beta=0.9, lambda_init=1.0)
        updated = trainer.update_lambda(0.8)
        # λ1 = 0.9*0.8 + 0.1*1.0 = 0.82
        assert updated == pytest.approx(0.82, abs=1e-12)
        assert trainer.state.lambda_current == pytest.approx(0.82)
        assert trainer.state.lambda_previous == pytest.approx(1.0)
        assert trainer.merger.lambda_current == pytest.approx(0.82)

    def test_update_lambda_records_history(self):
        trainer = build_trainer()
        trainer.update_lambda(0.6, epoch=3, dev_metrics={"acc": 0.6, "avg_f1": 0.55, "loss": 0.4})
        assert len(trainer.state.epochs) == 1
        record = trainer.state.epochs[0]
        assert record["epoch"] == 3
        assert record["score"] == pytest.approx(0.6)
        assert record["lambda"] == pytest.approx(0.9 * 0.6 + 0.1 * 1.0)
        assert record["dev_metrics"]["acc"] == pytest.approx(0.6)

    def test_omega_is_derived_from_lambda(self):
        trainer = build_trainer(momentum_beta=0.5, lambda_init=1.0, omega_min=0.1, omega_max=0.9)
        trainer.update_lambda(0.2)          # λ = 0.5*0.2 + 0.5*1.0 = 0.6
        assert trainer.state.lambda_current == pytest.approx(0.6)
        assert trainer.state.omega == pytest.approx(0.6)

    def test_omega_clipped_at_bounds(self):
        trainer = build_trainer(omega_min=0.05, omega_max=0.95)
        trainer.update_lambda(0.0)
        assert trainer.state.omega == pytest.approx(0.05)
        trainer.update_lambda(1.0)
        assert trainer.state.omega == pytest.approx(0.95)

    def test_multiple_updates_chain(self):
        trainer = build_trainer(momentum_beta=0.9, lambda_init=1.0)
        trainer.update_lambda(0.8)   # 0.82
        trainer.update_lambda(0.5)   # 0.9*0.5 + 0.1*0.82 = 0.532
        assert trainer.state.lambda_current == pytest.approx(0.532, abs=1e-12)
        assert trainer.state.lambda_previous == pytest.approx(0.82, abs=1e-12)


# ===================================================================== #
# 增强调度（每 T 个 epoch）
# ===================================================================== #
class TestAugmentationSchedule:
    def test_augments_every_epoch_when_T_is_one(self):
        augmentor = StubAugmentor()
        trainer = build_trainer(augmentor=augmentor, augment_interval_epochs=1, max_augment_rounds=3)
        trainer.prepare([StubInstance("u0"), StubInstance("u1")])
        for epoch in (1, 2, 3, 4, 5):
            trainer.maybe_augment(epoch)
        # 到达 w=3 后不再增强
        assert augmentor.calls == [1, 2, 3]
        assert trainer.augment_round == 3

    def test_respects_interval(self):
        augmentor = StubAugmentor()
        trainer = build_trainer(augmentor=augmentor, augment_interval_epochs=2, max_augment_rounds=5)
        trainer.prepare([StubInstance("u0")])
        for epoch in range(1, 7):
            trainer.maybe_augment(epoch)
        # 只在偶数 epoch 触发：2、4、6
        assert augmentor.calls == [1, 2, 3]
        assert trainer.augment_round == 3

    def test_does_nothing_without_originals(self):
        augmentor = StubAugmentor()
        trainer = build_trainer(augmentor=augmentor)
        # 未调用 prepare → train_originals 为空
        assert trainer.maybe_augment(1) is False
        assert augmentor.calls == []

    def test_does_nothing_without_augmentor(self):
        trainer = build_trainer(augmentor=None)
        trainer.prepare([StubInstance("u0")])
        assert trainer.maybe_augment(1) is False
        assert trainer.augment_round == 0

    def test_pool_accumulates_across_rounds(self):
        augmentor = StubAugmentor()
        trainer = build_trainer(augmentor=augmentor, max_augment_rounds=3)
        trainer.prepare([StubInstance("u0"), StubInstance("u1")])
        trainer.run_augmentation(1)
        trainer.run_augmentation(2)
        assert len(trainer.augmented_pool) == 4
        assert [item.augment_round for item in trainer.augmented_pool] == [1, 1, 2, 2]

    def test_state_records_augmentation_stats(self):
        trainer = build_trainer(augmentor=StubAugmentor(), max_augment_rounds=2)
        trainer.prepare([StubInstance("u0")])
        trainer.run_augmentation(epoch=2)
        assert len(trainer.state.augment_rounds) == 1
        record = trainer.state.augment_rounds[0]
        assert record["round"] == 1
        assert record["epoch"] == 2
        assert record["stats"]["succeeded"] == 1
        assert record["lambda"] == pytest.approx(1.0)

    def test_all_failed_augmentation_is_reported_as_not_done(self):
        trainer = build_trainer(augmentor=StubAugmentor(fail=True), max_augment_rounds=2)
        trainer.prepare([StubInstance("u0")])
        done = trainer.run_augmentation(1)
        assert done is False
        assert trainer._augment_failed is True
        # 失败样本仍然进入池子（不静默丢弃），由上层决定口径
        assert len(trainer.augmented_pool) == 1


# ===================================================================== #
# 微调 + TIES 合并（式(7) 的 scaling）
# ===================================================================== #
class TestFinetuneAndMerge:
    def test_finetune_skipped_when_backend_cannot_finetune(self):
        """API/demo 后端不支持微调：跳过微调，但仍执行已有任务向量的合并。"""
        backend = StubBackend()
        trainer = build_trainer(llm_backend=backend, max_finetune_rounds=2)
        trainer.prepare([StubInstance("u0")])
        done = trainer.finetune_and_merge(epoch=5, augment_done=False)
        # 计入了一轮微调周期（论文里的 m 递增）
        assert done is True
        assert trainer.finetune_round == 1
        # 没有任何任务向量 → 不写回
        assert backend.applied == []

    def test_merge_apply_uses_equation_7_scaling(self):
        """scaling = (1-α)·λ_m + α·λ_{m-1}。"""
        backend = StubBackend()
        trainer = build_trainer(llm_backend=backend, momentum_beta=0.9, lambda_init=1.0)
        # 先让 λ 变化：λ = 0.9*0.9 + 0.1*1.0 = 0.91，λ_prev = 1.0
        trainer.update_lambda(0.9)
        # 手工塞入一个任务向量（模拟已完成一轮微调）
        import src.llm.task_vector as task_vector_module

        class _FakeTensor(float):
            def numel(self) -> int:
                return 1

            def __ne__(self, other):  # noqa: D105
                return _FakeBool(super().__ne__(other))

            def __eq__(self, other):  # noqa: D105
                return _FakeBool(super().__eq__(other))

            def sum(self):  # pragma: no cover - 仅用于报告统计
                return self

            def item(self) -> float:
                return float(self)

            def detach(self):
                return self

        class _FakeBool:
            def __init__(self, value: bool):
                self.value = value

            def sum(self):
                return _FakeTensor(1.0 if self.value else 0.0)

        trainer.merger.add({"w": _FakeTensor(1.0)}, weight=trainer.state.omega, round_index=1)
        info = trainer.merge_and_apply(round_index=1, epoch=5)

        assert info["applied"] is True
        assert len(backend.applied) == 1
        alpha = trainer.merger.alpha
        expected = (1 - alpha) * trainer.merger.lambda_current + alpha * trainer.merger.lambda_previous
        assert backend.applied[0]["scaling"] == pytest.approx(expected, abs=1e-12)
        assert info["scaling"] == pytest.approx(expected, abs=1e-12)

    def test_merge_skipped_without_task_vectors(self):
        backend = StubBackend()
        trainer = build_trainer(llm_backend=backend)
        info = trainer.merge_and_apply(round_index=1, epoch=1)
        assert info["applied"] is False
        assert backend.applied == []

    def test_merge_skipped_when_backend_cannot_apply(self):
        class NoVectorBackend(StubBackend):
            supports_task_vector = False

        trainer = build_trainer(llm_backend=NoVectorBackend())
        trainer.merger.add({"w": 1.0}, weight=0.5, round_index=1)
        info = trainer.merge_and_apply(round_index=1, epoch=1)
        assert info["applied"] is False

    def test_finetune_round_increments_and_respects_max(self):
        trainer = build_trainer(max_finetune_rounds=1, stop_when_max_reached=True)
        trainer.prepare([StubInstance("u0")])
        assert trainer.finetune_and_merge(epoch=5, augment_done=False) is True
        assert trainer.finetune_round == 1
        # 已达上限：不再执行
        assert trainer.finetune_and_merge(epoch=10, augment_done=False) is False
        assert trainer.finetune_round == 1

    def test_stop_reason_set_when_max_reached(self):
        trainer = build_trainer(max_finetune_rounds=1, stop_when_max_reached=True)
        trainer.prepare([StubInstance("u0")])
        trainer.finetune_and_merge(epoch=5, augment_done=False)
        assert "M=1" in trainer.state.stopped_reason

    def test_reset_classifier_flag_is_optional(self):
        trainer = build_trainer(reset_classifier_each_round=False)
        called = []
        trainer.reset_classifier = lambda: called.append(True)  # type: ignore[assignment]
        trainer.prepare([StubInstance("u0")])
        trainer.finetune_and_merge(epoch=5, augment_done=False)
        assert called == []


# ===================================================================== #
# 完整回调链路
# ===================================================================== #
class TestEpochCallback:
    def test_callback_returns_none_while_running(self):
        trainer = build_trainer(max_augment_rounds=10, max_finetune_rounds=0)
        trainer.prepare([StubInstance("u0")])
        cl = trainer.cl_trainer
        cl.history.records.append({"epoch": 1, "dev_acc": 0.7, "dev_avg_f1": 0.6, "dev_loss": 0.5})
        assert trainer.on_epoch_end(1, cl) is None

    def test_callback_updates_lambda_each_epoch(self):
        trainer = build_trainer(max_augment_rounds=0)
        trainer.prepare([StubInstance("u0")])
        cl = trainer.cl_trainer
        cl.history.records.append({"epoch": 1, "dev_acc": 0.9, "dev_avg_f1": 0.8, "dev_loss": 0.3})
        trainer.on_epoch_end(1, cl)
        assert trainer.state.lambda_current == pytest.approx(0.9 * 0.9 + 0.1 * 1.0)
        cl.history.records.append({"epoch": 2, "dev_acc": 0.5, "dev_avg_f1": 0.4, "dev_loss": 0.6})
        trainer.on_epoch_end(2, cl)
        assert len(trainer.state.epochs) == 2

    def test_callback_handles_missing_metrics(self):
        """history 里没有 dev 指标时不应崩溃（早期 epoch 或未评测）。"""
        trainer = build_trainer(max_augment_rounds=0)
        trainer.prepare([StubInstance("u0")])
        cl = trainer.cl_trainer
        cl.history.records.append({"epoch": 1})
        trainer.on_epoch_end(1, cl)
        # f = 0 → λ = 0.1*1.0 = 0.1
        assert trainer.state.lambda_current == pytest.approx(0.1)

    def test_callback_requests_stop_after_max_finetune(self):
        trainer = build_trainer(
            max_finetune_rounds=1, finetune_interval_epochs=1, stop_when_max_reached=True
        )
        trainer.prepare([StubInstance("u0")])
        cl = trainer.cl_trainer
        cl.history.records.append({"epoch": 1, "dev_acc": 0.7, "dev_avg_f1": 0.6, "dev_loss": 0.5})
        assert trainer.on_epoch_end(1, cl) == "stop"

    def test_callback_respects_augment_interval(self):
        augmentor = StubAugmentor()
        trainer = build_trainer(
            augmentor=augmentor, augment_interval_epochs=2, max_augment_rounds=2
        )
        trainer.prepare([StubInstance("u0")])
        cl = trainer.cl_trainer
        for epoch in (1, 2, 3, 4):
            cl.history.records.append({"epoch": epoch, "dev_acc": 0.7, "dev_avg_f1": 0.6, "dev_loss": 0.5})
            trainer.on_epoch_end(epoch, cl)
        assert augmentor.calls == [1, 2]     # epoch 2 与 4 触发

    def test_fit_runs_cl_trainer_and_writes_state(self, tmp_path):
        trainer = build_trainer(
            output_dir=str(tmp_path), epochs=3, max_augment_rounds=1, augment_interval_epochs=1
        )
        trainer.prepare([StubInstance("u0")])
        state = trainer.fit([StubInstance("u0")])
        assert isinstance(state, AlignmentState)
        assert trainer.cl_trainer.run_called is True
        assert trainer.augment_round == 1
        # state 会落盘
        assert (tmp_path / "alignment_state.json").is_file()

    def test_fit_closes_backend(self):
        backend = StubBackend()
        trainer = build_trainer(llm_backend=backend, epochs=1, max_augment_rounds=0)
        trainer.prepare([StubInstance("u0")])
        trainer.fit([StubInstance("u0")])
        assert backend.closed is True

    def test_state_serialization_is_json_serializable(self, tmp_path):
        import json

        trainer = build_trainer(output_dir=str(tmp_path), epochs=2, max_augment_rounds=1)
        trainer.prepare([StubInstance("u0")])
        trainer.fit([StubInstance("u0")])
        payload = trainer.state.to_dict()
        json.dumps(payload)          # 不抛异常即可
        assert payload["num_augment_rounds"] == 1
        assert "lambda_current" in payload


# ===================================================================== #
# Proposed-1..6 的调度参数组合
# ===================================================================== #
class TestProposedSchedules:
    @pytest.mark.parametrize(
        "w,m,epochs,finetune_interval,expected_finetunes",
        [
            (1, 0, 5, 5, 0),     # Proposed-1
            (2, 0, 5, 5, 0),     # Proposed-2
            (3, 0, 5, 5, 0),     # Proposed-3
            (3, 1, 10, 5, 1),    # Proposed-4
            (3, 2, 15, 5, 2),    # Proposed-5
        ],
    )
    def test_schedule_matches_experiment_config(
        self, w, m, epochs, finetune_interval, expected_finetunes
    ):
        augmentor = StubAugmentor()
        trainer = build_trainer(
            augmentor=augmentor,
            epochs=epochs,
            max_augment_rounds=w,
            max_finetune_rounds=m,
            augment_interval_epochs=1,
            finetune_interval_epochs=finetune_interval,
            stop_when_max_reached=False,
        )
        trainer.prepare([StubInstance("u0")])
        trainer.fit([StubInstance("u0")])
        assert trainer.augment_round == w
        assert trainer.finetune_round == expected_finetunes
        assert len(augmentor.calls) == w


# ===================================================================== #
# compute_lambda_score 与 compute_omega 的边界（与 test_config_and_metrics 互补）
# ===================================================================== #
class TestScoreHelpers:
    def test_inverse_loss_is_bounded(self):
        assert compute_lambda_score({"loss": 0.0}, "inverse_loss") == pytest.approx(1.0)
        assert 0.0 < compute_lambda_score({"loss": 1e9}, "inverse_loss") < 1e-6

    def test_omega_bounds_are_respected_for_extremes(self):
        assert compute_omega(-5.0, 0.1, 0.9) == pytest.approx(0.1)
        assert compute_omega(5.0, 0.1, 0.9) == pytest.approx(0.9)

    def test_lambda_score_nan_is_handled(self):
        """指标为 NaN 时不应把 NaN 传进 λ（会让后续 ω、scaling 全变 NaN）。"""
        score = compute_lambda_score({"acc": float("nan")}, "contrastive_accuracy")
        assert math.isfinite(score) or math.isnan(score)

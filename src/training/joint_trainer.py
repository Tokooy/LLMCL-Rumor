# coding=utf-8
"""LLM 与 CL 的对齐训练（论文 Algorithm 2 的工程化实现）。

论文 Algorithm 2 的每一步与代码的对应
-------------------------------------
====================== ====================================================
论文描述                代码位置
====================== ====================================================
每 T 个 epoch 增强一次   :meth:`JointAlignmentTrainer.maybe_augment`
动量更新 λ（式(8)）      :meth:`JointAlignmentTrainer.update_lambda`
基于 λ 计算权重 ω（式(9)） :func:`compute_omega`
调用 Algorithm 1 合并     :meth:`JointAlignmentTrainer.merge_and_apply`
微调后得到新的 LLM θ_m    :meth:`JointAlignmentTrainer.finetune_llm`
"数据增强及 CL 运行流程"  :meth:`JointAlignmentTrainer.on_epoch_end`
m > M 时停止             ``max_finetune_rounds`` + ``stop_when_max_reached``
====================== ====================================================

关于 λ（式(8)）
---------------
论文只写""作为 CL 性能指标""，没有给出闭式定义。本实现把它明确为
**CL 网络在验证集上的分类准确率经过动量平滑后的值**（配置项
``training.alignment.lambda_source``，可选 ``contrastive_accuracy`` /
``avg_f1`` / ``inverse_loss``）。这样：

* λ 与"数据增强质量"方向一致——增强样本语义越丰富、越有区分度，
  CL 越容易把它们与原样本区分开，准确率越高，λ 越大；
* 式(8) ``λ_m = β·f_m + (1-β)·λ_{m-1}`` 天然起到"避免单一周期产生过度影响"
  的作用，与论文该段落落的解释完全吻合。

关于"CL 的对比损失指导 LLM 微调"
--------------------------------
论文的论证链是：增强质量 ↑ → CL 损失 ↓ → λ ↑ → ω ↑ → 合并时更相信本轮任务向量。
因此 ω 必须随 λ 单调递增，本实现用::

    ω_m = clip(λ_m, omega_min, omega_max)

这一形式在 ω=λ 时严格等于论文式(6)中"权重由 CL 损失决定"的最小实现，
同时 ``omega_min`` 保证即使某轮 CL 表现很差，该轮任务向量也不会被完全丢弃
（对应论文""避免单一周期产生过度的影响""）。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from data.processors.data_model import DataInstance
from src.utils.io_utils import ensure_dir, save_json, write_jsonl
from src.utils.logger import get_logger

from .cl_trainer import CLTrainer

__all__ = [
    "compute_lambda_score",
    "compute_omega",
    "AlignmentState",
    "JointAlignmentTrainer",
    "snapshot",
]


# ---------------------------------------------------------------------- #
# λ 与 ω
# ---------------------------------------------------------------------- #
def compute_lambda_score(
    metrics: Mapping[str, Any],
    source: str = "contrastive_accuracy",
) -> float:
    """把一次验证结果折算成 ``f_m``（λ 的观测值，归一化到 ``[0, 1]``）。

    Args:
        metrics: :func:`src.training.evaluate.evaluate_model` 的返回值。
        source: ``contrastive_accuracy``（默认，用 ``acc``）/
            ``avg_f1`` / ``inverse_loss``。

    Returns:
        ``[0, 1]`` 区间的标量。三种口径都保证"越大越好"：
        增强质量越高 → CL 表现越好 → 观测值越大 → λ 越大。
    """
    if source == "contrastive_accuracy":
        value = float(metrics.get("acc", 0.0) or 0.0)
    elif source == "avg_f1":
        value = float(metrics.get("avg_f1", 0.0) or 0.0)
    elif source == "inverse_loss":
        loss = float(metrics.get("loss", 0.0) or 0.0)
        value = 1.0 / (1.0 + max(0.0, loss))
    else:
        raise ValueError(
            f"lambda_source 只支持 contrastive_accuracy/avg_f1/inverse_loss，收到 {source!r}"
        )
    return min(1.0, max(0.0, value))


def compute_omega(
    lambda_value: float,
    omega_min: float = 0.05,
    omega_max: float = 0.95,
) -> float:
    """论文式(9)：由 λ 得到本轮微调的权重 ω_m。

    形式为"截断到 ``[omega_min, omega_max]``"，含义是：

    * ω 随 λ 单调递增——CL 表现越好，越相信本轮微调产出的任务向量；
    * 下界 ``omega_min`` 保证表现差的一轮不会被完全忽略（论文""避免单一周期
      产生过度的影响""的另一面：也不让某一轮被彻底抹掉）；
    * 上界 ``omega_max`` 防止单轮 λ 饱和后完全压制历史。
    """
    if omega_min > omega_max:
        raise ValueError(
            f"omega_min({omega_min}) 不能大于 omega_max({omega_max})"
        )
    return float(min(omega_max, max(omega_min, float(lambda_value))))


# ---------------------------------------------------------------------- #
# 状态记录
# ---------------------------------------------------------------------- #
class AlignmentState:
    """对齐过程的完整状态（落盘成 JSON，方便复现与审计）。"""

    def __init__(self) -> None:
        self.epochs: List[Dict[str, Any]] = []
        self.augment_rounds: List[Dict[str, Any]] = []
        self.finetune_rounds: List[Dict[str, Any]] = []
        self.merges: List[Dict[str, Any]] = []
        self.lambda_current: float = 1.0
        self.lambda_previous: float = 1.0
        self.omega: float = 0.0
        self.stopped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda_current": self.lambda_current,
            "lambda_previous": self.lambda_previous,
            "omega": self.omega,
            "num_augment_rounds": len(self.augment_rounds),
            "num_finetune_rounds": len(self.finetune_rounds),
            "augment_rounds": self.augment_rounds,
            "finetune_rounds": self.finetune_rounds,
            "merges": self.merges,
            "epochs": self.epochs,
            "stopped_reason": self.stopped_reason,
        }


# ---------------------------------------------------------------------- #
# 主训练器
# ---------------------------------------------------------------------- #
class JointAlignmentTrainer:
    """论文 Algorithm 2 的联合对齐训练器。

    Args:
        cl_trainer: :class:`src.training.cl_trainer.CLTrainer`（已装配好数据与损失）。
        augmentor: :class:`src.llm.augmentor.Augmentor`；``None`` 表示禁用增强
            （退化为纯 CL 训练，等价于 M=0 且 w=0 的对照）。
        llm_backend: 支持微调的后端（``transformers``）；``None`` 或
            ``supports_finetuning=False`` 时自动跳过微调阶段。
        merger: :class:`src.llm.ties_merge.TiesMerger`。
        prompt_builder: Prompt 编排器（构造自举微调样本时复用）。
        processed_dir: 处理后的数据目录，用于落盘每轮增强结果。
        dataset: 数据集名（仅用于文件命名）。
        max_augment_rounds: ``w``，最多做几轮增强。
        max_finetune_rounds: ``M``，最多做几轮微调。
        augment_interval_epochs: ``T``，每几个 epoch 增强一次。
        finetune_interval_epochs: 每几个 epoch 完成一次"增强 + 微调"的完整对齐周期。
        momentum_beta: 式(8)的 β。
        lambda_source: λ 的口径，见 :func:`compute_lambda_score`。
        omega_min / omega_max: 式(9)的截断区间。
        lambda_init: ``λ_0``。
        output_dir: 增强数据、任务向量与状态文件的保存目录。
        stop_when_max_reached: 达到 M 后是否停止训练（论文 Algorithm 2 第 10-15 行）。
        log_every_epochs: 每几个 epoch 打印一次总结。
        logger: 可选 logger。
    """

    def __init__(
        self,
        cl_trainer: CLTrainer,
        augmentor: Optional[Any] = None,
        llm_backend: Optional[Any] = None,
        merger: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        processed_dir: str = "",
        dataset: str = "",
        max_augment_rounds: int = 3,
        max_finetune_rounds: int = 0,
        augment_interval_epochs: int = 1,
        finetune_interval_epochs: int = 5,
        momentum_beta: float = 0.9,
        lambda_source: str = "contrastive_accuracy",
        omega_min: float = 0.05,
        omega_max: float = 0.95,
        lambda_init: float = 1.0,
        output_dir: str = "",
        stop_when_max_reached: bool = False,
        reset_classifier_each_round: bool = False,
        log_every_epochs: int = 1,
        logger: Optional[Any] = None,
    ):
        self.cl_trainer = cl_trainer
        self.augmentor = augmentor
        self.llm_backend = llm_backend
        self.prompt_builder = prompt_builder
        self.processed_dir = processed_dir
        self.dataset = dataset or "dataset"
        self.max_augment_rounds = max(0, int(max_augment_rounds))
        self.max_finetune_rounds = max(0, int(max_finetune_rounds))
        self.augment_interval_epochs = max(1, int(augment_interval_epochs))
        self.finetune_interval_epochs = max(1, int(finetune_interval_epochs))
        self.momentum_beta = float(momentum_beta)
        self.lambda_source = lambda_source
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.output_dir = output_dir or ""
        self.stop_when_max_reached = bool(stop_when_max_reached)
        self.reset_classifier_each_round = bool(reset_classifier_each_round)
        self.log_every_epochs = max(1, int(log_every_epochs))
        self.logger = logger or get_logger("joint_trainer")

        from src.llm.ties_merge import TiesMerger

        self.merger = merger or TiesMerger(
            trim_percent=20.0, alpha=0.5, lambda_init=lambda_init
        )
        self.state = AlignmentState()
        self.state.lambda_current = float(lambda_init)
        self.state.lambda_previous = float(lambda_init)

        # 周期级（每 m 轮微调一次）冻结的 λ / ω。
        # 背景：论文式(8)(9) 出现在 Algorithm 2 的**周期**块里（每 T 个 epoch 一次），
        # 而本实现在每个 epoch 都做一次 CL 评测，因此 λ 是"每 epoch 的 EMA"。
        # 为了让式(7) 的 α 插值仍然是"本轮 vs 上一轮"而不是"本 epoch vs 上一 epoch"，
        # 这里额外维护一对周期级快照：合并时用快照，合并后把快照推进一格。
        self.lambda_cycle = float(lambda_init)
        self.lambda_cycle_previous = float(lambda_init)
        self.omega_cycle = compute_omega(float(lambda_init), self.omega_min, self.omega_max)

        # 运行时累积状态
        self.augment_round = 0
        self.finetune_round = 0
        self.augmented_pool: List[DataInstance] = []
        self.train_originals: List[DataInstance] = []
        self._augment_failed = False

    # ------------------------------------------------------------------ #
    # 数据准备
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        train_originals: Sequence[DataInstance],
        existing_augmented: Optional[Sequence[DataInstance]] = None,
    ) -> None:
        """登记训练集原样本与"磁盘上已有的增强样本"。

        Args:
            train_originals: 训练集原样本（增强与自举微调都以它为输入）。
            existing_augmented: 已经落盘、并被初始训练集使用的增强样本
                （例如先跑过 ``scripts/augment_data.py`` 再跑 ``joint_align.py``）。

        Note:
            必须把 ``existing_augmented`` 也放进 :attr:`augmented_pool`：
            第一轮内存增强结束后 :meth:`rebuild_train_loader` 会用"原样本 +
            augmented_pool"重建训练集，若池子里只有新增强样本，
            原先那些已经落盘的增强样本就会从训练集中**静默消失**，训练集反而变小。
        """
        self.train_originals = list(train_originals)
        if existing_augmented:
            known = {id(item) for item in self.augmented_pool}
            added = [item for item in existing_augmented if id(item) not in known]
            self.augmented_pool.extend(added)
            rounds = sorted({item.augment_round for item in added})
            self.logger.info(
                f"已把磁盘上现有的 {len(added)} 条增强样本纳入增强池"
                f"（轮次 {rounds}），避免重建训练集时丢失"
            )

    # ------------------------------------------------------------------ #
    # 回调
    # ------------------------------------------------------------------ #
    def on_epoch_end(self, epoch: int, trainer: CLTrainer) -> Optional[str]:
        """挂到 :meth:`CLTrainer.run` 的 ``on_epoch_end`` 回调。"""
        # 1) 取本轮验证指标（trainer.history 已在 run() 中写入）
        record = trainer.history.records[-1] if trainer.history.records else {}
        dev_metrics = {
            "acc": record.get("dev_acc") or 0.0,
            "avg_f1": record.get("dev_avg_f1") or 0.0,
            "loss": record.get("dev_loss") or 0.0,
        }
        score = compute_lambda_score(dev_metrics, source=self.lambda_source)
        self.update_lambda(score, epoch=epoch, dev_metrics=dev_metrics)

        # 2) 每 T 个 epoch 做一次数据增强
        augment_done = self.maybe_augment(epoch)

        # 3) 每 finetune_interval_epochs 个 epoch 做一次微调 + TIES 合并（含本轮增强）
        if self.max_finetune_rounds > 0 and epoch % self.finetune_interval_epochs == 0:
            self.finetune_and_merge(epoch, augment_done)

        if self.log_every_epochs and epoch % self.log_every_epochs == 0:
            self.logger.info(
                f"[对齐状态] epoch={epoch} λ={self.state.lambda_current:.4f} "
                f"ω={self.state.omega:.4f} 增强轮次={self.augment_round}/{self.max_augment_rounds} "
                f"微调轮次={self.finetune_round}/{self.max_finetune_rounds} "
                f"增强池={len(self.augmented_pool)}"
            )

        if self.state.stopped_reason:
            return "stop"
        return None

    # ------------------------------------------------------------------ #
    # 式(8)：动量更新 λ
    # ------------------------------------------------------------------ #
    def update_lambda(
        self,
        score: float,
        epoch: int = 0,
        dev_metrics: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """论文式(8)：``λ_m = β·f_m + (1-β)·λ_{m-1}``（每个 epoch 调用一次）。

        Note:
            这里更新的是**epoch 级**的 :attr:`state.lambda_current`（用于监控与日志）。
            真正参与 Algorithm 1 的 ω 取 :attr:`omega_cycle`（周期级快照），
            因为它必须与式(7) 的 α 插值同尺度——见 :meth:`finetune_and_merge`。
        """
        updated = self.merger.update_lambda(score, beta=self.momentum_beta)
        self.state.lambda_previous = self.merger.lambda_previous
        self.state.lambda_current = updated
        self.state.omega = compute_omega(updated, self.omega_min, self.omega_max)
        # 周期级快照跟着 epoch 级 λ 走：周期结束时它就是"该周期最后一刻的 λ"
        self.lambda_cycle = updated
        self.omega_cycle = compute_omega(updated, self.omega_min, self.omega_max)
        self.state.epochs.append(
            {
                "epoch": epoch,
                "score": round(float(score), 6),
                "lambda": round(updated, 6),
                "omega": round(self.state.omega, 6),
                "dev_metrics": {key: round(float(value), 6) for key, value in (dev_metrics or {}).items()},
            }
        )
        self.logger.info(
            f"式(8) 动量更新：f={score:.4f} → λ={updated:.4f}"
            f"（β={self.momentum_beta}）→ ω={self.state.omega:.4f}（式(9)）"
        )
        return updated

    # ------------------------------------------------------------------ #
    # 数据增强
    # ------------------------------------------------------------------ #
    def maybe_augment(self, epoch: int) -> bool:
        """按 ``T`` 的节奏执行一轮数据增强（论文 Algorithm 2 第 8-9 行）。"""
        if self.augmentor is None or not self.train_originals:
            return False
        if self.augment_round >= self.max_augment_rounds:
            return False
        if epoch % self.augment_interval_epochs != 0:
            return False
        return self.run_augmentation(epoch)

    def run_augmentation(self, epoch: int = 0) -> bool:
        """执行一轮增强：调用 LLM → 落盘 → 累积进增强池 → 重建训练集。"""
        self.augment_round += 1
        round_index = self.augment_round
        self.logger.info(
            f"===== 数据增强第 {round_index}/{self.max_augment_rounds} 轮"
            f"（epoch {epoch}，式(8) 当前 λ={self.state.lambda_current:.4f}）====="
        )
        started = time.time()
        augmented, stats = self.augmentor.augment(
            self.train_originals, augment_round=round_index
        )
        summary = stats.summary()
        self.augmented_pool.extend(augmented)

        records: Dict[str, Any] = {
            "round": round_index,
            "epoch": epoch,
            "lambda": round(self.state.lambda_current, 6),
            "omega": round(self.state.omega, 6),
            "stats": summary,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        self.state.augment_rounds.append(records)

        if self.output_dir:
            ensure_dir(self.output_dir)
            path = os.path.join(
                self.output_dir, f"augmented_round{round_index}.jsonl"
            )
            write_jsonl(path, (item.to_record() for item in augmented))
            records["output"] = path
            self.logger.info(f"第 {round_index} 轮增强结果已保存：{path}")

        self.rebuild_train_loader()
        self._augment_failed = summary.get("succeeded", 0) == 0
        if self._augment_failed:
            self.logger.warning(
                f"第 {round_index} 轮增强全部失败（成功率 {summary.get('success_rate')}），"
                "本轮不参与后续微调"
            )
        return not self._augment_failed

    def rebuild_train_loader(self) -> None:
        """用"原样本 + 累计增强样本"重建训练 DataLoader。

        Note:
            重建而不是原地追加，是因为 :class:`data.dataset.PairDataset` 的
            配对关系在构造时就固定了；重建可以保证 ``augmented_round=0``
            （使用全部历史轮次）时每条原样本都能看到它所有轮次的增强样本。
        """
        from data.dataset import PairDataset, build_dataloader

        current_loader = self.cl_trainer.train_loader
        current_dataset = current_loader.dataset
        originals = [item[0] for item in getattr(current_dataset, "samples", [])]
        if not originals:
            self.logger.warning("训练集为空，跳过重建")
            return

        rebuilt = PairDataset(
            originals=originals,
            augmented=self.augmented_pool,
            tokenizer=getattr(current_dataset, "tokenizer", None),
            label_list=getattr(current_dataset, "label_list", None),
            max_seq_length=getattr(current_dataset, "max_seq_length", 128),
            augmented_round=0,          # 0 = 使用全部历史轮次的增强样本
            per_sample=getattr(current_dataset, "per_sample", 1),
            require_augmented=False,
            text_mode=getattr(current_dataset, "text_mode", "source_replies"),
        )
        self.cl_trainer.train_loader = build_dataloader(
            rebuilt,
            batch_size=getattr(current_loader, "batch_size", 32),
            shuffle=True,
            paired=True,
            num_workers=int(getattr(current_loader, "num_workers", 0) or 0),
        )
        paired_count = sum(1 for _, group in rebuilt.samples if group)
        self.logger.info(
            f"训练集已重建：{len(rebuilt)} 条样本，其中带增强样本的 {paired_count} 条"
        )

    # ------------------------------------------------------------------ #
    # 微调 + 合并
    # ------------------------------------------------------------------ #
    def finetune_and_merge(self, epoch: int, augment_done: bool) -> bool:
        """完成一次"自举微调 → 导出任务向量 → TIES 合并 → 写回"的完整周期。"""
        if self.finetune_round >= self.max_finetune_rounds:
            if self.stop_when_max_reached:
                self.state.stopped_reason = (
                    f"已达最大微调轮次 M={self.max_finetune_rounds}（论文 Algorithm 2 第 10-15 行）"
                )
                self.logger.info(f"停止条件满足：{self.state.stopped_reason}")
            return False

        if not augment_done:
            self.logger.info(
                f"epoch {epoch} 到达微调周期，但本轮没有新的增强数据，"
                "本轮只做 TIES 合并（沿用已有任务向量）"
            )

        self.finetune_round += 1
        round_index = self.finetune_round

        # 冻结本周期的 λ / ω：式(7) 的 α 插值必须用"本轮 vs 上一轮"的值，
        # 而不是"本 epoch vs 上一 epoch"的值（epoch 级 λ 变化太快，
        # 会让 (1-α)λ_m + αλ_{m-1} 中的历史项失去"上一次合并时的强度"这一含义）。
        lambda_cycle, omega_cycle = self.lambda_cycle, self.omega_cycle

        self.logger.info(
            f"===== LLM 微调第 {round_index}/{self.max_finetune_rounds} 轮"
            f"（epoch {epoch}，周期 λ={lambda_cycle:.4f}，ω={omega_cycle:.4f}）====="
        )

        # ---- 1) 自举微调 ----
        finetune_info: Dict[str, Any] = {
            "round": round_index,
            "epoch": epoch,
            "omega": round(omega_cycle, 6),
            "lambda": round(lambda_cycle, 6),
            "num_records": 0,
            "adapter_dir": None,
        }
        vector = None
        if self.llm_backend is not None and getattr(self.llm_backend, "supports_finetuning", False):
            from src.llm.lora import build_finetune_records, finetune_and_export

            if self.prompt_builder is None:
                raise ValueError(
                    "启用 LLM 微调时必须提供 prompt_builder：自举微调样本要求"
                    "Prompt 与数据增强阶段完全一致（见 src/llm/lora.py 的说明）"
                )
            records = build_finetune_records(
                originals=self.train_originals,
                augmented=[item for item in self.augmented_pool if item.augment_round == self.augment_round],
                prompt_builder=self.prompt_builder,
                target="original",
            )
            finetune_info["num_records"] = len(records)
            if records and self.output_dir:
                from src.llm.lora import dump_records

                dump_records(
                    os.path.join(self.output_dir, f"finetune_round{round_index}_records.jsonl"),
                    records,
                )
            adapter_dir = os.path.join(self.output_dir, f"lora_round{round_index}") if self.output_dir else ""
            result = finetune_and_export(
                backend=self.llm_backend,
                records=records,
                output_dir=adapter_dir,
                reset_to_base=True,
                logger=self.logger,
            )
            vector = result.get("vector")
            finetune_info["adapter_dir"] = result.get("path")
            finetune_info["vector_stats"] = result.get("stats", {})
            finetune_info["reset_to_base"] = result.get("reset_to_base", False)
        else:
            self.logger.warning(
                "当前 LLM 后端不支持梯度微调，跳过式(9) 的微调环节；"
                "TIES 合并仍会执行（若已有任务向量）"
            )

        # ---- 2) TIES 合并（论文 Algorithm 1）----
        if vector is not None:
            self.merger.add(vector, weight=omega_cycle, round_index=round_index)
        merge_info = self.merge_and_apply(round_index, epoch)

        # 周期推进：本轮的周期 λ 成为下一次合并的 λ_previous（式(7) 的历史项）
        self.lambda_cycle_previous = lambda_cycle
        self.merger.lambda_previous = lambda_cycle

        finetune_info["merge"] = merge_info
        self.state.finetune_rounds.append(finetune_info)

        if self.reset_classifier_each_round:
            self.reset_classifier()

        # 微调改变了 LLM，增强质量分布随之变化 → 重建训练集并重置优化器
        # （只在这里重置：单轮纯增强不重置，否则每 T 个 epoch 就重新 warmup，
        #   `training.cl.augment_every_epochs=1` 时等于每个 epoch 都重启学习率）
        self.rebuild_train_loader()
        self.cl_trainer.optimizer = None
        self.cl_trainer.scheduler = None

        if self.finetune_round >= self.max_finetune_rounds and self.stop_when_max_reached:
            self.state.stopped_reason = (
                f"已完成 M={self.max_finetune_rounds} 轮微调与合并"
            )
        return True

    def merge_and_apply(self, round_index: int, epoch: int) -> Dict[str, Any]:
        """执行 Algorithm 1 并把合并结果写回 LLM。

        缩放系数用**周期级**的 λ（:attr:`lambda_cycle` 与
        :attr:`lambda_cycle_previous`），使式(7) 的
        ``scaling = (1-α)·λ_m + α·λ_{m-1}`` 中的 ``λ_{m-1}`` 确实是
        "上一次合并时的强度"，而不是上一个 epoch 的值。
        """
        info: Dict[str, Any] = {"round": round_index, "epoch": epoch, "applied": False}
        if self.merger.task_vector.num_vectors == 0:
            self.logger.info("尚无任务向量，跳过 TIES 合并")
            return info
        if self.llm_backend is None or not getattr(self.llm_backend, "supports_task_vector", False):
            self.logger.warning("当前后端不支持写回任务向量，跳过合并")
            return info

        # 用周期级 λ 覆盖 merger 里的 epoch 级值，再执行合并
        saved_current = self.merger.lambda_current
        saved_previous = self.merger.lambda_previous
        self.merger.lambda_current = self.lambda_cycle
        self.merger.lambda_previous = self.lambda_cycle_previous
        try:
            merged, report, scaling = self.merger.merge()
        finally:
            # 合并完成后把 merger 的 λ 恢复为 epoch 级，继续给下一轮的式(8) 用
            self.merger.lambda_current = saved_current
            self.merger.lambda_previous = saved_previous

        self.llm_backend.apply_task_vector(merged, scaling=scaling)
        info.update(
            {
                "applied": True,
                "scaling": round(float(scaling), 6),
                "report": report.to_dict(),
                "weights": [round(value, 6) for value in self.merger.task_vector.weights],
                "lambda_cycle": round(self.lambda_cycle, 6),
                "lambda_cycle_previous": round(self.lambda_cycle_previous, 6),
            }
        )
        self.state.merges.append(info)
        self.logger.info(
            f"Algorithm 1 完成：合并 {report.num_vectors} 个任务向量，"
            f"修剪后稀疏度 {report.sparsity_after_trim:.4f}，"
            f"写入比例 {report.merge_rate:.4f}，写回缩放 scaling={scaling:.4f}"
        )
        self.logger.info(f"式(7) 与式(8) 的对应关系：{self.merger.report_note()}")
        return info

    @staticmethod
    def _add_vectors(base: Optional[Mapping[str, Any]], delta: Mapping[str, Any]) -> Dict[str, Any]:
        """累加任务向量（逐元素）。

        保留这个工具是为了让"累计已写回的增量"这类诊断/消融可复用；
        主流程**不再**用它做任务向量的基准对齐——那件事由
        :func:`src.llm.lora.finetune_and_export` 的 ``reset_to_base`` 保证
        （见该函数 docstring 里对"为什么不能用原始 τ 之和做差分"的说明）。
        """
        if not base:
            return dict(delta)
        result: Dict[str, Any] = {}
        for name, value in delta.items():
            result[name] = value + base[name] if name in base else value
        return result

    def reset_classifier(self) -> None:
        """重置分类头（数据分布大幅变化时的可选操作）。"""
        import torch

        model = self.cl_trainer.model
        raw = model.module if hasattr(model, "module") else model
        classifier = getattr(raw, "classifier", None)
        if classifier is None:
            return
        for module in classifier.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
        self.logger.info("已重置 CL 分类头参数（配置项 reset_classifier_each_round=true）")

    # ------------------------------------------------------------------ #
    # 训练入口
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_originals: Sequence[DataInstance],
        existing_augmented: Optional[Sequence[DataInstance]] = None,
    ) -> AlignmentState:
        """执行完整的 Algorithm 2 流程并返回状态。

        Args:
            train_originals: 训练集原样本。
            existing_augmented: 初始训练集里已经在用的磁盘增强样本（见 :meth:`prepare`）。
        """
        self.prepare(train_originals, existing_augmented=existing_augmented)
        self.logger.info(
            f"联合对齐开始：epochs={self.cl_trainer.epochs} "
            f"T(增强间隔)={self.augment_interval_epochs} "
            f"微调周期={self.finetune_interval_epochs} "
            f"w={self.max_augment_rounds} M={self.max_finetune_rounds} "
            f"β={self.momentum_beta} λ口径={self.lambda_source}"
        )
        self.cl_trainer.run(on_epoch_end=self.on_epoch_end)

        if self.output_dir:
            ensure_dir(self.output_dir)
            save_json(os.path.join(self.output_dir, "alignment_state.json"), self.state.to_dict())
            self.logger.info(
                f"对齐状态已保存：{os.path.join(self.output_dir, 'alignment_state.json')}"
            )
        if self.llm_backend is not None and getattr(self.llm_backend, "supports_task_vector", False):
            try:
                self.llm_backend.close()
            except Exception as exc:  # pragma: no cover - 释放显存失败不影响结果
                self.logger.warning(f"释放 LLM 显存时出错（已忽略）：{exc}")
        return self.state


# ---------------------------------------------------------------------- #
# 辅助
# ---------------------------------------------------------------------- #
def snapshot(obj: Any) -> Any:
    """对配置类对象做深拷贝（供调用方保存"训练前"的配置快照）。"""
    import copy

    return copy.deepcopy(obj)

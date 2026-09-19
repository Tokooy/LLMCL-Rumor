# coding=utf-8
"""对比学习训练器：一个增强轮次内的 CL 网络训练（论文 §3.2）。

职责边界
--------
本模块只负责"给定固定数据集，把 CL 分类网络训练好"：

* 前向 → 联合损失（交叉熵 + InfoNCE）→ 反向 → 更新；
* 每个 epoch 结束在验证集上评测、按 ``avg_f1`` 或 ``acc`` 保存最优权重；
* 支持早停（对应配置 ``training.cl.early_stop_patience``）。

**不含** LLM 增强与微调逻辑——那属于 :mod:`src.training.joint_trainer`
（论文 Algorithm 2）。两者通过 :meth:`CLTrainer.run` 的 ``on_epoch_end``
回调衔接：联合训练器传入的回调里做"增强 + 微调 + TIES 合并"。

与论文的对应
------------
* "在每个 batch 中 (x_i, x'_i) 配对为正样本" → ``anchor`` / ``augmented`` 两组张量
  分别过编码器，投影后送入 :class:`src.training.losses.ContrastiveLoss`；
* "参与损失计算的样本数量为 2B" → 对比损失在每个 batch 内看到 2B 个投影向量
  （B 个锚点 + B 个正样本），负样本来自同一 batch；
* "训练完成后 CL 模型参数被冻结" → :meth:`CLTrainer.freeze_after_training`。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

from .losses import CombinedLoss

__all__ = [
    "compute_loss",
    "build_optimizer",
    "CLTrainer",
    "TrainingHistory",
]

try:  # pragma: no cover - 环境探测
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------- #
# 前向 + 损失
# ---------------------------------------------------------------------- #
def _forward_batch(model: Any, batch: Mapping[str, Any], paired: bool = True) -> Dict[str, Any]:
    """跑一次前向，返回 logits / 投影 / 隐状态。

    Args:
        model: :class:`src.models.cl_model.ContrastiveModel`。
        batch: ``collate_pairs`` 或 ``collate_single`` 的输出。
        paired: 是否包含增强样本（决定是否额外跑一遍编码器）。
    """
    original = batch["original"]
    outputs = model(
        input_ids=original["input_ids"],
        attention_mask=original["attention_mask"],
        token_type_ids=original.get("token_type_ids"),
    )

    augmented_projection = None
    if paired:
        augmented = batch.get("augmented")
        if augmented is not None:
            augmented_projection = _encode_augmented(model, augmented)

    outputs["augmented_projection"] = augmented_projection
    return outputs


def _encode_augmented(model: Any, augmented: Any):
    """把增强样本编码成 ``[B, K, d]`` 的投影张量。

    ``collate_pairs`` 有两种输出形态：

    * 份数一致 → ``{"input_ids": [B, K, L], ...}``，直接 reshape 后一次前向算完；
    * 份数不一致 → ``[{"input_ids": [B_k, L], ...}, ...]``，按份数逐个前向。
    """
    if torch is None:  # pragma: no cover
        raise ImportError("需要 torch")

    if isinstance(augmented, Mapping):
        input_ids = augmented["input_ids"]
        batch_size, copies, length = input_ids.shape
        flat = {
            "input_ids": input_ids.reshape(batch_size * copies, length),
            "attention_mask": augmented["attention_mask"].reshape(batch_size * copies, length),
            "token_type_ids": augmented.get("token_type_ids"),
        }
        if flat["token_type_ids"] is not None:
            flat["token_type_ids"] = flat["token_type_ids"].reshape(batch_size * copies, length)
        outputs = model(
            input_ids=flat["input_ids"],
            attention_mask=flat["attention_mask"],
            token_type_ids=flat["token_type_ids"],
        )
        return outputs["projection"].reshape(batch_size, copies, -1)

    # 份数不一致：逐份编码，份数不足的样本用自身投影占位（等价于"无增强"）
    projections = []
    for layer in augmented:
        outputs = model(
            input_ids=layer["input_ids"],
            attention_mask=layer["attention_mask"],
            token_type_ids=layer.get("token_type_ids"),
        )
        projections.append(outputs["projection"])
    return projections


def compute_loss(
    criterion: CombinedLoss,
    outputs: Mapping[str, Any],
    labels: Any,
    group_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """把模型输出与损失函数接起来，返回损失字典。

    两种 batch 形态分别处理：

    * **份数一致**（``augmented_projection`` 是 ``[B, K, d]`` 张量）：
      直接交给 :class:`CombinedLoss`，一次算完全 batch 的交叉熵与 InfoNCE；
    * **份数不一致**（是长度为 K 的列表，第 k 层只含"至少有 k 份增强样本"的样本）：
      交叉熵**仍然在全 batch 上算一次**（分类损失与增强份数无关，按层重复计算
      会让 CE 被重复计入、日志口径也会失真）；InfoNCE 则逐层计算后按层内样本数
      加权平均——每层能看到的负样本集合不同，只能在层内比较。
    """
    augmented = outputs.get("augmented_projection")

    if isinstance(augmented, list):
        # 全 batch 的分类损失：只算一次，作为最终 CE 项
        ce_loss = criterion.cross_entropy(outputs["logits"], labels)

        losses: List[Any] = []
        weights: List[int] = []
        cl_values: List[float] = []
        for layer in augmented:
            count = layer.shape[0]
            if count == 0:
                continue
            cl_loss = criterion.contrastive(
                outputs["projection"][:count],
                layer.unsqueeze(1),
                labels=labels[:count],
                group_ids=list(group_ids)[:count] if group_ids else None,
            )
            losses.append(cl_loss)
            weights.append(count)
            cl_values.append(float(cl_loss.detach().cpu()))

        total = criterion.ce_weight * ce_loss
        cl_loss_value = 0.0
        if losses:
            total_weight = float(sum(weights))
            weighted_cl = sum(
                loss * (weight / total_weight) for loss, weight in zip(losses, weights)
            )
            total = total + criterion.cl_weight * weighted_cl
            # 报告口径：按层内样本数加权的 InfoNCE 均值
            cl_loss_value = sum(
                value * (weight / total_weight) for value, weight in zip(cl_values, weights)
            )

        criterion.last_components = {
            "ce_loss": float(ce_loss.detach().cpu()),
            "cl_loss": cl_loss_value,
            "total_loss": float(total.detach().cpu()),
        }
        return {
            "loss": total,
            "ce_loss": float(ce_loss.detach().cpu()),
            "cl_loss": cl_loss_value,
        }

    return criterion(
        outputs["logits"],
        labels,
        anchor_embeddings=outputs["projection"],
        augmented_embeddings=augmented,
        group_ids=group_ids,
    )


# ---------------------------------------------------------------------- #
# 优化器
# ---------------------------------------------------------------------- #
def build_optimizer(
    model: Any,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    layer_lr_decay: float = 1.0,
    num_encoder_layers: int = 12,
) -> Any:
    """构造 AdamW 优化器，按 BERT 惯例分成两组参数。

    * 所有 ``bias`` 与 ``LayerNorm`` 参数不做权重衰减（BERT 微调的标准做法，
      也是原开源项目 ``main.py`` 里的 ``no_decay`` 列表）；
    * ``layer_lr_decay < 1`` 时对底层做学习率衰减（``llrd``），默认 1.0 即不分层，
      与论文"所有实验的关键超参数保持一致"的表述相符。

    Args:
        model: :class:`ContrastiveModel`。
        learning_rate: 顶层学习率。
        weight_decay: 权重衰减系数。
        adam_epsilon: Adam 的 eps。
        layer_lr_decay: 层间学习率衰减系数。
        num_encoder_layers: 编码器层数，用于计算分层学习率。
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("需要 torch")

    named_parameters: List[Tuple[str, Any]] = []
    for name, param in model.projector.named_parameters():
        named_parameters.append((f"projector.{name}", param))
    for name, param in model.classifier.named_parameters():
        named_parameters.append((f"classifier.{name}", param))
    for name, param in model.named_encoder_parameters(prefix="encoder.model"):
        named_parameters.append((name, param))

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln_")
    grouped: Dict[Tuple[float, float], List[Any]] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        decay = 0.0 if any(token in name for token in no_decay) else weight_decay
        lr = learning_rate
        if layer_lr_decay < 1.0 and name.startswith("encoder.model.encoder.layer."):
            try:
                index = int(name.split("encoder.layer.")[1].split(".")[0])
            except (IndexError, ValueError):
                index = num_encoder_layers
            lr = learning_rate * (layer_lr_decay ** (num_encoder_layers - index))
        elif layer_lr_decay < 1.0 and name.startswith("encoder.model.embeddings"):
            lr = learning_rate * (layer_lr_decay ** (num_encoder_layers + 1))
        grouped.setdefault((decay, lr), []).append(param)

    parameter_groups = [
        {"params": params, "weight_decay": decay, "lr": lr}
        for (decay, lr), params in grouped.items()
    ]
    if not parameter_groups:  # pragma: no cover - 全冻结时的兜底
        raise ValueError("没有任何可训练参数；请检查 freeze_layers 与 freeze_encoder 设置")

    return torch.optim.AdamW(parameter_groups, eps=adam_epsilon)


def build_scheduler(
    optimizer: Any,
    num_training_steps: int,
    warmup_proportion: float = 0.1,
) -> Any:
    """线性 warmup + 线性衰减（与原开源项目用 BertAdam 的效果一致）。"""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("需要 torch")

    warmup_steps = int(max(0.0, min(0.9, warmup_proportion)) * max(1, num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining = max(0, num_training_steps - current_step)
        total = max(1, num_training_steps - warmup_steps)
        return float(remaining) / float(total)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------- #
# 训练历史
# ---------------------------------------------------------------------- #
class TrainingHistory:
    """逐 epoch 的训练/验证指标记录。"""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))

    @property
    def best(self) -> Optional[Dict[str, Any]]:
        return max(self.records, key=lambda item: item.get("dev_avg_f1", 0.0)) if self.records else None

    def to_dict(self) -> Dict[str, Any]:
        return {"epochs": self.records, "best": self.best}


# ---------------------------------------------------------------------- #
# 训练器
# ---------------------------------------------------------------------- #
class CLTrainer:
    """单轮次内的对比学习训练器。

    Args:
        model: :class:`ContrastiveModel`。
        train_loader / dev_loader: DataLoader（``paired=True``）。
        criterion: :class:`CombinedLoss`。
        device: ``torch.device``。
        epochs: 训练轮数。
        learning_rate / weight_decay / adam_epsilon: 优化器参数。
        warmup_proportion: warmup 比例。
        gradient_accumulation_steps: 梯度累积步数。
        max_grad_norm: 梯度裁剪。
        evaluate_every_epochs: 每多少个 epoch 评测一次。
        save_best_on: ``acc`` 或 ``avg_f1``。
        early_stop_patience: 连续多少个 epoch 没提升就停（0 = 不早停）。
        output_dir: checkpoint 保存目录。
        logger: 可选 logger。
    """

    def __init__(
        self,
        model: Any,
        train_loader: Any,
        dev_loader: Optional[Any],
        criterion: CombinedLoss,
        device: Any,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        warmup_proportion: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        evaluate_every_epochs: int = 1,
        save_best_on: str = "avg_f1",
        early_stop_patience: int = 10,
        output_dir: str = "",
        log_every_steps: int = 50,
        logger: Optional[Any] = None,
    ):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("使用 CLTrainer 需要先安装 torch")
        if save_best_on not in ("acc", "avg_f1"):
            raise ValueError(f"save_best_on 只支持 acc / avg_f1，收到 {save_best_on!r}")

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.criterion = criterion
        self.device = device
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adam_epsilon = float(adam_epsilon)
        self.warmup_proportion = float(warmup_proportion)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.max_grad_norm = float(max_grad_norm)
        self.evaluate_every_epochs = max(1, int(evaluate_every_epochs))
        self.save_best_on = save_best_on
        self.early_stop_patience = max(0, int(early_stop_patience))
        self.output_dir = output_dir or ""
        self.log_every_steps = max(1, int(log_every_steps))
        self.logger = logger or get_logger("cl_trainer")

        self.optimizer = None
        self.scheduler = None
        self.history = TrainingHistory()
        self.best_metric = -math.inf
        self.best_epoch = -1
        self._epochs_without_improvement = 0

    # ------------------------------------------------------------------ #
    def _ensure_optimizer(self) -> None:
        if self.optimizer is not None:
            return
        self.optimizer = build_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_epsilon=self.adam_epsilon,
        )
        steps_per_epoch = max(
            1, len(self.train_loader) // self.gradient_accumulation_steps
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            num_training_steps=steps_per_epoch * self.epochs,
            warmup_proportion=self.warmup_proportion,
        )

    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch，返回平均损失与训练集准确率。"""
        from .evaluate import classification_metrics

        self.model.train()
        self._ensure_optimizer()

        total_loss = 0.0
        total_ce = 0.0
        total_cl = 0.0
        steps = 0
        predictions: List[Any] = []
        references: List[Any] = []

        for step, batch in enumerate(self.train_loader):
            batch = _move_batch(batch, self.device)
            labels = batch["label"]
            group_ids = list(batch.get("uid") or range(labels.shape[0]))
            outputs = _forward_batch(self.model, batch, paired=True)
            components = compute_loss(self.criterion, outputs, labels, group_ids=group_ids)
            loss = components["loss"] / self.gradient_accumulation_steps
            loss.backward()

            total_loss += float(components["loss"].detach().cpu())
            total_ce += float(components["ce_loss"])
            total_cl += float(components["cl_loss"])
            steps += 1

            with torch.no_grad():
                predictions.append(outputs["logits"].argmax(dim=-1).detach().cpu())
                references.append(labels.detach().cpu())

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if (step + 1) % self.log_every_steps == 0:
                self.logger.info(
                    f"epoch {epoch} step {step + 1}/{len(self.train_loader)} "
                    f"loss={total_loss / steps:.4f} ce={total_ce / steps:.4f} "
                    f"cl={total_cl / steps:.4f}"
                )

        # 收尾：处理最后不足一个累积周期的梯度
        if len(self.train_loader) % self.gradient_accumulation_steps != 0 and steps:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        train_metrics = classification_metrics(predictions, references)
        return {
            "loss": total_loss / max(1, steps),
            "ce_loss": total_ce / max(1, steps),
            "cl_loss": total_cl / max(1, steps),
            "train_acc": train_metrics["acc"],
            "train_avg_f1": train_metrics["avg_f1"],
        }

    # ------------------------------------------------------------------ #
    def run(
        self,
        on_epoch_end: Optional[Callable[[int, "CLTrainer"], Optional[Any]]] = None,
        reset_optimizer: bool = False,
    ) -> TrainingHistory:
        """跑完全部 epoch。

        Args:
            on_epoch_end: ``(epoch_index, trainer) -> Any`` 回调。联合训练器用它
                在每个 epoch 结束后执行"数据增强 / LLM 微调 / TIES 合并"。
                回调返回 ``"stop"`` 表示提前结束训练。
            reset_optimizer: 为 True 时重建优化器与调度器（数据分布变化后重新
                warmup，论文 Algorithm 2 在增强后重新进入 CL 训练时适用）。

        Returns:
            :class:`TrainingHistory`。
        """
        if reset_optimizer:
            self.optimizer = None
            self.scheduler = None

        started = time.time()
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)

            dev_metrics: Dict[str, Any] = {}
            if self.dev_loader is not None and epoch % self.evaluate_every_epochs == 0:
                from .evaluate import evaluate_model

                dev_metrics = evaluate_model(
                    self.model, self.dev_loader, self.device, criterion=self.criterion
                )

            metric = float(
                dev_metrics.get("acc" if self.save_best_on == "acc" else "avg_f1", -math.inf)
            )
            improved = metric > self.best_metric
            if improved:
                self.best_metric = metric
                self.best_epoch = epoch
                self._epochs_without_improvement = 0
                if self.output_dir:
                    self.save_checkpoint(os.path.join(self.output_dir, "best.pt"), epoch)
            else:
                self._epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "dev_acc": dev_metrics.get("acc"),
                "dev_avg_f1": dev_metrics.get("avg_f1"),
                "dev_loss": dev_metrics.get("loss"),
                "best_so_far": improved,
                **train_metrics,
            }
            self.history.add(**record)
            self.logger.info(
                f"epoch {epoch}/{self.epochs} 完成："
                f"loss={train_metrics['loss']:.4f} "
                f"dev_acc={dev_metrics.get('acc', float('nan')):.4f} "
                f"dev_avg_f1={dev_metrics.get('avg_f1', float('nan')):.4f}"
                + ("  [最优]" if improved else "")
            )

            if on_epoch_end is not None:
                signal = on_epoch_end(epoch, self)
                if signal == "stop":
                    self.logger.info(f"回调请求提前结束训练（epoch {epoch}）")
                    break

            if (
                self.early_stop_patience
                and self._epochs_without_improvement >= self.early_stop_patience
            ):
                self.logger.info(
                    f"连续 {self._epochs_without_improvement} 个 epoch 未提升，触发早停"
                )
                break

        self.logger.info(
            f"训练结束：最优 epoch={self.best_epoch}，"
            f"{self.save_best_on}={self.best_metric:.4f}，"
            f"耗时 {time.time() - started:.1f}s"
        )
        return self.history

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str, epoch: int = -1, extra: Optional[Mapping[str, Any]] = None) -> str:
        """保存 checkpoint（模型 + 优化器 + 训练历史）。"""
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "save_best_on": self.save_best_on,
            "history": self.history.to_dict(),
        }
        if extra:
            payload.update(dict(extra))
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """加载 checkpoint。"""
        payload = torch.load(path, map_location=self.device)
        state_dict = payload.get("model", payload)
        self.model.load_state_dict(state_dict, strict=False)
        if load_optimizer and payload.get("optimizer") and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.best_metric = float(payload.get("best_metric", -math.inf))
        return payload

    # ------------------------------------------------------------------ #
    def freeze_after_training(self) -> None:
        """训练结束后冻结 CL 网络（论文测试阶段的行为）。"""
        self.model.set_encoder_grad(False)
        self.model.eval()
        self.logger.info("已冻结 CL 网络参数（论文测试阶段设置）")


# ---------------------------------------------------------------------- #
# 工具
# ---------------------------------------------------------------------- #
def _move_batch(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    """把 batch 里的张量搬到设备上（保持嵌套结构）。"""
    if torch is None:  # pragma: no cover
        return dict(batch)

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    return {key: move(value) for key, value in batch.items()}

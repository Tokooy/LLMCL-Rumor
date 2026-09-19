# coding=utf-8
"""评估模块：指标计算与模型评测（论文 Table 3–8 的口径）。

论文用的评价标准
----------------
* **ACC**：准确率；
* **逐类 F1**：TR / NR / FR / UR 各自的 F1；
* **Avg F1**：论文表格里的 "Avg F1" 列，实现为四类 F1 的**宏平均（macro）**。
  对照 Table 3 的数值关系可以确认：Proposed-1 的四个 F1 为
  ``90.68 / 69.42 / 75.73 / 74.12``，其均值 ``77.4875`` 与表中 ``77.48`` 一致，
  因此论文的 Avg F1 就是四类 F1 的算术平均，而非加权平均。
* 精确率/召回率：论文文字提到"精确度和召回率强调正确分类的实例的比例"，
  本模块按类别输出，写入评估报告。

指标实现基于 ``scikit-learn``，与数据划分、标签顺序解耦。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data.processors.data_model import ID_TO_LABEL, LABELS
from src.utils.io_utils import ensure_dir, save_json, write_jsonl
from src.utils.logger import get_logger

__all__ = [
    "classification_metrics",
    "classification_report",
    "evaluate_model",
    "extract_features",
    "format_report",
    "save_predictions",
    "save_report",
]

try:  # pragma: no cover - 环境探测
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------- #
# 指标
# ---------------------------------------------------------------------- #
def _to_numpy_list(values: Any) -> List[Any]:
    """把"张量列表 / 数组 / 单个张量"统一成一维 Python 列表。"""
    if values is None:
        return []
    if _TORCH_AVAILABLE and isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        flattened: List[Any] = []
        for item in values:
            flattened.extend(_to_numpy_list(item))
        return flattened
    if hasattr(values, "reshape"):  # numpy 数组
        return list(values.reshape(-1).tolist())
    return list(values)


def classification_metrics(
    predictions: Any,
    references: Any,
    label_list: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """计算准确率、逐类 F1、宏/加权 F1、精确率与召回率。

    Args:
        predictions: 预测标签（下标、标签名或它们的列表/张量）。
        references: 真实标签，形态同上。
        label_list: 标签顺序，默认 ``["NR", "FR", "TR", "UR"]``。

    Returns:
        含 ``acc`` / ``avg_f1`` / ``macro_f1`` / ``weighted_f1`` / ``per_class`` /
        ``report`` 的字典。空输入返回全 0，避免上层除零。
    """
    from sklearn import metrics as sk_metrics

    labels = list(label_list or LABELS)
    label_ids = list(range(len(labels)))

    preds = _to_numpy_list(predictions)
    refs = _to_numpy_list(references)
    if len(preds) != len(refs):
        raise ValueError(f"预测与标签数量不一致：{len(preds)} vs {len(refs)}")
    if not preds:
        return {
            "acc": 0.0,
            "avg_f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "per_class": {label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                          for label in labels},
            "report": {},
            "num_samples": 0,
        }

    report = sk_metrics.classification_report(
        refs,
        preds,
        labels=label_ids,
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    # 论文的 Avg F1 = 四类 F1 的算术平均（已验证与 Table 3 数值一致）
    avg_f1 = sum(per_class[label]["f1"] for label in labels) / len(labels)

    return {
        "acc": float(sk_metrics.accuracy_score(refs, preds)),
        "avg_f1": float(avg_f1),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "per_class": per_class,
        "report": report,
        "num_samples": len(refs),
    }


def classification_report(
    predictions: Any,
    references: Any,
    label_list: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """返回 sklearn 风格的完整报告字典（评测脚本保存用）。"""
    return classification_metrics(predictions, references, label_list)["report"]


# ---------------------------------------------------------------------- #
# 评测
# ---------------------------------------------------------------------- #
def evaluate_model(
    model: Any,
    dataloader: Any,
    device: Any,
    criterion: Optional[Any] = None,
    label_list: Optional[Sequence[str]] = None,
    max_batches: int = 0,
) -> Dict[str, Any]:
    """在给定数据集上评测模型，返回指标字典。

    Args:
        model: :class:`ContrastiveModel`。
        dataloader: ``paired=False`` 或 ``True`` 均可（评测只用原样本）。
        device: 设备。
        criterion: 可选的损失函数（用于汇报验证损失）。
        max_batches: ``>0`` 时只跑前 N 个 batch（调试用）。

    Returns:
        含 ``loss`` 与 :func:`classification_metrics` 全部键的字典。
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("evaluate_model 需要 torch")

    model.eval()
    all_predictions: List[Any] = []
    all_references: List[Any] = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if max_batches and index >= max_batches:
                break
            inputs = batch.get("original", batch)
            labels = batch["label"].to(device)
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                token_type_ids=(
                    inputs["token_type_ids"].to(device)
                    if inputs.get("token_type_ids") is not None
                    else None
                ),
            )
            logits = outputs["logits"]
            all_predictions.append(logits.argmax(dim=-1).cpu())
            all_references.append(labels.cpu())
            if criterion is not None:
                total_loss += float(criterion(logits, labels)["loss"].detach().cpu())
                steps += 1

    metrics = classification_metrics(all_predictions, all_references, label_list)
    metrics["loss"] = (total_loss / steps) if steps else 0.0
    return metrics


def extract_features(
    model: Any,
    dataloader: Any,
    device: Any,
    use_projection: bool = True,
    max_samples: int = 0,
) -> Tuple[Any, Any, List[str]]:
    """抽取特征与标签，供 t-SNE 可视化（论文 Fg.7–Fg.10）。

    Args:
        model: :class:`ContrastiveModel`。
        dataloader: 数据加载器。
        device: 设备。
        use_projection: True 取投影特征 ``z``（对比空间），False 取句向量 ``h``。
        max_samples: ``>0`` 时最多抽取这么多样本。

    Returns:
        ``(features [N, d] 的 numpy 数组, labels [N] 的 numpy 数组, uids)``。
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("extract_features 需要 torch")
    import numpy as np

    model.eval()
    features: List[Any] = []
    labels: List[Any] = []
    uids: List[str] = []
    collected = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.get("original", batch)
            # 三个张量必须一起搬到 device：只搬 input_ids/attention_mask 而漏掉
            # token_type_ids，在 GPU 上第一次调用就会 RuntimeError（device mismatch）。
            token_type_ids = inputs.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                token_type_ids=token_type_ids,
            )
            tensor = outputs["projection"] if use_projection else outputs.get("hidden")
            if tensor is None:
                tensor = outputs["projection"]
            features.append(tensor.detach().cpu().numpy())
            labels.append(batch["label"].detach().cpu().numpy())
            uids.extend(list(batch.get("uid") or []))
            collected += tensor.shape[0]
            if max_samples and collected >= max_samples:
                break

    if not features:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), []

    feature_array = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    if max_samples and feature_array.shape[0] > max_samples:
        feature_array = feature_array[:max_samples]
        label_array = label_array[:max_samples]
        uids = uids[:max_samples]
    return feature_array, label_array, uids


# ---------------------------------------------------------------------- #
# 报告与落盘
# ---------------------------------------------------------------------- #
def format_report(metrics: Mapping[str, Any], label_list: Optional[Sequence[str]] = None) -> str:
    """把指标格式化成与论文 Table 3–8 同构的一张表（便于直接抄进论文）。"""
    labels = list(label_list or LABELS)
    lines: List[str] = []
    header = "Method      ACC      " + "".join(f"F1-{label:<6}" for label in labels) + "Avg F1"
    lines.append(header)
    lines.append("-" * len(header))
    per_class = metrics.get("per_class", {})
    cells = "".join(
        f"{per_class.get(label, {}).get('f1', 0.0) * 100:<9.2f}%" for label in labels
    )
    lines.append(
        f"{metrics.get('name', 'Proposed'):<11} "
        f"{metrics.get('acc', 0.0) * 100:<8.2f}% "
        f"{cells}"
        f"{metrics.get('avg_f1', 0.0) * 100:.2f}%"
    )
    lines.append("")
    lines.append(
        f"样本数 {metrics.get('num_samples', 0)}   "
        f"宏平均 P/R {metrics.get('macro_precision', 0.0):.4f}/"
        f"{metrics.get('macro_recall', 0.0):.4f}   "
        f"加权 F1 {metrics.get('weighted_f1', 0.0):.4f}"
    )
    return "\n".join(lines)


def save_predictions(
    path: str,
    uids: Sequence[str],
    predictions: Sequence[Any],
    references: Sequence[Any],
    probabilities: Optional[Any] = None,
) -> str:
    """保存 ``uid / pred / true`` 三元组（论文附录常用的预测明细）。

    Args:
        probabilities: 可选的 ``[N, C]`` 概率矩阵，写入每类的置信度。
    """
    records: List[Dict[str, Any]] = []
    for index, uid in enumerate(uids):
        pred = predictions[index]
        true = references[index]
        pred_id = _as_int(pred)
        true_id = _as_int(true)
        record: Dict[str, Any] = {
            "uid": str(uid),
            "pred": ID_TO_LABEL.get(pred_id, str(pred)) if pred_id is not None else str(pred),
            "true": ID_TO_LABEL.get(true_id, str(true)) if true_id is not None else str(true),
            "pred_id": pred_id,
            "true_id": true_id,
        }
        if probabilities is not None:
            try:
                record["probabilities"] = [float(value) for value in probabilities[index]]
            except (TypeError, IndexError):  # pragma: no cover
                pass
        records.append(record)
    return write_jsonl(path, records)


def _as_int(value: Any) -> Optional[int]:
    """把张量 / numpy 标量 / Python 数字统一成 int；不可转换时返回 None。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):  # pragma: no cover
            return None
    return None


def save_report(path: str, metrics: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> str:
    """把指标保存成 JSON（去掉 sklearn 报告里不可序列化的部分）。"""
    payload: Dict[str, Any] = {
        key: value
        for key, value in metrics.items()
        if key != "report"
    }
    report = metrics.get("report")
    if isinstance(report, Mapping):
        payload["sklearn_report"] = {
            key: value for key, value in report.items() if isinstance(value, Mapping)
        }
    if extra:
        payload.update(dict(extra))
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    save_json(path, payload)
    return path


def log_metrics(metrics: Mapping[str, Any], logger: Optional[Any] = None) -> None:
    """把指标打到日志里。"""
    log = logger or get_logger("evaluate")
    log.info(format_report(metrics))

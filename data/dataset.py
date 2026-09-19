# coding=utf-8
"""PyTorch 数据集：把"原样本 + 其 LLM 增强样本"配对成对比学习的输入。

对比学习的正样本对来自论文 §3.2 的描述：

    在每个 batch 中，(x_i, x'_i) 被配对为正样本，而 (x_i, x'_j) 被视为负样本。

因此 Dataset 的 ``__getitem__`` 返回一个**样本组**：

    {
        "uid": str,
        "label": int,
        "original": {"input_ids", "attention_mask", "token_type_ids"},
        "augmented": [ {...}, ... ],     # 该样本在指定轮次下的所有增强样本
        "n_augmented": int,
    }

``collate_pairs`` 负责把 batch 内的样本组整理成形状规整的张量：

* 若同一 batch 内所有样本的增强份数相同，返回 ``[B, K, L]`` 的规整张量；
* 若不同（某些样本增强失败被丢弃），返回张量列表 ``[[B, L], ...]``，
  由损失函数按最小份数对齐，避免为了形状规整而丢弃数据。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - 环境探测分支
    import torch
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """torch 缺失时的占位基类，保证本模块可被静态导入/语法检查。"""

    class DataLoader:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("使用 DataLoader 需要先安装 torch")

    _TORCH_AVAILABLE = False

from data.processors.data_model import LABELS, DataInstance, record_to_instance
from data.processors.reply_flatten import build_encoder_text
from src.utils.io_utils import iter_jsonl

__all__ = [
    "EncodedText",
    "PairDataset",
    "collate_pairs",
    "collate_single",
    "build_dataloader",
    "load_instances",
    "group_by_uid",
    "TORCH_AVAILABLE",
]

TORCH_AVAILABLE = _TORCH_AVAILABLE


class EncodedText(Dict[str, Any]):
    """编码结果的轻量容器。

    之所以用 dict 子类而不是 dataclass：它可以直接被 ``**`` 展开喂给模型
    ``forward(input_ids=..., attention_mask=..., token_type_ids=...)``。
    """

    @property
    def length(self) -> int:
        return len(self["input_ids"])


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "本模块的 Dataset/DataLoader 需要 torch。"
            "若只想做数据预处理，请直接使用 data.processors（不依赖 torch）。"
        )


# ---------------------------------------------------------------------- #
# 读取
# ---------------------------------------------------------------------- #
def load_instances(path: str, limit: int = 0, logger: Any = None) -> List[DataInstance]:
    """从 JSONL 读取实例列表。

    Args:
        path: JSONL 路径。
        limit: ``>0`` 时只读取前 N 条（用于冒烟测试）。
        logger: 可选 logger。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"数据文件不存在：{path}。请先运行 scripts/prepare_data.py"
        )
    instances: List[DataInstance] = []
    for index, record in enumerate(iter_jsonl(path)):
        if limit and index >= limit:
            break
        instances.append(record_to_instance(record))
    if logger is not None:
        logger.info(f"从 {path} 读取 {len(instances)} 条实例")
    return instances


def group_by_uid(instances: Iterable[DataInstance]) -> Dict[str, List[DataInstance]]:
    """按 ``original_uid`` 分组，把增强样本挂到对应原样本下。

    Returns:
        ``{original_uid: [instance, ...]}``，列表第一个元素是**原样本**
        （若该 uid 没有原样本，则第一条增强样本占据该位置），
        其余是增强样本，按 ``augment_round`` 升序。
    """
    buckets: Dict[str, Dict[str, List[DataInstance]]] = {}
    for instance in instances:
        key = instance.original_uid or instance.uid
        bucket = buckets.setdefault(key, {"original": [], "augmented": []})
        bucket["augmented" if instance.is_augmented else "original"].append(instance)

    groups: Dict[str, List[DataInstance]] = {}
    for key, bucket in buckets.items():
        augmented = sorted(
            bucket["augmented"], key=lambda item: (item.augment_round, item.uid)
        )
        originals = bucket["original"]
        if originals:
            groups[key] = list(originals) + augmented
        else:
            # 只有增强文件被加载时，用第一条增强样本占位，保证长度语义一致
            groups[key] = list(augmented)
    return groups


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class PairDataset(Dataset):
    """原样本 + 增强样本的配对数据集。

    Args:
        originals: 原样本列表（``split`` 已确定）。
        augmented: 增强样本列表；按 ``original_uid`` 自动挂到对应原样本。
        tokenizer: 任何提供 ``encode(text, max_length) -> dict`` 的对象
            （见 :class:`src.models.tokenization.BertTextEncoder`）。
        label_list: 标签顺序，默认 ``["NR","FR","TR","UR"]``。
        max_seq_length: 序列最大长度。
        augmented_round: 只使用该轮次的增强样本；``0`` 表示使用全部轮次。
        per_sample: 每个样本最多使用几份增强样本（``0`` 表示不限）。
        require_augmented: 为 True 时丢弃没有增强样本的样本。
    """

    def __init__(
        self,
        originals: Sequence[DataInstance],
        augmented: Optional[Sequence[DataInstance]] = None,
        tokenizer: Optional[Any] = None,
        label_list: Optional[Sequence[str]] = None,
        max_seq_length: int = 128,
        augmented_round: int = 0,
        per_sample: int = 1,
        require_augmented: bool = False,
        text_mode: str = "source_replies",
    ):
        _require_torch()
        self.label_list = list(label_list or LABELS)
        self.label_to_id = {label: index for index, label in enumerate(self.label_list)}
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.augmented_round = augmented_round
        self.per_sample = per_sample
        self.text_mode = text_mode

        augmented = list(augmented or [])
        self.augmented_by_uid: Dict[str, List[DataInstance]] = {}
        for item in augmented:
            if augmented_round and item.augment_round != augmented_round:
                continue
            self.augmented_by_uid.setdefault(item.original_uid or item.uid, []).append(item)
        for key in self.augmented_by_uid:
            self.augmented_by_uid[key].sort(key=lambda item: (item.augment_round, item.uid))

        self.samples: List[Tuple[DataInstance, List[DataInstance]]] = []
        skipped_no_aug = 0
        for original in originals:
            group = self.augmented_by_uid.get(original.uid, [])
            if self.per_sample and self.per_sample > 0:
                group = group[: self.per_sample]
            if require_augmented and not group:
                skipped_no_aug += 1
                continue
            self.samples.append((original, group))

        self.skipped_no_aug = skipped_no_aug

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, instance: DataInstance) -> Dict[str, Any]:
        # 增强样本落盘时可能没有 text 字段（LLM 只回传 uid/string_value/replies），
        # 这里用与原样本完全相同的规则现场推导，保证两者序列长度分布一致。
        text = instance.text or build_encoder_text(
            instance,
            text_mode=self.text_mode,
            max_seq_length=self.max_seq_length,
        )
        if self.tokenizer is None:
            # 无 tokenizer 时退化为字符级 id，仅用于测试张量形状
            ids = [min(ord(ch), 1000) for ch in text[: self.max_seq_length]]
            ids = ids + [0] * (self.max_seq_length - len(ids))
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [1 if token != 0 else 0 for token in ids], dtype=torch.long
                ),
                "token_type_ids": torch.zeros(self.max_seq_length, dtype=torch.long),
            }
        encoded = self.tokenizer(text, max_length=self.max_seq_length)
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(
                encoded.get("token_type_ids", [0] * self.max_seq_length), dtype=torch.long
            ),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        original, group = self.samples[index]
        label_id = self.label_to_id.get(original.label, original.label_id or 0)
        return {
            "uid": original.uid,
            "label": int(label_id),
            "original": self._encode(original),
            "augmented": [self._encode(item) for item in group],
            "n_augmented": len(group),
        }


# ---------------------------------------------------------------------- #
# collate
# ---------------------------------------------------------------------- #
def _stack(features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """把一组编码结果堆叠成 ``[B, L]`` 张量。"""
    keys = ("input_ids", "attention_mask", "token_type_ids")
    return {key: torch.stack([item[key] for item in features], dim=0) for key in keys}


def collate_pairs(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """把 :class:`PairDataset` 的 batch 整理成模型输入。

    Returns:
        ``{"uid", "label", "original", "augmented", "n_augmented"}``；
        ``augmented`` 在份数对齐时为 ``[B, K, L]`` 张量，否则为 ``[[B, L], ...]`` 列表。
    """
    _require_torch()
    uids = [item["uid"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    originals = _stack([item["original"] for item in batch])

    counts = {item["n_augmented"] for item in batch}
    augmented: Any
    if len(counts) == 1 and counts != {0}:
        # 份数一致：直接堆成 [B, K, L]
        stacked = [
            _stack([item["augmented"][k] for item in batch])
            for k in range(counts.pop())
        ]
        augmented = {
            key: torch.stack([layer[key] for layer in stacked], dim=1)
            for key in ("input_ids", "attention_mask", "token_type_ids")
        }
    else:
        # 份数不一致（或全部为 0）：返回列表，由损失函数按最小份数对齐
        max_k = max((item["n_augmented"] for item in batch), default=0)
        augmented = []
        for k in range(max_k):
            members = [item for item in batch if item["n_augmented"] > k]
            augmented.append(_stack([item["augmented"][k] for item in members]))

    return {
        "uid": uids,
        "label": labels,
        "original": originals,
        "augmented": augmented,
        "n_augmented": [item["n_augmented"] for item in batch],
    }


def collate_single(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """只要原样本的 batch（用于纯分类评测 / 特征提取）。"""
    _require_torch()
    return {
        "uid": [item["uid"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "original": _stack([item["original"] for item in batch]),
    }


# ---------------------------------------------------------------------- #
# DataLoader 工厂
# ---------------------------------------------------------------------- #
def build_dataloader(
    dataset: "Dataset",
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    paired: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> "DataLoader":
    """构造 DataLoader。

    Note:
        ``paired=True`` 时使用 :func:`collate_pairs`；评测/特征提取场景用
        ``paired=False`` 走 :func:`collate_single`，避免无谓地堆叠增强样本。
    """
    _require_torch()
    generator = None
    if seed is not None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_pairs if paired else collate_single,
        generator=generator,
    )

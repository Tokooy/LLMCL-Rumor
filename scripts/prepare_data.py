# coding=utf-8
"""数据预处理入口：原始数据 → ``data/processed/<dataset>/{train,dev,test}.jsonl``。

对应论文 §3.1 的"数据预处理"与 Fig.2 的数据结构。

用法::

    # Twitter15（需先把原始文件放进 data/raw/twitter15/）
    python scripts/prepare_data.py --config configs/base.yaml --dataset twitter15

    # 演示数据（无需任何外部数据集，用于跑通流程）
    python scripts/prepare_data.py --config configs/base.yaml --dataset demo

    # 覆盖配置项
    python scripts/prepare_data.py --config configs/base.yaml --dataset twitter15 \\
        --text-mode hierarchical --max-seq-length 256 --reply-order dfs
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.processors.data_model import DataInstance, LABELS  # noqa: E402
from data.processors.reply_flatten import build_encoder_text  # noqa: E402
from data.processors.twitter_rumor import DATASET_DEFAULT_SPLIT  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io_utils import ensure_dir, write_jsonl  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DEMO_DIR = os.path.join("data", "samples")
SPLITS = ("train", "dev", "test")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMCL-Rumor 数据预处理")
    parser.add_argument("--config", action="append", default=None,
                        help="配置文件路径，可重复传入，后者覆盖前者")
    parser.add_argument("--dataset", default=None,
                        help="数据集名：twitter15 / twitter16 / demo；默认取配置里的 data.name")
    parser.add_argument("--raw-dir", default=None, help="原始数据根目录，默认取配置 paths.raw_dir")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认 data/processed/<dataset>")
    parser.add_argument("--text-mode", default=None,
                        choices=["source_only", "source_replies", "hierarchical"])
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--reply-order", default=None, choices=["bfs", "dfs"])
    parser.add_argument("--max-replies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条样本（调试用），0 表示全部")
    return parser.parse_args(argv)


def _resolve_config_paths(configs: List[str] | None) -> List[str]:
    if configs:
        return configs
    return [os.path.join(REPO_ROOT, "configs", "base.yaml")]


def load_demo_instances(limit: int = 0) -> List[DataInstance]:
    """加载仓库自带的演示样本（``data/samples/demo_twitter15.jsonl``）。"""
    from data.dataset import load_instances  # 延迟导入，避免无 torch 时报错

    path = os.path.join(REPO_ROOT, DEMO_DIR, "demo_twitter15.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"演示数据不存在：{path}。请先运行 python data/samples/make_demo.py"
        )
    return load_instances(path, limit=limit)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(*_resolve_config_paths(args.config))

    dataset = args.dataset or config.get_path("data.name", "twitter15")
    raw_dir = args.raw_dir or config.get_path("paths.raw_dir", "data/raw")
    text_mode = args.text_mode or config.get_path("data.text_mode", "source_replies")
    max_seq_length = args.max_seq_length or int(config.get_path("data.max_seq_length", 128))
    reply_order = args.reply_order or config.get_path("data.reply_order", "bfs")
    max_replies = args.max_replies if args.max_replies is not None else int(
        config.get_path("data.max_replies", 20)
    )
    seed = args.seed if args.seed is not None else int(config.get_path("seed", 42))
    split_cfg = config.get_path("data.split", {}) or {}

    out_dir = args.out_dir or os.path.join(
        config.get_path("paths.processed_dir", "data/processed"), dataset
    )
    ensure_dir(os.path.join(REPO_ROOT, out_dir))

    logger = get_logger("prepare_data")
    logger.info(f"数据集={dataset}  text_mode={text_mode}  max_seq_length={max_seq_length}")
    logger.info(f"原始目录={raw_dir}  输出目录={out_dir}")

    set_seed(seed)

    # ------------------------------------------------------------------ #
    # 1) 加载
    # ------------------------------------------------------------------ #
    if dataset == "demo":
        instances = load_demo_instances(limit=args.limit)
        # 演示数据没有官方划分，按配置比例分层抽样
        from data.processors.twitter_rumor import assign_splits

        ratios = dict(DATASET_DEFAULT_SPLIT)
        ratios.update({key: float(value) for key, value in (split_cfg or {}).items()})
        counts = assign_splits(
            instances,
            train_ratio=ratios.get("train", 0.7),
            dev_ratio=ratios.get("dev", 0.1),
            test_ratio=ratios.get("test", 0.2),
            seed=seed,
        )
        logger.info(f"演示数据划分完成：{counts}")
    else:
        from data.processors.twitter_rumor import load_raw_dataset

        raw_root = raw_dir if os.path.isabs(raw_dir) else os.path.join(REPO_ROOT, raw_dir)
        # 演示数据放在了 data/samples 下，这里把 raw_root 归一化，便于统一处理
        instances = load_raw_dataset(
            raw_dir=raw_root,
            name=dataset,
            text_mode=text_mode,
            max_seq_length=max_seq_length,
            reply_order=reply_order,
            max_replies=max_replies,
            split_ratios=split_cfg,
            seed=seed,
            prefer_official_split=bool(split_cfg.get("prefer_official_split", True)),
            logger=logger,
        )

    if args.limit:
        instances = instances[: args.limit]

    # ------------------------------------------------------------------ #
    # 2) 回填编码器输入（demo 路径需要；twitter15/16 路径已在加载时做过）
    # ------------------------------------------------------------------ #
    for instance in instances:
        instance.text = build_encoder_text(
            instance,
            text_mode=text_mode,
            max_seq_length=max_seq_length,
            reply_order=reply_order,
            max_replies=max_replies,
        )
        instance.dataset = dataset

    # ------------------------------------------------------------------ #
    # 3) 按 split 落盘 + 打印统计
    # ------------------------------------------------------------------ #
    summary: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        members = [item for item in instances if item.split == split]
        path = os.path.join(REPO_ROOT, out_dir, f"{split}.jsonl")
        write_jsonl(path, (item.to_record() for item in members))
        label_counts = {label: 0 for label in LABELS}
        for item in members:
            label_counts[item.label] = label_counts.get(item.label, 0) + 1
        summary[split] = {"total": len(members), **label_counts}
        logger.info(
            f"{split}: {len(members)} 条  "
            + "  ".join(f"{label}={label_counts[label]}" for label in LABELS)
            + f"  -> {path}"
        )

    # 类别分布核对论文 Table 1（Twitter15: 374/370/372/374，Twitter16: 205/205/207/201）
    total = sum(item["total"] for item in summary.values())
    logger.info(f"合计 {total} 条样本")
    if dataset in ("twitter15", "twitter16"):
        expected = 1490 if dataset == "twitter15" else 818
        if total != expected:
            logger.warning(
                f"论文 Table 1 中 {dataset} 共 {expected} 条源推文，当前解析出 {total} 条。"
                "差异通常来自已被删除的推文或原始文件不完整，请在论文中说明实际样本量。"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

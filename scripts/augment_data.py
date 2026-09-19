# coding=utf-8
"""LLM 数据增强入口（论文 §3.1，对应 Fg.3 / Fg.4）。

用法::

    # 用演示后端跑通链路（不需要 GPU、不需要模型权重）
    python scripts/augment_data.py --config configs/experiments/proposed-1.yaml \\
        --dataset demo --round 1 --backend demo

    # 用本地 Qwen 做真实增强
    python scripts/augment_data.py --config configs/experiments/proposed-1.yaml \\
        --dataset twitter15 --round 1 --backend transformers

    # 用 OpenAI 兼容接口
    python scripts/augment_data.py --config configs/experiments/proposed-3.yaml \\
        --dataset twitter15 --round 2 --backend api --concurrency 8

输出：``data/processed/<dataset>/augmented_round<k>.jsonl``，每行是一条增强样本
（``augmented=true``、``augment_round=k``、``original_uid=<原样本 uid>``）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.llm.augmentor import load_original_split  # noqa: E402
from src.llm.factory import build_augmentor  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io_utils import ensure_dir, write_jsonl  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 数据增强")
    parser.add_argument("--config", action="append", default=None,
                        help="配置文件路径，可重复传入，后者覆盖前者")
    parser.add_argument("--dataset", default=None, help="数据集名，默认取配置 data.name")
    parser.add_argument("--round", type=int, default=1, dest="augment_round",
                        help="增强轮次编号（论文的 w 轮增强，从 1 开始）")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"],
                        help="对哪个划分做增强；论文只增强训练集")
    parser.add_argument("--backend", default=None,
                        help="覆盖 llm.backend：transformers / api / demo")
    parser.add_argument("--prompt-file", default=None, help="覆盖 Prompt 模板路径")
    parser.add_argument("--copies", type=int, default=None, help="每个样本生成几份增强")
    parser.add_argument("--concurrency", type=int, default=None, help="并发数")
    parser.add_argument("--max-retries", type=int, default=None, help="单条最大重试次数")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试用）")
    parser.add_argument("--out", default=None, help="输出文件路径，默认按轮次自动命名")
    parser.add_argument("--no-cache", action="store_true", help="禁用增强缓存")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_paths = args.config or [os.path.join(REPO_ROOT, "configs", "base.yaml")]
    config = load_config(*config_paths)

    dataset = args.dataset or config.get_path("data.name", "twitter15")
    seed = args.seed if args.seed is not None else int(config.get_path("seed", 42))
    set_seed(seed)

    processed_dir = os.path.join(
        REPO_ROOT, config.get_path("paths.processed_dir", "data/processed"), dataset
    )
    ensure_dir(processed_dir)

    logger = get_logger(
        "augment_data",
        log_file=os.path.join(
            REPO_ROOT, config.get_path("paths.log_dir", "outputs/logs"),
            f"augment_{dataset}_round{args.augment_round}.log",
        ),
    )
    logger.info(f"配置文件：{[os.path.abspath(path) for path in config_paths]}")
    logger.info(f"数据集={dataset}  轮次={args.augment_round}  划分={args.split}")

    # ------------------------------------------------------------------ #
    # 1) 读原样本
    # ------------------------------------------------------------------ #
    instances = load_original_split(processed_dir, split=args.split, limit=args.limit)
    if not instances:
        logger.error(f"{processed_dir}/{args.split}.jsonl 中没有样本，请先运行 prepare_data.py")
        return 1
    logger.info(f"待增强样本 {len(instances)} 条")

    # ------------------------------------------------------------------ #
    # 2) 装配增强器
    # ------------------------------------------------------------------ #
    overrides = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.copies is not None:
        overrides["copies_per_sample"] = args.copies
    if args.concurrency is not None:
        overrides["concurrency"] = args.concurrency
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries

    augmentor = build_augmentor(
        config,
        template_file=args.prompt_file,
        logger=logger,
        overrides=overrides,
    )
    if args.no_cache:
        augmentor.cache_dir = ""
    logger.info(f"后端：{augmentor.backend.describe()}")
    logger.info(
        f"增强参数：copies={augmentor.copies_per_sample} 并发={augmentor.concurrency} "
        f"重试={augmentor.max_retries} 缓存={'启用' if augmentor.cache_dir else '关闭'}"
    )

    # ------------------------------------------------------------------ #
    # 3) 增强
    # ------------------------------------------------------------------ #
    augmented, stats = augmentor.augment(instances, augment_round=args.augment_round)

    # ------------------------------------------------------------------ #
    # 4) 落盘
    # ------------------------------------------------------------------ #
    out_path = args.out or os.path.join(
        processed_dir, f"augmented_round{args.augment_round}.jsonl"
    )
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    write_jsonl(out_path, (item.to_record() for item in augmented))
    logger.info(f"已写出 {len(augmented)} 条增强样本 -> {out_path}")

    summary = stats.summary()
    logger.info(f"统计：{summary}")
    if summary["success_rate"] < 0.9:
        logger.warning(
            "增强成功率低于 90%。失败样本以'未增强副本'保留，"
            "若比例过高建议检查 Prompt 约束或降低温度后重跑（缓存会自动跳过已成功样本）"
        )

    # 质量指标均值，便于和论文"多样性提升/语义一致"的定性描述对照
    overlaps = [
        item.quality.get("word_overlap")
        for item in augmented
        if item.quality and item.quality.get("word_overlap") is not None
    ]
    if overlaps:
        logger.info(
            f"词级重合度：均值 {sum(overlaps) / len(overlaps):.3f}，"
            f"最小 {min(overlaps):.3f}，最大 {max(overlaps):.3f}（越低表示改写幅度越大）"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

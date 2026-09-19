# coding=utf-8
"""CL 网络训练入口（论文 §3.2，对应 Proposed-1/2/3 的 M=0 设置）。

用法::

    # 用演示数据跑通（需要 torch + transformers，但不下载 BERT 权重也能跑——见 --dry-run）
    python scripts/train_cl.py --config configs/experiments/proposed-1.yaml --dataset demo

    # 正式训练
    python scripts/train_cl.py --config configs/experiments/proposed-3.yaml --dataset twitter15

    # 指定增强轮次（决定读取哪个 augmented_round<k>.jsonl）
    python scripts/train_cl.py --config configs/experiments/proposed-2.yaml --augment-round 2

本脚本只做"给定数据，训练 CL 网络"。需要"增强 + LLM 微调"的闭环请用
``scripts/joint_align.py``（论文 Algorithm 2）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.config import load_config  # noqa: E402
from src.utils.io_utils import ensure_dir  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

SPLIT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL 网络训练（论文 §3.2）")
    parser.add_argument("--config", action="append", default=None, help="配置文件，可重复")
    parser.add_argument("--dataset", default=None, help="数据集名，默认取配置 data.name")
    parser.add_argument("--augment-round", type=int, default=None,
                        help="使用第几轮增强数据；0 表示不使用增强（纯分类基线）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本（调试）")
    parser.add_argument("--output-dir", default=None, help="checkpoint 输出目录")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def resolve_processed_dir(config: Any, dataset: str) -> str:
    path = os.path.join(
        REPO_ROOT, config.get_path("paths.processed_dir", "data/processed"), dataset
    )
    return path


def build_tokenizer(config: Any, max_seq_length: int):
    """构造 BERT 分词器（三个划分共用同一份，避免重复加载词表）。"""
    from src.models.tokenization import BertTextEncoder

    encoder_cfg = config.get_path("model.encoder", {}) or {}
    return BertTextEncoder(
        model_name=encoder_cfg.get("name", "bert-base-uncased"),
        local_dir=encoder_cfg.get("local_dir", "") or "",
        max_seq_length=max_seq_length,
        do_lower_case=bool(encoder_cfg.get("do_lower_case", True)),
    )


def load_datasets(
    processed_dir: str,
    config: Any,
    max_seq_length: int,
    augment_round: int,
    tokenizer: Any = None,
    limit: int = 0,
    logger: Optional[Any] = None,
):
    """构造 train/dev/test 的 DataLoader。

    增强样本只挂在训练集上——论文只对训练集做增强（若对测试集增强，
    就变成"用 LLM 改写过的测试集评测"，与论文口径不符）。
    """
    from data.dataset import PairDataset, build_dataloader, load_instances

    label_list = config.get_path("data.label_list", None) or ["NR", "FR", "TR", "UR"]
    per_sample = int(config.get_path("data.augmented_per_sample", 1) or 0)
    text_mode = config.get_path("data.text_mode", "source_replies")

    num_workers = int(config.get_path("training.cl.num_workers", 0))
    pin_memory = bool(config.get_path("training.cl.pin_memory", True))
    batch_size = int(config.get_path("training.cl.batch_size", 32))
    eval_batch_size = int(config.get_path("training.cl.eval_batch_size", 64))

    train_originals = load_instances(
        os.path.join(processed_dir, SPLIT_FILES["train"]), limit=limit, logger=logger
    )
    train_augmented = []
    aug_path = os.path.join(processed_dir, f"augmented_round{augment_round}.jsonl")
    if augment_round > 0 and os.path.isfile(aug_path):
        train_augmented = load_instances(aug_path, limit=0, logger=logger)
        if logger is not None:
            logger.info(f"加载第 {augment_round} 轮增强样本 {len(train_augmented)} 条：{aug_path}")
    elif augment_round > 0 and logger is not None:
        logger.warning(
            f"找不到 {aug_path}，本轮不使用增强样本。请先运行 scripts/augment_data.py"
        )

    def _make(originals, augmented, round_index: int, per: int) -> "PairDataset":
        return PairDataset(
            originals=originals,
            augmented=augmented,
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_length,
            augmented_round=round_index,
            per_sample=per,
            require_augmented=False,
            text_mode=text_mode,
        )

    train_dataset = _make(train_originals, train_augmented, augment_round, per_sample)
    dev_dataset = _make(
        load_instances(os.path.join(processed_dir, SPLIT_FILES["dev"]), logger=logger), [], 0, 0
    )
    test_dataset = _make(
        load_instances(os.path.join(processed_dir, SPLIT_FILES["test"]), logger=logger), [], 0, 0
    )

    train_loader = build_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, paired=True,
        num_workers=num_workers, pin_memory=pin_memory,
        seed=int(config.get_path("seed", 42)),
    )
    dev_loader = build_dataloader(
        dev_dataset, batch_size=eval_batch_size, shuffle=False, paired=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = build_dataloader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, paired=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, dev_loader, test_loader, label_list


def build_model(config: Any, num_labels: int, logger: Optional[Any] = None):
    """按配置构造 CL 网络。"""
    from src.models.cl_model import build_contrastive_model

    model_cfg = config.get_path("model", {}) or {}
    model = build_contrastive_model(
        encoder_config=model_cfg.get("encoder", {}) or {},
        projector_config=model_cfg.get("projector", {}) or {},
        classifier_config=model_cfg.get("classifier", {}) or {},
        num_labels=num_labels,
    )
    if logger is not None:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.trainable_parameters())
        logger.info(
            f"模型：{model_cfg.get('encoder', {}).get('name', 'bert-base-uncased')} + "
            f"MLP 投影头(d={model.projection_dim}) + 分类器(C={num_labels})；"
            f"参数 {total / 1e6:.2f}M（可训练 {trainable / 1e6:.2f}M）"
        )
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_paths = args.config or [os.path.join(REPO_ROOT, "configs", "base.yaml")]
    config = load_config(*config_paths)

    dataset = args.dataset or config.get_path("data.name", "twitter15")
    seed = args.seed if args.seed is not None else int(config.get_path("seed", 42))
    max_seq_length = args.max_seq_length or int(config.get_path("data.max_seq_length", 128))
    augment_round = (
        args.augment_round
        if args.augment_round is not None
        else int(config.get_path("experiment.augment_rounds", 0) or 0)
    )

    # 覆盖配置（命令行优先）
    if args.epochs is not None:
        config.set_path("training.cl.epochs", args.epochs)
    if args.batch_size is not None:
        config.set_path("training.cl.batch_size", args.batch_size)
    if args.learning_rate is not None:
        config.set_path("training.cl.learning_rate", args.learning_rate)

    experiment = config.get_path("experiment.name", "default")
    output_dir = args.output_dir or os.path.join(
        REPO_ROOT,
        config.get_path("paths.checkpoint_dir", "outputs/checkpoints"),
        f"{experiment}_{dataset}",
    )
    ensure_dir(output_dir)

    logger = get_logger(
        "train_cl",
        log_file=os.path.join(
            REPO_ROOT, config.get_path("paths.log_dir", "outputs/logs"),
            f"train_cl_{experiment}_{dataset}.log",
        ),
    )
    logger.info(f"实验={experiment} 数据集={dataset} 增强轮次={augment_round}")
    logger.info(f"配置文件：{[os.path.abspath(path) for path in config_paths]}")

    set_seed(seed, deterministic=bool(config.get_path("deterministic", False)),
             cuda_devices=config.get_path("device.cuda_devices", "") or None)

    import torch

    from src.training.cl_trainer import CLTrainer
    from src.training.evaluate import evaluate_model, format_report, save_report
    from src.training.losses import CombinedLoss

    device, n_gpu = _resolve_device(config, torch, logger)

    processed_dir = resolve_processed_dir(config, dataset)
    tokenizer = build_tokenizer(config, max_seq_length)
    train_loader, dev_loader, test_loader, label_list = load_datasets(
        processed_dir, config, max_seq_length, augment_round,
        tokenizer=tokenizer, limit=args.limit, logger=logger,
    )
    logger.info(
        f"数据规模：train={len(train_loader.dataset)} dev={len(dev_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    model = build_model(config, num_labels=len(label_list), logger=logger).to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    contrastive_cfg = config.get_path("model.contrastive", {}) or {}
    criterion = CombinedLoss(
        temperature=float(contrastive_cfg.get("temperature", 0.07)),
        pairing=contrastive_cfg.get("pairing", "paired"),
        ce_weight=float(config.get_path("training.cl.ce_weight", 1.0)),
        cl_weight=float(config.get_path("training.cl.cl_weight", 1.0)),
        joint_objective=bool(config.get_path("training.cl.joint_objective", True)),
    ).to(device)

    trainer = CLTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        criterion=criterion,
        device=device,
        epochs=int(config.get_path("training.cl.epochs", 5)),
        learning_rate=float(config.get_path("training.cl.learning_rate", 2e-5)),
        weight_decay=float(config.get_path("training.cl.weight_decay", 0.01)),
        adam_epsilon=float(config.get_path("training.cl.adam_epsilon", 1e-8)),
        warmup_proportion=float(config.get_path("training.cl.warmup_proportion", 0.1)),
        gradient_accumulation_steps=int(
            config.get_path("training.cl.gradient_accumulation_steps", 1)
        ),
        max_grad_norm=float(config.get_path("training.cl.max_grad_norm", 1.0)),
        evaluate_every_epochs=int(config.get_path("training.cl.evaluate_every_epochs", 1)),
        save_best_on=config.get_path("training.cl.save_best_on", "avg_f1"),
        early_stop_patience=int(config.get_path("training.cl.early_stop_patience", 10)),
        output_dir=output_dir,
        log_every_steps=int(config.get_path("training.cl.log_every_steps", 50)),
        logger=logger,
    )

    trainer.run()

    # ------------------------------------------------------------------ #
    # 测试集评测
    # ------------------------------------------------------------------ #
    best_path = os.path.join(output_dir, "best.pt")
    if os.path.isfile(best_path):
        trainer.load_checkpoint(best_path, load_optimizer=False)
        logger.info(f"已加载最优 checkpoint：{best_path}")

    raw_model = model.module if hasattr(model, "module") else model
    test_metrics = evaluate_model(raw_model, test_loader, device, label_list=label_list)
    test_metrics["name"] = experiment
    logger.info("测试集结果：")
    logger.info("\n" + format_report(test_metrics, label_list))

    result_path = os.path.join(
        REPO_ROOT,
        config.get_path("paths.result_dir", "outputs/results"),
        f"{experiment}_{dataset}_cl.json",
    )
    save_report(result_path, test_metrics, extra={
        "dataset": dataset,
        "augment_round": augment_round,
        "config_sources": config.get_path("_config_sources", []),
    })
    logger.info(f"指标已保存：{result_path}")
    return 0


def _resolve_device(config: Any, torch: Any, logger: Any):
    """解析设备与 GPU 数量。"""
    cuda_devices = config.get_path("device.cuda_devices", "") or ""
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices).replace(",", " ")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        n_gpu = torch.cuda.device_count()
        logger.info(f"使用 GPU：{n_gpu} 张（CUDA_VISIBLE_DEVICES={cuda_devices!r}）")
    else:
        device = torch.device("cpu")
        n_gpu = 0
        logger.warning("未检测到可用 GPU，将在 CPU 上训练（不推荐）")
    return device, n_gpu


if __name__ == "__main__":
    sys.exit(main())

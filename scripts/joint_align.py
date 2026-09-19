# coding=utf-8
"""LLM 与 CL 联合对齐入口（论文 Algorithm 2，对应 Proposed-4/5/6）。

流程（每一步都对应论文里的一个动作）::

    for epoch in 1..N:
        CL 训练一个 epoch（论文式(3) 的 InfoNCE + 式(4) 的分类损失）
        验证集评测 → f_m
        式(8) 动量更新 λ
        每 T 个 epoch：调用 LLM 做一轮数据增强，扩充训练集
        每 M 个 epoch 周期：自举微调 LLM → 导出 τ → Algorithm 1 合并 → 写回
        达到 M 轮微调后停止（可配置）

用法::

    # 演示数据 + 规则后端，不需要 GPU 与模型权重
    python scripts/joint_align.py --config configs/experiments/proposed-4.yaml \\
        --dataset demo --backend demo

    # 正式实验
    python scripts/joint_align.py --config configs/experiments/proposed-5.yaml \\
        --dataset twitter15
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
from src.utils.io_utils import ensure_dir, save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM/CL 联合对齐（论文 Algorithm 2）")
    parser.add_argument("--config", action="append", default=None, help="配置文件，可重复")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--backend", default=None, help="覆盖 llm.backend：transformers/api/demo")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="CL 训练总 epoch 数 N")
    parser.add_argument("--max-augment-rounds", type=int, default=None, help="w")
    parser.add_argument("--max-finetune-rounds", type=int, default=None, help="M")
    parser.add_argument("--augment-interval", type=int, default=None, help="T：每几个 epoch 增强一次")
    parser.add_argument("--finetune-interval", type=int, default=None,
                        help="每几个 epoch 完成一次微调 + 合并")
    parser.add_argument("--lambda-source", default=None,
                        choices=["contrastive_accuracy", "avg_f1", "inverse_loss"])
    parser.add_argument("--beta", type=float, default=None, help="式(8) 的 β")
    parser.add_argument("--alpha", type=float, default=None, help="式(7) 的 α")
    parser.add_argument("--trim-percent", type=float, default=None, help="Algorithm 1 的 q")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--augment-round", type=int, default=0,
                        help="磁盘上已有增强数据的最大轮次；对齐流程从该轮次之后继续")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本（调试）")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-finetune", action="store_true",
                        help="强制 M=0（只做增强，不做微调）")
    return parser.parse_args(argv)


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    """把命令行参数写回配置（命令行优先于配置文件）。"""
    mapping = [
        ("--epochs", args.epochs, "training.cl.epochs"),
        ("--max-augment-rounds", args.max_augment_rounds, "training.alignment.max_augment_rounds"),
        ("--max-finetune-rounds", args.max_finetune_rounds, "training.alignment.max_finetune_rounds"),
        ("--augment-interval", args.augment_interval, "training.cl.augment_every_epochs"),
        ("--finetune-interval", args.finetune_interval, "training.alignment.finetune_interval_epochs"),
        ("--lambda-source", args.lambda_source, "training.alignment.lambda_source"),
        ("--beta", args.beta, "training.alignment.momentum_beta"),
        ("--alpha", args.alpha, "llm.merge.scaling_alpha"),
        ("--trim-percent", args.trim_percent, "llm.merge.trim_percent"),
        ("--batch-size", args.batch_size, "training.cl.batch_size"),
        ("--learning-rate", args.learning_rate, "training.cl.learning_rate"),
    ]
    for _flag, value, path in mapping:
        if value is not None:
            config.set_path(path, value)
    if args.no_finetune:
        config.set_path("training.alignment.max_finetune_rounds", 0)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_paths = args.config or [os.path.join(REPO_ROOT, "configs", "base.yaml")]
    config = load_config(*config_paths)
    _apply_overrides(config, args)

    dataset = args.dataset or config.get_path("data.name", "twitter15")
    seed = args.seed if args.seed is not None else int(config.get_path("seed", 42))
    max_seq_length = args.max_seq_length or int(config.get_path("data.max_seq_length", 128))
    experiment = config.get_path("experiment.name", "joint")

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT,
        config.get_path("paths.checkpoint_dir", "outputs/checkpoints"),
        f"{experiment}_{dataset}_joint",
    )
    ensure_dir(output_dir)

    logger = get_logger(
        "joint_align",
        log_file=os.path.join(
            REPO_ROOT, config.get_path("paths.log_dir", "outputs/logs"),
            f"joint_align_{experiment}_{dataset}.log",
        ),
    )
    logger.info(f"实验={experiment} 数据集={dataset}")
    logger.info(f"配置文件：{[os.path.abspath(path) for path in config_paths]}")

    set_seed(
        seed,
        deterministic=bool(config.get_path("deterministic", False)),
        cuda_devices=config.get_path("device.cuda_devices", "") or None,
    )

    import torch

    from data.dataset import PairDataset, build_dataloader, load_instances
    from src.llm.augmentor import load_original_split
    from src.llm.factory import build_augmentor, build_backend, build_prompt_builder
    from src.models.cl_model import build_contrastive_model
    from src.models.tokenization import BertTextEncoder
    from src.training.cl_trainer import CLTrainer
    from src.training.joint_trainer import JointAlignmentTrainer
    from src.training.losses import CombinedLoss
    from src.llm.ties_merge import TiesMerger

    device = _resolve_device(config, torch, logger)

    processed_dir = os.path.join(
        REPO_ROOT, config.get_path("paths.processed_dir", "data/processed"), dataset
    )

    # ------------------------------------------------------------------ #
    # 1) 数据
    # ------------------------------------------------------------------ #
    label_list = config.get_path("data.label_list", None) or ["NR", "FR", "TR", "UR"]
    text_mode = config.get_path("data.text_mode", "source_replies")
    tokenizer = BertTextEncoder(
        model_name=config.get_path("model.encoder.name", "bert-base-uncased"),
        local_dir=config.get_path("model.encoder.local_dir", "") or "",
        max_seq_length=max_seq_length,
        do_lower_case=bool(config.get_path("model.encoder.do_lower_case", True)),
    )

    train_originals = load_original_split(processed_dir, split="train", limit=args.limit)
    if not train_originals:
        logger.error(f"{processed_dir}/train.jsonl 为空，请先运行 scripts/prepare_data.py")
        return 1
    logger.info(f"训练集原样本 {len(train_originals)} 条")

    def _make_dataset(originals, augmented, round_index: int, per_sample: int) -> Any:
        return PairDataset(
            originals=originals,
            augmented=augmented,
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_length,
            augmented_round=round_index,
            per_sample=per_sample,
            require_augmented=False,
            text_mode=text_mode,
        )

    per_sample = int(config.get_path("data.augmented_per_sample", 1) or 0)
    # 起始时若磁盘上已有增强数据（例如先跑了 augment_data.py），直接纳入；
    # 并用其最大轮次初始化 augment_round，避免重复增强同一轮
    bootstrap_round = int(args.augment_round or 0)
    existing_augmented = []
    for round_index in range(1, 4):
        candidate = os.path.join(processed_dir, f"augmented_round{round_index}.jsonl")
        if os.path.isfile(candidate):
            existing_augmented.extend(load_instances(candidate))
            bootstrap_round = max(bootstrap_round, round_index)
    if existing_augmented:
        logger.info(
            f"检测到磁盘上已有的增强数据 {len(existing_augmented)} 条"
            f"（最大轮次 {bootstrap_round}），将纳入训练集"
        )

    train_dataset = _make_dataset(train_originals, existing_augmented, 0, per_sample)
    dev_dataset = _make_dataset(
        load_instances(os.path.join(processed_dir, "dev.jsonl")), [], 0, 0
    )
    test_dataset = _make_dataset(
        load_instances(os.path.join(processed_dir, "test.jsonl")), [], 0, 0
    )

    num_workers = int(config.get_path("training.cl.num_workers", 0))
    pin_memory = bool(config.get_path("training.cl.pin_memory", True))
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config.get_path("training.cl.batch_size", 32)),
        shuffle=True, paired=True, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
    )
    dev_loader = build_dataloader(
        dev_dataset,
        batch_size=int(config.get_path("training.cl.eval_batch_size", 64)),
        shuffle=False, paired=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=int(config.get_path("training.cl.eval_batch_size", 64)),
        shuffle=False, paired=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------ #
    # 2) 模型与损失
    # ------------------------------------------------------------------ #
    model_cfg = config.get_path("model", {}) or {}
    model = build_contrastive_model(
        encoder_config=model_cfg.get("encoder", {}) or {},
        projector_config=model_cfg.get("projector", {}) or {},
        classifier_config=model_cfg.get("classifier", {}) or {},
        num_labels=len(label_list),
    ).to(device)
    contrastive_cfg = model_cfg.get("contrastive", {}) or {}
    criterion = CombinedLoss(
        temperature=float(contrastive_cfg.get("temperature", 0.07)),
        pairing=contrastive_cfg.get("pairing", "paired"),
        ce_weight=float(config.get_path("training.cl.ce_weight", 1.0)),
        cl_weight=float(config.get_path("training.cl.cl_weight", 1.0)),
        joint_objective=bool(config.get_path("training.cl.joint_objective", True)),
    ).to(device)

    # ------------------------------------------------------------------ #
    # 3) LLM 侧：增强器 + 后端 + 合并器
    # ------------------------------------------------------------------ #
    llm_backend = build_backend(config, args.backend)
    prompt_builder = build_prompt_builder(config, template_file=args.prompt_file)
    augmentor = build_augmentor(
        config, backend=llm_backend, template_file=args.prompt_file, logger=logger
    )
    logger.info(f"LLM 后端：{llm_backend.describe()}")

    merger = TiesMerger(
        trim_percent=float(config.get_path("llm.merge.trim_percent", 20.0)),
        alpha=(
            None
            if config.get_path("llm.merge.scaling_alpha", 0.5) is None
            else float(config.get_path("llm.merge.scaling_alpha", 0.5))
        ),
        lambda_init=float(config.get_path("training.alignment.lambda_init", 1.0)),
        merge_all_checkpoints=bool(config.get_path("llm.merge.merge_all_checkpoints", True)),
    )

    # ------------------------------------------------------------------ #
    # 4) 训练器
    # ------------------------------------------------------------------ #
    cl_trainer = CLTrainer(
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

    joint = JointAlignmentTrainer(
        cl_trainer=cl_trainer,
        augmentor=augmentor,
        llm_backend=llm_backend,
        merger=merger,
        prompt_builder=prompt_builder,
        processed_dir=processed_dir,
        dataset=dataset,
        max_augment_rounds=int(config.get_path("training.alignment.max_augment_rounds", 3)),
        max_finetune_rounds=int(config.get_path("training.alignment.max_finetune_rounds", 0)),
        augment_interval_epochs=int(config.get_path("training.cl.augment_every_epochs", 1)),
        finetune_interval_epochs=int(
            config.get_path("training.alignment.finetune_interval_epochs", 5)
        ),
        momentum_beta=float(config.get_path("training.alignment.momentum_beta", 0.9)),
        lambda_source=config.get_path("training.alignment.lambda_source", "contrastive_accuracy"),
        omega_min=float(config.get_path("training.alignment.omega_min", 0.05)),
        omega_max=float(config.get_path("training.alignment.omega_max", 0.95)),
        lambda_init=float(config.get_path("training.alignment.lambda_init", 1.0)),
        output_dir=output_dir,
        stop_when_max_reached=bool(
            config.get_path("training.alignment.stop_when_max_reached", False)
        ),
        reset_classifier_each_round=bool(
            config.get_path("training.alignment.reset_classifier_each_round", False)
        ),
        logger=logger,
    )
    joint.augment_round = bootstrap_round  # 已有增强数据时从该轮次之后继续

    # ------------------------------------------------------------------ #
    # 5) 跑 Algorithm 2
    # ------------------------------------------------------------------ #
    # existing_augmented 必须一并交给训练器：否则第一次重建训练集时，
    # 这些"已经在训练集里用了"的磁盘增强样本会从训练集中静默消失。
    state = joint.fit(train_originals, existing_augmented=existing_augmented)

    # ------------------------------------------------------------------ #
    # 6) 测试集评测
    # ------------------------------------------------------------------ #
    from src.training.evaluate import evaluate_model, format_report, save_report

    best_path = os.path.join(output_dir, "best.pt")
    if os.path.isfile(best_path):
        cl_trainer.load_checkpoint(best_path, load_optimizer=False)
        logger.info(f"已加载最优 checkpoint：{best_path}")

    raw_model = model.module if hasattr(model, "module") else model
    test_metrics = evaluate_model(raw_model, test_loader, device, label_list=label_list)
    test_metrics["name"] = experiment
    logger.info("测试集结果：")
    logger.info("\n" + format_report(test_metrics, label_list))

    result_dir = os.path.join(REPO_ROOT, config.get_path("paths.result_dir", "outputs/results"))
    result_path = os.path.join(result_dir, f"{experiment}_{dataset}_joint.json")
    save_report(
        result_path,
        test_metrics,
        extra={
            "dataset": dataset,
            "alignment_state": state.to_dict(),
            "merger": merger.describe(),
            "config_sources": config.get_path("_config_sources", []),
        },
    )
    logger.info(f"指标与对齐状态已保存：{result_path}")

    save_json(
        os.path.join(output_dir, "run_summary.json"),
        {
            "experiment": experiment,
            "dataset": dataset,
            "test_metrics": {key: value for key, value in test_metrics.items() if key != "report"},
            "alignment_state": state.to_dict(),
            "merger": merger.describe(),
        },
    )
    return 0


def _resolve_device(config: Any, torch: Any, logger: Any):
    cuda_devices = config.get_path("device.cuda_devices", "") or ""
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices).replace(",", " ")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(
            f"使用 GPU：{torch.cuda.device_count()} 张"
            f"（CUDA_VISIBLE_DEVICES={cuda_devices!r}）"
        )
    else:
        device = torch.device("cpu")
        logger.warning("未检测到可用 GPU，将在 CPU 上训练（不推荐）")
    return device


if __name__ == "__main__":
    sys.exit(main())

# coding=utf-8
"""评估入口：加载 checkpoint，在测试集上评测并导出指标与 t-SNE 图。

对应论文 Table 3–8 的指标与 Fg.7–Fg.10 的特征分布图。

用法::

    python scripts/evaluate.py --config configs/experiments/proposed-5.yaml \\
        --dataset twitter15 \\
        --checkpoint outputs/checkpoints/proposed-5_twitter15_joint/best.pt

    # 同时输出 t-SNE 图（默认开启，可用 --no-tsne 关闭）
    python scripts/evaluate.py --config configs/experiments/proposed-3.yaml \\
        --dataset twitter16 --checkpoint outputs/checkpoints/... --tsne-perplexity 40
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL 网络评测与特征可视化")
    parser.add_argument("--config", action="append", default=None, help="配置文件，可重复")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="评测 batch size")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试）")
    parser.add_argument("--output-dir", default=None, help="结果输出目录")
    parser.add_argument("--no-tsne", action="store_true", help="不绘制 t-SNE 图")
    parser.add_argument("--tsne-perplexity", type=float, default=None)
    parser.add_argument("--tsne-samples", type=int, default=0,
                        help="t-SNE 使用的样本数上限，0 表示全部")
    parser.add_argument("--name", default=None, help="结果里的方法名（默认取实验名）")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_paths = args.config or [os.path.join(REPO_ROOT, "configs", "base.yaml")]
    config = load_config(*config_paths)

    dataset = args.dataset or config.get_path("data.name", "twitter15")
    seed = args.seed if args.seed is not None else int(config.get_path("seed", 42))
    max_seq_length = args.max_seq_length or int(config.get_path("data.max_seq_length", 128))
    experiment = args.name or config.get_path("experiment.name", "eval")

    logger = get_logger(
        "evaluate",
        log_file=os.path.join(
            REPO_ROOT, config.get_path("paths.log_dir", "outputs/logs"),
            f"evaluate_{experiment}_{dataset}.log",
        ),
    )
    set_seed(seed, cuda_devices=config.get_path("device.cuda_devices", "") or None)

    import torch

    from data.dataset import PairDataset, build_dataloader, load_instances
    from src.models.cl_model import build_contrastive_model
    from src.models.tokenization import BertTextEncoder
    from src.training.evaluate import (
        evaluate_model,
        extract_features,
        format_report,
        save_predictions,
        save_report,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("未检测到 GPU，将在 CPU 上评测")

    processed_dir = os.path.join(
        REPO_ROOT, config.get_path("paths.processed_dir", "data/processed"), dataset
    )
    split_path = os.path.join(processed_dir, f"{args.split}.jsonl")
    if not os.path.isfile(split_path):
        logger.error(f"找不到 {split_path}，请先运行 scripts/prepare_data.py")
        return 1

    label_list = config.get_path("data.label_list", None) or ["NR", "FR", "TR", "UR"]
    text_mode = config.get_path("data.text_mode", "source_replies")
    tokenizer = BertTextEncoder(
        model_name=config.get_path("model.encoder.name", "bert-base-uncased"),
        local_dir=config.get_path("model.encoder.local_dir", "") or "",
        max_seq_length=max_seq_length,
        do_lower_case=bool(config.get_path("model.encoder.do_lower_case", True)),
    )

    instances = load_instances(split_path, limit=args.limit, logger=logger)
    dataset_obj = PairDataset(
        originals=instances,
        augmented=[],
        tokenizer=tokenizer,
        label_list=label_list,
        max_seq_length=max_seq_length,
        augmented_round=0,
        per_sample=0,
        text_mode=text_mode,
    )
    dataloader = build_dataloader(
        dataset_obj,
        batch_size=args.batch_size or int(config.get_path("training.cl.eval_batch_size", 64)),
        shuffle=False,
        paired=False,
    )

    model_cfg = config.get_path("model", {}) or {}
    model = build_contrastive_model(
        encoder_config=model_cfg.get("encoder", {}) or {},
        projector_config=model_cfg.get("projector", {}) or {},
        classifier_config=model_cfg.get("classifier", {}) or {},
        num_labels=len(label_list),
    ).to(device)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        logger.error(f"checkpoint 不存在：{checkpoint_path}")
        return 1
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"已加载 {checkpoint_path}"
        f"（epoch={payload.get('epoch', '?')}，missing={len(missing)}，unexpected={len(unexpected)}）"
    )
    model.eval()

    # ------------------------------------------------------------------ #
    # 指标
    # ------------------------------------------------------------------ #
    metrics = evaluate_model(model, dataloader, device, label_list=label_list)
    metrics["name"] = experiment
    logger.info(f"{args.split} 集结果：")
    logger.info("\n" + format_report(metrics, label_list))

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT,
        config.get_path("paths.result_dir", "outputs/results"),
        f"{experiment}_{dataset}",
    )
    ensure_dir(output_dir)
    save_report(
        os.path.join(output_dir, f"{args.split}_metrics.json"),
        metrics,
        extra={"dataset": dataset, "split": args.split, "checkpoint": checkpoint_path},
    )

    # ------------------------------------------------------------------ #
    # 预测明细
    # ------------------------------------------------------------------ #
    if bool(config.get_path("evaluation.save_predictions", True)):
        try:
            probabilities = _collect_probabilities(model, dataloader, device)
            predictions = probabilities.argmax(axis=1)
            references = _collect_labels(dataloader)
            uids = _collect_uids(dataloader)
            path = save_predictions(
                os.path.join(output_dir, f"{args.split}_predictions.jsonl"),
                uids=uids,
                predictions=predictions.tolist(),
                references=references.tolist(),
                probabilities=probabilities,
            )
            logger.info(f"预测明细已保存：{path}")
        except Exception as exc:  # pragma: no cover - 明细不是核心产物
            logger.warning(f"保存预测明细失败（已跳过）：{exc}")

    # ------------------------------------------------------------------ #
    # t-SNE
    # ------------------------------------------------------------------ #
    if not args.no_tsne and bool(config.get_path("visualization.tsne.enabled", True)):
        from src.training.visualize import (
            plot_feature_distribution,
            reduce_tsne_with_indices,
            save_tsne_coordinates,
        )

        features, labels, uids = extract_features(
            model, dataloader, device, use_projection=True, max_samples=args.tsne_samples
        )
        figure_dir = os.path.join(
            REPO_ROOT, config.get_path("paths.figure_dir", "outputs/figures")
        )
        figure_path = os.path.join(figure_dir, f"tsne_{experiment}_{dataset}_{args.split}.png")
        plot_perplexity = (
            args.tsne_perplexity
            if args.tsne_perplexity is not None
            else float(config.get_path("visualization.tsne.perplexity", 30.0))
        )
        plot_iterations = int(config.get_path("visualization.tsne.n_iter", 1000))
        plot_max_samples = int(config.get_path("visualization.tsne.sample_size", 0) or 0)

        # 先算坐标并拿回采样下标，再用同一组下标取标签——这样即便 t-SNE 做了下采样，
        # 图上每个点的颜色与 save_tsne_coordinates 里的 uid 也都是对的。
        coordinates, sampled_indices = reduce_tsne_with_indices(
            features,
            perplexity=plot_perplexity,
            n_iter=plot_iterations,
            seed=seed,
            max_samples=plot_max_samples,
        )
        plot_feature_distribution(
            features,
            labels,
            title=f"{experiment} on {dataset} ({args.split})",
            output_path=figure_path,
            perplexity=plot_perplexity,
            n_iter=plot_iterations,
            seed=seed,
            max_samples=plot_max_samples,
            label_list=label_list,
            coordinates=coordinates,
        )
        save_tsne_coordinates(
            os.path.join(output_dir, f"{args.split}_tsne.jsonl"),
            coordinates,
            labels[sampled_indices],
            uids=[uids[index] for index in sampled_indices] if uids else None,
            label_list=label_list,
        )
        logger.info(f"t-SNE 图已保存：{figure_path}")

    return 0


def _collect_probabilities(model: Any, dataloader: Any, device: Any):
    """收集全部样本的 softmax 概率 ``[N, C]``。"""
    import numpy as np
    import torch

    chunks: List[Any] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.get("original", batch)
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                token_type_ids=(
                    inputs["token_type_ids"].to(device)
                    if inputs.get("token_type_ids") is not None
                    else None
                ),
            )
            chunks.append(torch.softmax(outputs["logits"].float(), dim=-1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 4), dtype="float32")
    return np.concatenate(chunks, axis=0)


def _collect_labels(dataloader: Any):
    """收集全部标签。"""
    import numpy as np

    chunks = [batch["label"].detach().cpu().numpy() for batch in dataloader]
    if not chunks:
        return np.zeros((0,), dtype="int64")
    return np.concatenate(chunks, axis=0)


def _collect_uids(dataloader: Any) -> List[str]:
    """收集全部 uid。"""
    uids: List[str] = []
    for batch in dataloader:
        uids.extend([str(item) for item in (batch.get("uid") or [])])
    return uids


if __name__ == "__main__":
    sys.exit(main())

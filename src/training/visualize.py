# coding=utf-8
"""特征分布可视化：t-SNE 图（论文 Fg.7–Fg.10）。

论文的用法
----------
论文用 t-SNE 展示"不同增强轮次 / 不同微调轮次下四类样本的特征分布"，
并据此定性判断"类别边界是否更清晰、类内是否更聚集"。例如：

* Fg.7(a)(b)：Proposed-1 的 TR 特征点比 Proposed-2 更聚集；
* Fg.9(a)(b)(c)：微调后 FR 类聚集效应更强，但 NR 与 UR 出现交叉。

因此本模块的输出必须**同一坐标系下可比**，实现上固定三件事：

1. 使用投影特征 ``z``（对比学习空间）而不是句向量——论文分析的是 CL 的特征分布；
2. t-SNE 的 ``random_state`` 固定，保证同一配置多次运行图形一致；
3. 每次绘制都返回坐标数组，便于把多轮结果拼成一张对比图。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data.processors.data_model import LABELS
from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

__all__ = [
    "LABEL_COLORS",
    "reduce_tsne",
    "plot_feature_distribution",
    "plot_multiple_rounds",
    "save_tsne_coordinates",
]

#: 四类的固定配色（TR/NR/FR/UR）。固定配色是必要的：
#: 论文 Fg.7–Fg.10 需要跨图对比"同一类别的分布变化"，颜色不一致就没法看。
LABEL_COLORS: Dict[str, str] = {
    "NR": "#4C72B0",   # 蓝
    "FR": "#DD8452",   # 橙
    "TR": "#55A868",   # 绿
    "UR": "#C44E52",   # 红
}


def _require_sklearn():
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("t-SNE 可视化需要 scikit-learn，请先 pip install scikit-learn") from exc
    return TSNE


def reduce_tsne(
    features: Any,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
    max_samples: int = 0,
) -> Any:
    """对特征做 t-SNE 降维，返回 ``[N, 2]`` 数组。

    Args:
        features: ``[N, d]`` 的 numpy 数组。
        perplexity: t-SNE 的困惑度；会被自动裁剪到 ``< N``（sklearn 要求）。
        n_iter: 迭代次数。
        seed: 随机种子（固定以保证可复现）。
        max_samples: ``>0`` 时先随机下采样到该规模（t-SNE 复杂度为 O(N²)）。

    Returns:
        ``[N, 2]`` 的 numpy 数组。样本过少（<3）时返回零坐标。
    """
    import numpy as np

    TSNE = _require_sklearn()

    array = np.asarray(features, dtype="float32")
    if array.ndim != 2:
        raise ValueError(f"features 必须是二维数组，收到形状 {array.shape}")
    count = array.shape[0]
    if count < 3:
        return np.zeros((count, 2), dtype="float32")

    if max_samples and count > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(count, size=max_samples, replace=False)
        array = array[indices]

    # perplexity 必须小于样本数，按论文常用值 30 与 N/4 取小
    effective_perplexity = float(min(perplexity, max(5.0, (array.shape[0] - 1) / 3.0)))
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        n_iter=int(n_iter),
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(array)


def plot_feature_distribution(
    features: Any,
    labels: Any,
    title: str = "Feature distribution",
    output_path: Optional[str] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
    max_samples: int = 0,
    label_list: Optional[Sequence[str]] = None,
    coordinates: Optional[Any] = None,
    dpi: int = 200,
) -> Any:
    """绘制并保存 t-SNE 特征分布图（论文 Fg.7–Fg.10 的样式）。

    Args:
        features: ``[N, d]`` 特征。
        labels: ``[N]`` 标签下标。
        title: 图标题。
        output_path: 保存路径（``.png``）；``None`` 表示只绘图不保存。
        coordinates: 若已算过 t-SNE，可直接传入 ``[N, 2]`` 复用，避免重复计算。
        label_list: 类别名列表，默认 ``["NR","FR","TR","UR"]``。

    Returns:
        ``(fig, ax, coordinates)``，方便调用方进一步拼图。
    """
    import matplotlib

    matplotlib.use("Agg")  # 服务器无显示环境
    import matplotlib.pyplot as plt
    import numpy as np

    classes = list(label_list or LABELS)
    label_array = np.asarray(labels).reshape(-1)
    coords = (
        np.asarray(coordinates, dtype="float32")
        if coordinates is not None
        else reduce_tsne(features, perplexity=perplexity, n_iter=n_iter, seed=seed, max_samples=max_samples)
    )
    if coords.shape[0] != label_array.shape[0]:
        # 下采样过：同步裁剪标签
        label_array = label_array[: coords.shape[0]]

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(classes):
        mask = label_array == index
        if not mask.any():
            continue
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
            alpha=0.7,
            c=LABEL_COLORS.get(name, "#7F7F7F"),
            label=name,
            edgecolors="none",
        )
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("t-SNE dim 1")
    axis.set_ylabel("t-SNE dim 2")
    axis.legend(loc="best", fontsize=9, framealpha=0.9)
    axis.grid(alpha=0.15, linestyle="--")
    figure.tight_layout()

    if output_path:
        ensure_dir(os.path.dirname(os.path.abspath(output_path)))
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        get_logger("visualize").info(f"特征分布图已保存：{output_path}")

    return figure, axis, coords


def plot_multiple_rounds(
    panels: Sequence[Mapping[str, Any]],
    output_path: str,
    suptitle: str = "",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
    max_samples: int = 0,
    label_list: Optional[Sequence[str]] = None,
    dpi: int = 200,
) -> str:
    """把多轮结果拼成一张子图并排的对比图（对应论文 Fg.7/Fg.8/Fg.9/Fg.10）。

    Args:
        panels: 每个元素形如
            ``{"title": "Proposed-1", "features": array, "labels": array}``。
        output_path: 输出 png 路径。

    Returns:
        实际保存路径。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not panels:
        raise ValueError("panels 不能为空")

    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.6 * rows), squeeze=False)

    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        axis = axes[row][column]
        coords = reduce_tsne(
            panel["features"],
            perplexity=perplexity,
            n_iter=n_iter,
            seed=seed,
            max_samples=max_samples,
        )
        labels = np.asarray(panel["labels"]).reshape(-1)[: coords.shape[0]]
        classes = list(label_list or LABELS)
        for class_index, name in enumerate(classes):
            mask = labels == class_index
            if not mask.any():
                continue
            axis.scatter(
                coords[mask, 0], coords[mask, 1], s=10, alpha=0.7,
                c=LABEL_COLORS.get(name, "#7F7F7F"), label=name, edgecolors="none",
            )
        axis.set_title(str(panel.get("title", f"Panel {index + 1}")), fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])

    # 隐藏多余子图
    for index in range(len(panels), rows * columns):
        row, column = divmod(index, columns)
        axes[row][column].axis("off")

    handles, labels_ = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels_, loc="lower center", ncol=len(labels_), fontsize=10)
    if suptitle:
        figure.suptitle(suptitle, fontsize=13)
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    get_logger("visualize").info(f"多轮对比图已保存：{output_path}")
    return output_path


def save_tsne_coordinates(
    path: str,
    coordinates: Any,
    labels: Any,
    uids: Optional[Sequence[str]] = None,
    label_list: Optional[Sequence[str]] = None,
) -> str:
    """把 t-SNE 坐标落成 JSONL（便于后续用别的工具重绘或做量化分析）。"""
    import numpy as np

    from src.utils.io_utils import write_jsonl

    classes = list(label_list or LABELS)
    coord_array = np.asarray(coordinates)
    label_array = np.asarray(labels).reshape(-1)
    records: List[Dict[str, Any]] = []
    for index in range(min(coord_array.shape[0], label_array.shape[0])):
        class_index = int(label_array[index])
        records.append(
            {
                "uid": str(uids[index]) if uids and index < len(uids) else str(index),
                "label_id": class_index,
                "label": classes[class_index] if 0 <= class_index < len(classes) else "?",
                "x": float(coord_array[index, 0]),
                "y": float(coord_array[index, 1]),
            }
        )
    return write_jsonl(path, records)

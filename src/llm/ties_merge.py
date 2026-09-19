# coding=utf-8
"""TIES-Merging：论文 Algorithm 1 的实现（Algorithm 1: Merge(...)）。

论文原文把合并过程分成三步，本模块逐个函数对应：

1. **修剪（Trim）**：""在中，magnitude 的前 q% 被保留，其余被设置为 0""
   → :func:`trim_task_vector`
2. **选举符号（Elect Sign）**：""合并 τ1 和 τ2 的先决条件是解决不同向量之间的
   符号冲突问题""；第 e 个 entry 的符号由**加权**符号和决定
   （权重即论文式(6)的 ω，由 CL 损失决定）
   → :func:`elect_sign`
3. **基于权重的不相交合并（Weight-based Disjoint Merge）**：""对于 τ^m 中的第 e 个
   参数，我们仅保留来自模型且符号与聚合后选定符号一致的参数值""
   → :func:`disjoint_merge`

最后按 TIES-Merging 的标准形式合成模型::

    θ = θ_0 + λ · τ_merged

论文式(7)写成：:

    θ_m^n = (1 - α) · θ̃_m^n + α · θ_m^{n-1}

其中 ``θ̃`` 是本轮合并结果、``θ^{n-1}`` 是上一轮参数、``α`` 是合并超参数。
本模块通过 :func:`resolve_scaling` 把两者统一起来。设 τ 为本轮合并出的任务向量，
把两轮参数都写成"基座 + 缩放后的任务向量"（``θ̃ = θ_0 + λ_m·τ``、
``θ^{n-1} = θ_0 + λ_{m-1}·τ``），代入式(7) 得::

    θ_m^n = θ_0 + [(1-α)·λ_m + α·λ_{m-1}]·τ

因此实现上只需维护 λ（式(8)的动量更新）与基座 θ_0，无需保存上一轮完整参数，
写回时使用 ``scaling = (1-α)·λ_m + α·λ_{m-1}``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .task_vector import TaskVector

__all__ = [
    "TrimResult",
    "MergeReport",
    "trim_task_vector",
    "elect_sign",
    "disjoint_merge",
    "merge_task_vectors",
    "resolve_scaling",
    "TiesMerger",
]


# ---------------------------------------------------------------------- #
# 数据结构
# ---------------------------------------------------------------------- #
@dataclass
class TrimResult:
    """修剪结果：布尔掩码 + 被保留的比例。"""

    keep_mask: Any
    kept_ratio: float

    def __repr__(self) -> str:  # pragma: no cover
        return f"TrimResult(kept_ratio={self.kept_ratio:.4f})"


@dataclass
class MergeReport:
    """合并过程的诊断信息（写入日志/结果文件，便于核查 Algorithm 1 的行为）。

    注意两个容易混淆的比率：

    * :attr:`sparsity_after_trim` 是**修剪掉**的比例，约等于 ``1 - q/100``；
    * :attr:`trim_percent` 是**保留**的比例 ``q``。
    """

    num_vectors: int = 0
    num_parameters: int = 0
    total_entries: int = 0
    kept_entries: int = 0
    merged_entries: int = 0
    weights: List[float] = field(default_factory=list)
    trim_percent: float = 0.0
    scaling: float = 0.0
    rounds: List[int] = field(default_factory=list)

    @property
    def sparsity_after_trim(self) -> float:
        """修剪后被置 0 的元素占比（≈ ``1 - trim_percent/100``）。"""
        if not self.total_entries:
            return 0.0
        return 1.0 - (self.kept_entries / self.total_entries)

    @property
    def merge_rate(self) -> float:
        """被保留参数中最终写入合并结果的比例。"""
        if not self.kept_entries:
            return 0.0
        return self.merged_entries / self.kept_entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_vectors": self.num_vectors,
            "num_parameters": self.num_parameters,
            "total_entries": self.total_entries,
            "kept_entries": self.kept_entries,
            "merged_entries": self.merged_entries,
            "trim_percent": self.trim_percent,
            "sparsity_after_trim": round(self.sparsity_after_trim, 4),
            "merge_rate": round(self.merge_rate, 4),
            "weights": [round(value, 6) for value in self.weights],
            "rounds": list(self.rounds),
            "scaling": round(self.scaling, 6),
        }


# ---------------------------------------------------------------------- #
# 第 1 步：修剪
# ---------------------------------------------------------------------- #
def trim_task_vector(vector: Mapping[str, Any], trim_percent: float) -> Dict[str, Any]:
    """**Algorithm 1 第 3 行（Trim）**。

    ``trim_percent`` 的语义 = **保留**幅值最大的前 ``q%``，其余置 0。
    这与论文原文"magnitude 的前 q% 被保留，其余被设置为 0"以及
    TIES-Merging[35] 的 ``k = 20%``（keep top-20%）一致。

    Args:
        vector: ``{参数名: 张量}``。
        trim_percent: ``q``，**保留**百分比，取值 ``(0, 100]``。

    Returns:
        新的字典，键与形状与输入一致，只是被剪掉的元素置 0。

    Note:
        阈值按**整个向量的全局分位数**计算，而不是逐参数张量各算一次。
        TIES-Merging 原文也是全局层面比较幅值（同一个 entry 在不同任务向量间
        比较才有意义），逐张量分位数会让小张量的噪声被放大。

    Note:
        语义容易搞反，这里写明换算关系：``q=20`` 表示保留 20%、剪掉 80%
        （TIES 原文默认、也是论文的写法）；想让"剪掉 20%"就传 ``q=80``。
    """
    import torch

    if not vector:
        return {}
    if not 0 < trim_percent <= 100:
        raise ValueError(
            f"trim_percent 是**保留**百分比，须落在 (0, 100]，收到 {trim_percent}"
        )

    if trim_percent >= 100:
        return {name: value.clone() for name, value in vector.items()}

    magnitudes = torch.cat([value.detach().abs().reshape(-1).float() for value in vector.values()])
    if magnitudes.numel() == 0:
        return {name: value.clone() for name, value in vector.items()}

    # 保留幅值最大的 q%：分位点取 (1 - q/100)，保留"大于等于该分位点"的元素。
    # 例如 q=20 → 分位点 0.8 → 只留下幅值排在前 20% 的参数。
    quantile = 1.0 - float(trim_percent) / 100.0
    threshold = torch.quantile(magnitudes, quantile)
    trimmed: Dict[str, Any] = {}
    for name, value in vector.items():
        mask = value.detach().abs() >= threshold
        trimmed[name] = torch.where(
            mask, value, torch.zeros_like(value)
        )
    return trimmed


# ---------------------------------------------------------------------- #
# 第 2 步：选举符号
# ---------------------------------------------------------------------- #
def elect_sign(
    trimmed_vectors: Sequence[Mapping[str, Any]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """**Algorithm 1 第 8 行（Elect Sign）**：逐 entry 决定合并后的符号。

    论文式(6)给出的规则是"加权符号和"：

        sign_e = sign( Σ_m ω_m · τ_m[e] )

    与 TIES-Merging 原文的"绝对值和最大者胜"在**单调性**上等价
    （都是在比较各方向的支持强度），区别只在于论文额外引入了由 CL 损失决定的
    权重 ω_m；本实现严格按论文式(6)取加权和，权重缺省为 1 时退化为原文行为。

    Returns:
        ``{参数名: 符号张量}``，取值 ``{-1, 0, +1}``（0 表示所有向量在该 entry 上
        恰好抵消，此时该 entry 不参与合并）。
    """
    import torch

    if not trimmed_vectors:
        return {}
    if weights is None:
        weights = [1.0] * len(trimmed_vectors)
    if len(weights) != len(trimmed_vectors):
        raise ValueError(
            f"weights 长度 {len(weights)} 与向量数量 {len(trimmed_vectors)} 不一致"
        )

    names = list(trimmed_vectors[0].keys())
    signs: Dict[str, Any] = {}
    for name in names:
        accumulator = None
        for vector, weight in zip(trimmed_vectors, weights):
            term = vector[name].detach().float() * float(weight)
            accumulator = term if accumulator is None else accumulator + term
        signs[name] = torch.sign(accumulator)
    return signs


# ---------------------------------------------------------------------- #
# 第 3 步：基于权重的不相交合并
# ---------------------------------------------------------------------- #
def disjoint_merge(
    trimmed_vectors: Sequence[Mapping[str, Any]],
    signs: Mapping[str, Any],
    weights: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """**Algorithm 1 第 9-11 行（Weight-based Disjoint Merge）**。

    论文式：""对于 τ^m 中的第 e 个参数，我们仅保留来自模型且符号与聚合后选定符号
    一致的参数值""，即

        τ_merged[e] = ( Σ_{m: sign(τ_m[e]) == sign_e} ω_m · τ_m[e] )
                      / ( Σ_{m: sign(τ_m[e]) == sign_e} ω_m )

    也就是"符号一致者加权平均"，论文称之为 *weight-based disjoint merge*。

    Note:
        分母为 0（没有任何向量与该 entry 的选举符号一致）时输出 0。
        这在符号选举取加权和、而某个向量权重为负的极端情况下可能出现，
        用 ``eps`` 保护避免 NaN。
    """
    import torch

    if not trimmed_vectors:
        return {}
    if weights is None:
        weights = [1.0] * len(trimmed_vectors)

    names = list(trimmed_vectors[0].keys())
    merged: Dict[str, Any] = {}
    for name in names:
        sign = signs.get(name)
        if sign is None:
            merged[name] = torch.zeros_like(trimmed_vectors[0][name])
            continue

        numerator = None
        denominator = None
        for vector, weight in zip(trimmed_vectors, weights):
            value = vector[name].detach().float()
            agree = (torch.sign(value) == sign) & (sign != 0) & (value != 0)
            contribution = torch.where(agree, value * float(weight), torch.zeros_like(value))
            count = torch.where(
                agree, torch.full_like(value, float(weight)), torch.zeros_like(value)
            )
            numerator = contribution if numerator is None else numerator + contribution
            denominator = count if denominator is None else denominator + count

        merged[name] = (numerator / (denominator + eps)).to(trimmed_vectors[0][name].dtype)
    return merged


# ---------------------------------------------------------------------- #
# 组合：完整的 Algorithm 1
# ---------------------------------------------------------------------- #
def _count_entries(value: Any) -> Tuple[int, int]:
    """统计一个参数张量的 ``(总元素数, 非零元素数)``。

    读取失败时返回 ``(0, 0)``：**合并结果的正确性不依赖统计**，
    因此这里宁可让诊断数据缺一点，也不能因为统计把一个"合并已经算完"的流程炸掉。
    （这也让接口对"非张量但支持逐元素运算"的实现保持可用，便于单测用轻量替身。）
    """
    try:
        total = int(value.numel())
    except Exception:  # noqa: BLE001 - 统计失败不影响主流程
        return 0, 0
    try:
        nonzero = int((value != 0).sum().item())
    except Exception:  # noqa: BLE001
        nonzero = 0
    return total, nonzero


def merge_task_vectors(
    task_vector: TaskVector,
    trim_percent: float = 20.0,
    report: Optional[MergeReport] = None,
) -> Dict[str, Any]:
    """按 Algorithm 1 的 1→3 步顺序合并所有任务向量。

    Args:
        task_vector: :class:`src.llm.task_vector.TaskVector`（含权重 ω）。
        trim_percent: ``q``，修剪百分比。
        report: 可选的外部报告对象，用于回填诊断信息。

    Returns:
        合并后的任务向量 ``τ_merged``。
    """
    if task_vector.num_vectors == 0:
        return {}

    trimmed = [trim_task_vector(vector, trim_percent) for vector in task_vector.vectors]
    signs = elect_sign(trimmed, task_vector.weights)
    merged = disjoint_merge(trimmed, signs, task_vector.weights)

    if report is not None:
        report.num_vectors = task_vector.num_vectors
        report.num_parameters = len(merged)
        report.trim_percent = float(trim_percent)
        report.weights = list(task_vector.weights)
        report.rounds = list(task_vector.rounds)
        report.total_entries = 0
        report.kept_entries = 0
        report.merged_entries = 0
        for value in merged.values():
            total, nonzero = _count_entries(value)
            report.total_entries += total
            report.merged_entries += nonzero
        for vector in trimmed:
            for value in vector.values():
                report.kept_entries += _count_entries(value)[1]
    return merged


# ---------------------------------------------------------------------- #
# 缩放系数 λ（对应论文式(7)(8)(9)）
# ---------------------------------------------------------------------- #
def resolve_scaling(
    lambda_current: float,
    lambda_previous: float,
    alpha: Optional[float] = None,
) -> float:
    """计算写回模型时的缩放系数 λ（把式(7)与式(8)对起来）。

    论文式(8)给出 λ 的动量更新（``β`` 为影响因子）::

        λ_m = β · f_m + (1 - β) · λ_{m-1}

    论文式(7)给出最终参数::

        θ_m^n = (1 - α) · θ̃_m^n + α · θ_m^{n-1}

    其中 ``θ̃`` 是本轮合并得到的新参数、``θ^{n-1}`` 是上一轮参数、``α`` 是合并超参数。
    设 ``τ`` 是本轮合并出的任务向量（来自全部历史任务向量的 TIES 合并结果），
    并把两轮参数都写成"基座 + 缩放后的任务向量"：

        θ̃   = θ_0 + λ_m · τ
        θ^{n-1} = θ_0 + λ_{m-1} · τ

    代入式(7) 即得本函数返回的系数::

        scaling = (1 - α) · λ_m + α · λ_{m-1}

    含义很直观：**本轮新合并的任务向量取 ``(1-α)`` 的权重，历史累积取 ``α`` 的权重**。
    ``α = 0`` 时完全采用新合并结果（论文 Table 2 的 Proposed-4/5 若追求"每轮独立"可用），
    ``α = 1`` 时完全沿用历史（不更新），默认 ``α = 0.5`` 取折中。

    Args:
        lambda_current: 本轮 λ（式(8)更新后的值）。
        lambda_previous: 上一轮 λ；首轮为 ``lambda_init``。
        alpha: 合并超参数 α；``None`` 表示退化为标准 TIES-Merging 的
            ``θ = θ_0 + λ · τ``（即只用本轮 λ，不做历史插值）。

    Returns:
        实际写回模型时使用的缩放系数。
    """
    if alpha is None:
        return float(lambda_current)
    return float((1.0 - alpha) * float(lambda_current) + float(alpha) * float(lambda_previous))


# ---------------------------------------------------------------------- #
# 面向使用的封装
# ---------------------------------------------------------------------- #
class TiesMerger:
    """把 Algorithm 1 与式(7)(8)(9) 组装成一个可复用的合并器。

    Args:
        trim_percent: ``q``，修剪百分比（配置 ``llm.merge.trim_percent``）。
        alpha: 合并超参数 α（配置 ``llm.merge.scaling_alpha``）；
            ``None`` 表示使用纯 TIES-Merging 的缩放。
        lambda_init: ``λ_0``；默认 1.0（表示"从基座出发"）。
        merge_all_checkpoints: 为 True 时合并累计的全部任务向量，
            为 False 时只用最近一轮（消融对照）。
    """

    def __init__(
        self,
        trim_percent: float = 20.0,
        alpha: Optional[float] = 0.5,
        lambda_init: float = 1.0,
        merge_all_checkpoints: bool = True,
    ):
        self.trim_percent = float(trim_percent)
        self.alpha = None if alpha is None else float(alpha)
        self.lambda_init = float(lambda_init)
        self.lambda_previous = float(lambda_init)
        self.lambda_current = float(lambda_init)
        self.merge_all_checkpoints = bool(merge_all_checkpoints)
        self.task_vector = TaskVector()
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def add(
        self,
        vector: Mapping[str, Any],
        weight: float = 1.0,
        round_index: Optional[int] = None,
    ) -> None:
        """登记一轮微调产生的任务向量。"""
        self.task_vector.append(vector, weight=weight, round_index=round_index)

    # ------------------------------------------------------------------ #
    def merge(self) -> tuple:
        """执行合并。

        Returns:
            ``(τ_merged, MergeReport, scaling)``。``scaling`` 是写回模型时应使用的
            ``λ``（见 :func:`resolve_scaling`）。
        """
        if self.task_vector.num_vectors == 0:
            return {}, MergeReport(), 0.0

        source = self.task_vector
        if not self.merge_all_checkpoints and source.num_vectors > 1:
            source = source.subset([source.num_vectors - 1])
            # 只合并最近一轮时，历史插值项应当落在"同一轮"上，
            # 因此把 λ_previous 对齐到 λ_current，使
            # scaling = (1-α)·λ_m + α·λ_m = λ_m，即纯粹采用本轮结果
            self.lambda_previous = self.lambda_current

        report = MergeReport()
        merged = merge_task_vectors(source, trim_percent=self.trim_percent, report=report)
        scaling = resolve_scaling(
            self.lambda_current, self.lambda_previous, alpha=self.alpha
        )
        report.scaling = scaling
        self.history.append(report.to_dict())
        return merged, report, scaling

    # ------------------------------------------------------------------ #
    def update_lambda(self, score: float, beta: float = 0.9) -> float:
        """论文式(8)：``λ_m = β · f_m + (1 - β) · λ_{m-1}``。

        Args:
            score: ``f_m``，本轮的 CL 性能指标（已归一化到 ``[0, 1]``，
                见 :func:`src.training.joint_trainer.compute_lambda_score`）。
            beta: 影响因子 β。

        Returns:
            更新后的 ``λ_m``。
        """
        beta = float(beta)
        updated = beta * float(score) + (1.0 - beta) * self.lambda_current
        self.lambda_previous = self.lambda_current
        self.lambda_current = updated
        return updated

    # ------------------------------------------------------------------ #
    @staticmethod
    def report_note() -> str:
        """返回式(7)与本实现的对应说明（写进结果文件，便于复核）。"""
        return (
            "推导：设 τ 为本轮合并出的任务向量（由全部历史任务向量经 Algorithm 1 "
            "Trim→Elect Sign→Disjoint Merge 得到），并令 θ̃ = θ_0 + λ_m·τ、"
            "θ^{n-1} = θ_0 + λ_{m-1}·τ，代入论文式(7) "
            "θ_m^n = (1-α)·θ̃_m^n + α·θ_m^{n-1} 得 "
            "θ_m^n = θ_0 + [(1-α)·λ_m + α·λ_{m-1}]·τ，"
            "即写回模型时的 scaling = (1-α)·λ_m + α·λ_{m-1}。"
            "本实现只维护 λ（式(8)的动量更新）与基座 θ_0，不需要保存上一轮完整参数；"
            "alpha=None 时退化为标准 TIES-Merging 的 θ = θ_0 + λ·τ。"
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "trim_percent": self.trim_percent,
            "alpha": self.alpha,
            "lambda_init": self.lambda_init,
            "lambda_current": self.lambda_current,
            "lambda_previous": self.lambda_previous,
            "merge_all_checkpoints": self.merge_all_checkpoints,
            "num_vectors": self.task_vector.num_vectors,
            "weights": list(self.task_vector.weights),
            "rounds": list(self.task_vector.rounds),
            "history": list(self.history),
            "equivalence_note": self.report_note(),
        }

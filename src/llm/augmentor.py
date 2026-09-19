# coding=utf-8
"""LLM 数据增强流水线（论文 §3.1）。

一条增强样本的完整生命周期：

    DataInstance
      → PromptBuilder.build()       渲染 Prompt（只含 uid / string_value / replies）
      → 缓存查询                     命中则直接复用，保证"同一 Prompt 只调用一次 LLM"
      → LLMBackend.generate()       生成
      → parse_augmentation_response 抽 JSON + 基础字段校验
      → build_quality_report        结构/多样性/语义 指标
      → merge_augmentation          产出带 augmented 标记的新 DataInstance
      → JSONL 落盘 / 返回内存

失败处理：单条样本的失败**不会**中断整批。重试 ``max_retries`` 次后仍失败，
则返回"未增强副本"（``meta['augment_failed']=True``、``quality.problems`` 记录原因），
由上层统计失败率并决定是否丢弃——绝不静默吞掉。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data.processors.data_model import DataInstance

from .base import GenerationResult, LLMBackend
from .parser import QualityReport, build_quality_report, merge_augmentation, parse_augmentation_response
from .prompts import DEFAULT_TARGET_FIELD, PromptBuilder, PromptSpec
from src.utils.io_utils import ensure_dir, iter_jsonl, write_jsonl

__all__ = ["AugmentationStats", "Augmentor"]


class AugmentationStats:
    """一批增强任务的统计信息。"""

    def __init__(self) -> None:
        self.total = 0
        self.cached = 0
        self.succeeded = 0
        self.failed = 0
        self.retried = 0
        self.quality_flagged = 0
        self.elapsed_seconds = 0.0

    def summary(self) -> Dict[str, Any]:
        success_rate = (self.succeeded / self.total) if self.total else 0.0
        return {
            "total": self.total,
            "cached": self.cached,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "retried": self.retried,
            "quality_flagged": self.quality_flagged,
            "success_rate": round(success_rate, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"AugmentationStats({self.summary()})"


class Augmentor:
    """驱动一个 LLM 后端完成一轮数据增强。

    Args:
        backend: :class:`src.llm.base.LLMBackend` 实例。
        prompt_builder: :class:`src.llm.prompts.PromptBuilder` 实例。
        cache_dir: 增强缓存目录；``None`` 或空串表示禁用缓存。
        max_retries: 单条样本的重试次数（解析失败或生成失败都算）。
        copies_per_sample: 每条样本生成几份增强样本。
        strict_format: 是否把结构校验失败视为硬失败（True 时重试）。
        concurrency: 并发数；``1`` 表示顺序执行。
        semantic_encoder: 可选的语义相似度编码器（见 :func:`src.llm.parser.text_similarity`）。
        overlap_max: 词级重合度上限（约束 C3）。
        max_replies_in_prompt: 送入 Prompt 的最大回复条数。
        logger: 可选 logger。
    """

    def __init__(
        self,
        backend: LLMBackend,
        prompt_builder: PromptBuilder,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        copies_per_sample: int = 1,
        strict_format: bool = True,
        concurrency: int = 1,
        semantic_encoder: Optional[Any] = None,
        overlap_max: float = 0.75,
        temperature_jitter: float = 0.0,
        text_mode: str = "source_replies",
        max_seq_length: int = 128,
        logger: Optional[Any] = None,
    ):
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.cache_dir = cache_dir or ""
        self.max_retries = max(1, int(max_retries))
        self.copies_per_sample = max(1, int(copies_per_sample))
        self.strict_format = bool(strict_format)
        self.concurrency = max(1, int(concurrency))
        self.semantic_encoder = semantic_encoder
        self.overlap_max = float(overlap_max)
        self.temperature_jitter = float(temperature_jitter)
        self.text_mode = text_mode
        self.max_seq_length = max_seq_length
        self.logger = logger

        if self.cache_dir:
            ensure_dir(self.cache_dir)
        self._cache_lock = threading.Lock()
        self._memory_cache: Dict[str, str] = {}
        # 当前增强轮次：写入每个增强样本的 augment_round 字段
        self._current_round = 1

    # ------------------------------------------------------------------ #
    # 日志
    # ------------------------------------------------------------------ #
    def _log(self, message: str, level: str = "info") -> None:
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(message)

    # ------------------------------------------------------------------ #
    # 缓存
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(prompt_hash: str, copy_index: int, model_name: str) -> str:
        digest = hashlib.sha1(
            f"{prompt_hash}|{copy_index}|{model_name}".encode("utf-8")
        ).hexdigest()[:24]
        return digest

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _read_cache(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        with self._cache_lock:
            if key in self._memory_cache:
                return self._memory_cache[key]
        path = self._cache_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        text = payload.get("text")
        if isinstance(text, str):
            with self._cache_lock:
                self._memory_cache[key] = text
            return text
        return None

    def _write_cache(self, key: str, spec: PromptSpec, text: str) -> None:
        if not self.cache_dir:
            return
        with self._cache_lock:
            self._memory_cache[key] = text
        path = self._cache_path(key)
        payload = {
            "text": text,
            "uid": spec.meta.get("uid"),
            "prompt_hash": spec.prompt_hash,
            "model_name": self.backend.model_name,
            "backend": self.backend.name,
            "target_field": spec.target_field,
        }
        try:
            ensure_dir(self.cache_dir)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
        except OSError as exc:  # pragma: no cover - 磁盘问题
            self._log(f"写入增强缓存失败（不影响流程）：{exc}", level="warning")

    # ------------------------------------------------------------------ #
    # 单条样本
    # ------------------------------------------------------------------ #
    def _augment_one(self, instance: DataInstance, copy_index: int) -> Tuple[DataInstance, bool, bool]:
        """增强一条样本的一份副本。

        流程：渲染 Prompt → 查缓存 → 缺失则调用后端（失败/解析失败按
        ``max_retries`` 重试）→ 质量校验 → 合并成新实例。

        Returns:
            ``(增强后的实例, 是否命中缓存, 是否成功)``。
        """
        spec = self.prompt_builder.build(
            instance, temperature_hint=self.temperature_jitter * ((copy_index % 3) - 1)
        )
        cache_key = self._cache_key(spec.prompt_hash, copy_index, self.backend.model_name)

        cached_text = self._read_cache(cache_key)
        if cached_text is not None:
            return self._finalize(instance, cached_text, spec, from_cache=True), True, True

        last_problems: List[str] = []
        for _attempt in range(self.max_retries):
            results: List[GenerationResult] = self.backend.generate([spec])
            result = results[0] if results else GenerationResult(
                uid=instance.uid, prompt_hash=spec.prompt_hash, error="后端未返回结果"
            )
            if not result.ok:
                last_problems = [f"生成失败：{result.error}"]
                continue

            final = self._finalize(instance, result.text, spec, from_cache=False)
            quality = final.quality or {}
            if quality.get("structure_ok") and (
                not self.strict_format or quality.get("semantic_ok", True)
            ):
                # 只有通过校验的结果才写缓存，避免坏结果被永久复用
                self._write_cache(cache_key, spec, result.text)
                return final, False, True

            last_problems = list(quality.get("problems", [])) + list(quality.get("warnings", []))

        self._log(
            f"样本 {instance.uid} 第 {copy_index + 1} 份增强失败，已重试 {self.max_retries} 次："
            f"{last_problems[:2]}",
            level="warning",
        )
        return self._failed_copy(instance, spec, last_problems), False, False

    def _drop_cache(self, key: str) -> None:
        if not self.cache_dir:
            return
        with self._cache_lock:
            self._memory_cache.pop(key, None)
        path = self._cache_path(key)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:  # pragma: no cover
                pass

    def _finalize(
        self,
        instance: DataInstance,
        text: str,
        spec: PromptSpec,
        from_cache: bool = False,
    ) -> DataInstance:
        """把模型输出转成数据实例（解析 + 质量报告 + 合并）。"""
        report: Optional[QualityReport] = None
        payload: Optional[Mapping[str, Any]] = None
        try:
            payload = parse_augmentation_response(text)
            report = build_quality_report(
                instance,
                payload,
                encoder=self.semantic_encoder,
                overlap_max=self.overlap_max,
            )
        except ValueError as exc:
            report = QualityReport(structure_ok=False, problems=[f"解析失败：{exc}"])

        success = bool(report is not None and report.structure_ok and report.semantic_ok)
        augmented = merge_augmentation(
            instance,
            payload,
            report=report,
            augment_round=self._current_round,
            model_name=self.backend.model_name,
            prompt_hash=spec.prompt_hash,
            text_mode=self.text_mode,
            max_seq_length=self.max_seq_length,
            success=success,
        )
        augmented.meta["augment_cache_hit"] = bool(from_cache)
        return augmented

    def _failed_copy(
        self, instance: DataInstance, spec: PromptSpec, problems: Sequence[str]
    ) -> DataInstance:
        """产出"未增强副本"，保留原文本并在 quality 中记录失败原因。"""
        report = QualityReport(structure_ok=False, semantic_ok=False, problems=list(problems))
        augmented = merge_augmentation(
            instance,
            payload=None,
            report=report,
            augment_round=self._current_round,
            model_name=self.backend.model_name,
            prompt_hash=spec.prompt_hash,
            text_mode=self.text_mode,
            max_seq_length=self.max_seq_length,
            success=False,
        )
        return augmented

    # ------------------------------------------------------------------ #
    # 批量
    # ------------------------------------------------------------------ #
    def augment(
        self,
        instances: Sequence[DataInstance],
        augment_round: int = 1,
        progress_every: int = 50,
    ) -> Tuple[List[DataInstance], AugmentationStats]:
        """对一批样本做增强。

        Args:
            instances: 原样本列表。
            augment_round: 轮次编号（写入 ``augment_round`` 字段）。
            progress_every: 每处理多少条打印一次进度。

        Returns:
            ``(增强样本列表, 统计信息)``。返回列表长度 =
            ``len(instances) * copies_per_sample``（含失败副本）。
        """
        import time

        self._current_round = int(augment_round)
        stats = AugmentationStats()
        stats.total = len(instances) * self.copies_per_sample
        started = time.time()

        tasks: List[Tuple[DataInstance, int]] = [
            (instance, copy_index)
            for instance in instances
            for copy_index in range(self.copies_per_sample)
        ]

        results: List[DataInstance] = []
        with_cached: List[bool] = []

        if self.concurrency <= 1 or len(tasks) <= 1:
            for index, (instance, copy_index) in enumerate(tasks):
                augmented, cached, ok = self._augment_one(instance, copy_index)
                results.append(augmented)
                with_cached.append(cached)
                if ok:
                    stats.succeeded += 1
                else:
                    stats.failed += 1
                    stats.retried += self.max_retries
                if progress_every and (index + 1) % progress_every == 0:
                    self._log(f"增强进度 {index + 1}/{len(tasks)}")
        else:
            # 线程池只负责并发发请求；每个任务本身是独立的（无共享可变状态）
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._augment_one, instance, copy_index): (instance, copy_index)
                    for instance, copy_index in tasks
                }
                done = 0
                for future in as_completed(futures):
                    done += 1
                    try:
                        augmented, cached, ok = future.result()
                    except Exception as exc:  # pragma: no cover - 后端异常兜底
                        instance, copy_index = futures[future]
                        spec = self.prompt_builder.build(instance)
                        augmented = self._failed_copy(
                            instance, spec, [f"任务异常：{type(exc).__name__}: {exc}"]
                        )
                        cached, ok = False, False
                    results.append(augmented)
                    with_cached.append(cached)
                    if ok:
                        stats.succeeded += 1
                    else:
                        stats.failed += 1
                        stats.retried += self.max_retries
                    if progress_every and done % progress_every == 0:
                        self._log(f"增强进度 {done}/{len(tasks)}")

        stats.cached = sum(1 for flag in with_cached if flag)
        stats.quality_flagged = sum(
            1
            for item in results
            if item.quality and (item.quality.get("warnings") or item.quality.get("problems"))
        )
        stats.elapsed_seconds = time.time() - started

        summary = stats.summary()
        self._log(
            "增强完成："
            f"总数 {summary['total']}，成功 {summary['succeeded']}，失败 {summary['failed']}，"
            f"缓存命中 {summary['cached']}，质量告警 {summary['quality_flagged']}，"
            f"耗时 {summary['elapsed_seconds']}s，成功率 {summary['success_rate']:.2%}"
        )
        if stats.failed:
            self._log(
                "存在失败样本：它们以'未增强副本'形式保留（meta.augment_failed=true），"
                "训练时会被 PairDataset 按需过滤或当作同一样本使用，请注意统计口径",
                level="warning",
            )
        return results, stats


# ---------------------------------------------------------------------- #
# 工具函数
# ---------------------------------------------------------------------- #
def load_original_split(
    processed_dir: str,
    split: str = "train",
    limit: int = 0,
) -> List[DataInstance]:
    """从 ``data/processed/<dataset>/<split>.jsonl`` 读取原样本。

    放在本模块是为了让 ``scripts/augment_data.py`` 不必直接依赖 ``data.dataset``
    （后者需要 torch），从而让数据准备阶段可以完全脱离深度学习依赖运行。
    """
    from data.processors.data_model import record_to_instance

    path = os.path.join(processed_dir, f"{split}.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 {path}；请先运行 scripts/prepare_data.py 生成数据划分"
        )
    instances: List[DataInstance] = []
    for index, record in enumerate(iter_jsonl(path)):
        if limit and index >= limit:
            break
        instances.append(record_to_instance(record))
    return instances


def dump_augmented(path: str, instances: Iterable[DataInstance]) -> int:
    """把增强样本写入 JSONL，返回写入条数。"""
    return write_jsonl(path, (item.to_record() for item in instances))

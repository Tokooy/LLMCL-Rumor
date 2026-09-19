# coding=utf-8
"""Prompt 编排与增强输出校验的单元测试（论文 Fg.3 的四条设计目标）。

全部为纯文本逻辑，**不需要 torch**。
"""

from __future__ import annotations

import json
import os

import pytest


# ===================================================================== #
# Prompt 编排
# ===================================================================== #
class TestPromptBuilder:
    def test_template_placeholders_are_filled(self, prompt_builder, demo_instances):
        spec = prompt_builder.build(demo_instances[0])
        assert "{instance_json}" not in spec.user
        assert "{label_descriptions}" not in spec.user
        assert "{target_field}" not in spec.user
        assert demo_instances[0].string_value in spec.user
        assert "non-rumor" in spec.user

    def test_prompt_never_contains_label(self, prompt_builder, demo_instances):
        """标签必须外挂，不得进入 Prompt（否则增强会泄漏监督信号）。"""
        instance = demo_instances[0]
        spec = prompt_builder.build(instance)
        rendered = spec.as_prompt_text()
        # Prompt 里出现的 JSON 载荷不应含 label 字段
        assert '"label"' not in rendered
        assert '"label_id"' not in rendered
        # 但标签描述文本本身是可以出现的（用于解释哪些内容需保留）
        assert instance.uid in rendered

    def test_system_and_user_are_split(self, prompt_builder, demo_instances):
        spec = prompt_builder.build(demo_instances[0])
        assert spec.system, "应能从模板中切出 system message"
        assert "social-media data augmentation specialist" in spec.system
        messages = spec.as_messages()
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_prompt_hash_is_stable_and_content_dependent(self, prompt_builder, demo_instances):
        first = prompt_builder.build(demo_instances[0])
        second = prompt_builder.build(demo_instances[0])
        other = prompt_builder.build(demo_instances[1])
        assert first.prompt_hash == second.prompt_hash
        assert first.prompt_hash != other.prompt_hash

    def test_temperature_hint_changes_hash(self, prompt_builder, demo_instances):
        base = prompt_builder.build(demo_instances[0], temperature_hint=0.0)
        jittered = prompt_builder.build(demo_instances[0], temperature_hint=0.05)
        assert base.prompt_hash != jittered.prompt_hash

    def test_reply_truncation_in_prompt(self, prompt_builder, demo_instances):
        from src.llm.prompts import PromptBuilder

        instance = demo_instances[0]
        builder = PromptBuilder(
            template=prompt_builder.template,
            label_descriptions={"NR": "x"},
            max_replies_in_prompt=1,
        )
        spec = builder.build(instance)
        payload_start = spec.user.index("<instance>") + len("<instance>")
        payload_end = spec.user.index("</instance>")
        payload = json.loads(spec.user[payload_start:payload_end].strip())
        assert len(payload["replies"]) == 1
        assert payload["replies_truncated"] >= 1

    def test_missing_placeholder_raises(self):
        from src.llm.prompts import PromptBuilder

        with pytest.raises(ValueError, match="占位符"):
            PromptBuilder(template="no placeholders here", label_descriptions={})

    def test_missing_template_file_raises(self):
        from src.llm.prompts import load_prompt_template

        with pytest.raises(FileNotFoundError):
            load_prompt_template("configs/prompts/does_not_exist.txt")


# ===================================================================== #
# 输出解析
# ===================================================================== #
class TestResponseParsing:
    def test_plain_json(self):
        from src.llm.parser import parse_augmentation_response

        text = '{"uid": "u1", "string_value": "hello world", "replies": []}'
        payload = parse_augmentation_response(text)
        assert payload["uid"] == "u1"

    def test_json_inside_markdown_fence(self):
        from src.llm.parser import parse_augmentation_response

        text = '```json\n{"uid": "u1", "string_value": "hi", "replies": []}\n```'
        assert parse_augmentation_response(text)["string_value"] == "hi"

    def test_json_with_surrounding_prose(self):
        from src.llm.parser import parse_augmentation_response

        text = (
            "Sure, here is the rewritten instance:\n"
            '{"uid": "u1", "string_value": "hi", "replies": []}\n'
            "Let me know if you need more."
        )
        assert parse_augmentation_response(text)["uid"] == "u1"

    def test_thinking_tags_are_stripped(self):
        from src.llm.parser import parse_augmentation_response

        text = (
            " thinkingthe user wants a rewrite\u2026<｜end▁of▁thinking｜>"
            '{"uid": "u1", "string_value": "hi", "replies": []}'
        )
        assert parse_augmentation_response(text)["string_value"] == "hi"

    def test_qwen_style_think_block_is_stripped(self):
        """Qwen 的推理块闭合标签是 ``</think>``（不是 ``</think>``），必须能剥掉。

        回归测试：原实现的正则是 ``<{tag}>.*?</{tag}>``，对 Qwen 完全失效，
        只能靠括号扫描兜底——而思考过程里经常出现示例 JSON，
        扫描可能抽出**思考文本里的**那个对象。这里刻意在思考块里放一个诱饵 JSON。
        """
        from src.llm.parser import parse_augmentation_response

        text = (
            "Thinking Process:\n"
            "1. The user wants a rewrite.\n"
            "Example of the required shape: "
            '{"uid": "DECOY", "string_value": "I am a decoy", "replies": []}\n'
            "Now produce the answer.\n"
            "<｜end▁of▁thinking｜>"
            '{"uid": "u1", "string_value": "the real rewrite", "replies": []}'
        )
        payload = parse_augmentation_response(text)
        assert payload["uid"] == "u1", "抽到了思考块里的诱饵 JSON"
        assert payload["string_value"] == "the real rewrite"

    def test_think_block_with_standard_closing_tag_is_stripped(self):
        """DeepSeek 风格的 ``<｜end▁of▁thinking｜>`` 同样要能剥掉。"""
        from src.llm.parser import parse_augmentation_response

        text = (
            "reasoning here with a decoy "
            '{"uid": "DECOY", "string_value": "nope", "replies": []}'
            "<｜end▁of▁thinking｜>"
            '{"uid": "u2", "string_value": "kept", "replies": []}'
        )
        payload = parse_augmentation_response(text)
        assert payload["uid"] == "u2"
        assert payload["string_value"] == "kept"

    def test_harmony_style_analysis_block_is_stripped(self):
        """Harmony 风格 ``<|channel|>analysis<|message|>…<|end|>`` 也要能剥掉。"""
        from src.llm.parser import parse_augmentation_response

        text = (
            "<|channel|>analysis<|message|>let me think "
            '{"uid": "DECOY", "string_value": "nope", "replies": []}'
            "<|end|>"
            '{"uid": "u3", "string_value": "final", "replies": []}'
        )
        payload = parse_augmentation_response(text)
        assert payload["uid"] == "u3"

    def test_markdown_fence_and_reasoning_combined(self):
        from src.llm.parser import parse_augmentation_response

        text = (
            "Thinking Process:\nchecking the constraints\n"
            "<｜end▁of▁thinking｜>\n"
            "```json\n"
            '{"uid": "u4", "string_value": "both wrappers", "replies": []}\n'
            "```"
        )
        assert parse_augmentation_response(text)["uid"] == "u4"

    def test_nested_json_with_braces_in_strings(self):
        from src.llm.parser import extract_json_object

        text = 'prefix {"uid": "u1", "string_value": "a {b} c", "replies": []} suffix'
        payload = extract_json_object(text)
        assert payload["string_value"] == "a {b} c"

    def test_nested_replies_are_parsed(self):
        from src.llm.parser import parse_augmentation_response

        text = json.dumps(
            {
                "uid": "u1",
                "string_value": "source",
                "replies": [
                    {"uid": "r1", "string_value": "a", "replies": [
                        {"uid": "r2", "string_value": "b", "replies": []}
                    ]}
                ],
            }
        )
        payload = parse_augmentation_response(text)
        assert payload["replies"][0]["replies"][0]["uid"] == "r2"

    def test_missing_string_value_raises(self):
        from src.llm.parser import parse_augmentation_response

        with pytest.raises(ValueError, match="string_value"):
            parse_augmentation_response('{"uid": "u1", "replies": []}')

    def test_replies_must_be_list(self):
        from src.llm.parser import parse_augmentation_response

        with pytest.raises(ValueError, match="replies"):
            parse_augmentation_response('{"uid": "u1", "string_value": "x", "replies": {}}')

    def test_non_json_raises(self):
        from src.llm.parser import extract_json_object

        with pytest.raises(ValueError):
            extract_json_object("no json at all")


# ===================================================================== #
# 相似度指标
# ===================================================================== #
class TestSimilarityMetrics:
    def test_identical_texts_have_overlap_one(self):
        from src.llm.parser import char_ngram_overlap, word_overlap

        text = "the mayor secretly sold the public park"
        assert char_ngram_overlap(text, text) == pytest.approx(1.0)
        assert word_overlap(text, text) == pytest.approx(1.0)

    def test_rewritten_text_has_lower_overlap(self):
        from src.llm.parser import word_overlap

        source = "the mayor secretly sold the public park to a hotel chain"
        rewrite = "city hall quietly handed the public green space to hotel developers"
        assert word_overlap(source, rewrite) < 0.5

    def test_empty_inputs_are_handled(self):
        from src.llm.parser import char_ngram_overlap, word_overlap

        assert char_ngram_overlap("", "") == pytest.approx(1.0)
        assert char_ngram_overlap("abc", "") == pytest.approx(0.0)
        assert word_overlap("", "") == pytest.approx(1.0)

    def test_text_similarity_without_encoder_is_none(self):
        from src.llm.parser import text_similarity

        assert text_similarity("a", "b", encoder=None) is None

    def test_text_similarity_with_stub_encoder(self):
        from src.llm.parser import text_similarity

        class StubEncoder:
            def encode(self, texts):
                vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "a2": [1.0, 0.0]}
                return [vectors[text] for text in texts]

        assert text_similarity("a", "a2", encoder=StubEncoder()) == pytest.approx(1.0)
        assert text_similarity("a", "b", encoder=StubEncoder()) == pytest.approx(0.0)


# ===================================================================== #
# 结构校验与质量报告
# ===================================================================== #
class TestStructureValidation:
    def test_matching_structure_passes(self, demo_instances):
        from src.llm.parser import check_structure

        instance = demo_instances[0]
        payload = instance.prompt_payload()
        ok, problems = check_structure(instance, payload)
        assert ok, problems

    def test_changed_uid_fails(self, demo_instances):
        from src.llm.parser import check_structure

        instance = demo_instances[0]
        payload = dict(instance.prompt_payload())
        payload["uid"] = "hacked"
        ok, problems = check_structure(instance, payload)
        assert not ok
        assert any("uid" in problem for problem in problems)

    def test_reply_count_mismatch_fails(self, demo_instances):
        from src.llm.parser import check_structure

        instance = next(item for item in demo_instances if item.replies)
        payload = dict(instance.prompt_payload())
        payload["replies"] = payload["replies"][:1]
        ok, problems = check_structure(instance, payload)
        assert not ok
        assert any("回复数量" in problem for problem in problems)

    def test_missing_field_fails(self, demo_instances):
        from src.llm.parser import check_structure

        instance = demo_instances[0]
        payload = {"uid": instance.uid, "string_value": instance.string_value}
        ok, problems = check_structure(instance, payload)
        assert not ok
        assert any("缺少必需字段" in problem for problem in problems)

    def test_quality_report_flags_overlap(self, demo_instances):
        from src.llm.parser import build_quality_report

        instance = demo_instances[0]
        # 只改了几个字 → 词级重合度很高 → 多样性告警
        payload = instance.prompt_payload()
        payload["string_value"] = instance.string_value + " indeed"
        report = build_quality_report(instance, payload, overlap_max=0.5)
        assert report.diversity_ok is False
        assert any("约束 C3" in warning for warning in report.warnings)

    def test_quality_report_flags_length_ratio(self, demo_instances):
        from src.llm.parser import build_quality_report

        instance = demo_instances[0]
        payload = instance.prompt_payload()
        payload["string_value"] = "short"
        report = build_quality_report(instance, payload)
        # 严重摘要化 → 语义告警
        assert report.semantic_ok is False
        assert any("C4" in warning for warning in report.warnings)

    def test_to_dict_is_json_serializable(self, demo_instances):
        from src.llm.parser import build_quality_report

        instance = demo_instances[0]
        payload = instance.prompt_payload()
        report = build_quality_report(instance, payload)
        json.dumps(report.to_dict())  # 不抛异常即可


# ===================================================================== #
# 合并成增强实例
# ===================================================================== #
class TestMergeAugmentation:
    def test_label_is_inherited_from_original(self, demo_instances):
        from src.llm.parser import build_quality_report, merge_augmentation

        instance = demo_instances[0]
        payload = dict(instance.prompt_payload())
        payload["string_value"] = "a completely rephrased version of the same claim"
        report = build_quality_report(instance, payload)
        augmented = merge_augmentation(
            instance, payload, report=report, augment_round=2, model_name="stub"
        )
        assert augmented.label == instance.label
        assert augmented.label_id == instance.label_id
        assert augmented.uid == instance.uid
        assert augmented.original_uid == instance.uid
        assert augmented.augmented is True
        assert augmented.augment_round == 2
        assert augmented.quality is not None

    def test_failed_copy_keeps_original_text(self, demo_instances):
        from src.llm.parser import merge_augmentation

        instance = demo_instances[0]
        failed = merge_augmentation(instance, None, success=False, augment_round=1)
        assert failed.string_value == instance.string_value
        assert failed.meta.get("augment_failed") is True

    def test_text_is_regenerated_for_augmented(self, demo_instances):
        from src.llm.parser import build_quality_report, merge_augmentation

        instance = demo_instances[0]
        payload = dict(instance.prompt_payload())
        payload["string_value"] = "a fully rewritten source sentence with new wording"
        report = build_quality_report(instance, payload)
        augmented = merge_augmentation(instance, payload, report=report)
        assert augmented.text
        assert "a fully rewritten source sentence" in augmented.text

    def test_missing_reply_text_falls_back_to_original(self, demo_instances):
        from src.llm.parser import merge_augmentation

        instance = next(item for item in demo_instances if item.replies)
        payload = instance.prompt_payload()
        # 把第一条回复的正文清空，模拟模型漏写
        payload["replies"][0]["string_value"] = ""
        augmented = merge_augmentation(instance, payload)
        assert augmented.replies[0].string_value == instance.replies[0].string_value

    def test_reply_uid_kept(self, demo_instances):
        from src.llm.parser import merge_augmentation

        instance = next(item for item in demo_instances if item.replies)
        payload = instance.prompt_payload()
        augmented = merge_augmentation(instance, payload)
        assert augmented.replies[0].uid == instance.replies[0].uid


# ===================================================================== #
# 增强器装配（并发策略）
# ===================================================================== #
class TestAugmentorFactory:
    """``build_augmentor`` 的并发决策：不同后端应有不同的默认值。

    背景：``llm.augmentation.batch_size`` 对两种后端含义不同——
    api 后端是"并发请求数"，而 transformers 后端内部已经**分批**生成，
    再叠加线程池会让多个线程同时跑前向、各自持有 KV cache，单卡 24GB 极易 OOM。
    因此默认策略是：支持微调的后端（本地权重）→ 串行；其余 → 按 batch_size。
    但**显式传入的 concurrency 必须被尊重**（早期实现先 pop 再判断 key，
    条件恒为 False，显式覆盖被静默忽略）。
    """

    def test_local_backend_defaults_to_serial(self):
        from src.llm.factory import build_augmentor
        from src.utils.config import load_config

        config = load_config(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "base.yaml",
            )
        )
        augmentor = build_augmentor(
            config, overrides={"backend": "transformers", "max_retries": 1}
        )
        assert augmentor.concurrency == 1
        assert augmentor.backend.supports_finetuning is True

    def test_api_backend_keeps_configured_concurrency(self):
        from src.llm.factory import build_augmentor
        from src.utils.config import load_config

        config = load_config(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "base.yaml",
            )
        )
        augmentor = build_augmentor(config, overrides={"backend": "api", "max_retries": 1})
        assert augmentor.concurrency == int(config.get_path("llm.augmentation.batch_size", 4))

    def test_explicit_concurrency_is_respected(self):
        """显式覆盖必须生效——这正是回归测试要钉住的行为。"""
        from src.llm.factory import build_augmentor
        from src.utils.config import load_config

        config = load_config(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "base.yaml",
            )
        )
        augmentor = build_augmentor(
            config,
            overrides={"backend": "transformers", "concurrency": 8, "max_retries": 1},
        )
        assert augmentor.concurrency == 8

    def test_extra_body_empty_mapping_is_accepted(self):
        """`extra_body: {}`（空流式映射）必须能被装配流程接受。"""
        from src.llm.api_backend import APIBackend
        from src.llm.factory import build_backend
        from src.utils.config import load_config

        config = load_config(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "base.yaml",
            )
        )
        backend = build_backend(config, "api")
        assert isinstance(backend, APIBackend)
        assert backend.extra_body == {}


# ===================================================================== #
# demo 后端（端到端但不需要模型）
# ===================================================================== #
class TestDemoBackend:
    def test_generate_produces_valid_json(self, demo_instances, prompt_builder):
        from src.llm.demo_backend import DemoBackend

        backend = DemoBackend()
        specs = [prompt_builder.build(item) for item in demo_instances[:3]]
        results = backend.generate(specs)
        assert len(results) == 3
        for result, instance in zip(results, demo_instances[:3]):
            assert result.ok, result.error
            payload = json.loads(result.text)
            assert payload["uid"] == instance.uid
            assert payload["string_value"] != instance.string_value
            assert len(payload["replies"]) == len(instance.replies)

    def test_augmentor_end_to_end_without_models(self, demo_instances, prompt_builder, tmp_path):
        from src.llm.augmentor import Augmentor
        from src.llm.demo_backend import DemoBackend

        augmentor = Augmentor(
            backend=DemoBackend(),
            prompt_builder=prompt_builder,
            cache_dir=str(tmp_path / "cache"),
            max_retries=2,
            copies_per_sample=1,
            strict_format=False,   # 伪增强不满足多样性约束，测试里放宽
            concurrency=1,
        )
        augmented, stats = augmentor.augment(demo_instances, augment_round=1)
        assert len(augmented) == len(demo_instances)
        assert stats.total == len(demo_instances)
        assert stats.failed == 0
        assert all(item.augmented for item in augmented)
        assert all(item.augment_round == 1 for item in augmented)
        assert all(item.original_uid == item.uid for item in augmented)

    def test_cache_avoids_second_call(self, demo_instances, prompt_builder, tmp_path):
        from src.llm.augmentor import Augmentor
        from src.llm.demo_backend import DemoBackend

        cache_dir = str(tmp_path / "cache")
        augmentor = Augmentor(
            backend=DemoBackend(),
            prompt_builder=prompt_builder,
            cache_dir=cache_dir,
            strict_format=False,
        )
        augmentor.augment(demo_instances, augment_round=1)
        _second, stats = augmentor.augment(demo_instances, augment_round=1)
        assert stats.cached == len(demo_instances)

    def test_no_cache_means_no_reuse(self, demo_instances, prompt_builder):
        from src.llm.augmentor import Augmentor
        from src.llm.demo_backend import DemoBackend

        augmentor = Augmentor(
            backend=DemoBackend(),
            prompt_builder=prompt_builder,
            cache_dir=None,
            strict_format=False,
        )
        _first, stats_first = augmentor.augment(demo_instances, augment_round=1)
        _second, stats_second = augmentor.augment(demo_instances, augment_round=1)
        assert stats_first.cached == 0
        assert stats_second.cached == 0

    def test_failed_generation_returns_unqueued_copy(self, prompt_builder, demo_instances):
        """后端报错时必须返回"未增强副本"，而不是丢样本或抛异常。"""
        from src.llm.augmentor import Augmentor

        class BrokenBackend:
            name = "broken"
            model_name = "broken"
            supports_finetuning = False
            supports_task_vector = False

            def generate(self, prompts, temperature=None, **kwargs):
                from src.llm.base import GenerationResult

                return [
                    GenerationResult(
                        uid=str(spec.meta.get("uid", "")),
                        prompt_hash=spec.prompt_hash,
                        error="模拟失败",
                    )
                    for spec in prompts
                ]

        augmentor = Augmentor(
            backend=BrokenBackend(),
            prompt_builder=prompt_builder,
            cache_dir=None,
            max_retries=1,
        )
        augmented, stats = augmentor.augment(demo_instances[:2], augment_round=1)
        assert stats.failed == 2
        assert len(augmented) == 2
        assert all(item.meta.get("augment_failed") for item in augmented)
        assert augmented[0].string_value == demo_instances[0].string_value


# ===================================================================== #
# 自举微调样本构造
# ===================================================================== #
class TestFinetuneRecords:
    def test_records_use_same_prompt_as_augmentation(self, demo_instances, prompt_builder):
        from src.llm.lora import build_finetune_records

        augmented = demo_instances[:3]
        records = build_finetune_records(
            originals=demo_instances,
            augmented=augmented,
            prompt_builder=prompt_builder,
            target="original",
        )
        assert len(records) == 3
        for record in records:
            # completion 必须是原样本的规范化 JSON（不是增强结果）
            payload = json.loads(record["completion"])
            assert payload["uid"] == record["uid"]
            original = next(item for item in demo_instances if item.uid == record["uid"])
            assert payload["string_value"] == original.string_value
            # prompt 必须包含该样本本身
            assert original.string_value in record["prompt"]

    def test_empty_augmented_returns_empty(self, demo_instances, prompt_builder):
        from src.llm.lora import build_finetune_records

        assert build_finetune_records(demo_instances, [], prompt_builder) == []

    def test_max_records_truncates(self, demo_instances, prompt_builder):
        from src.llm.lora import build_finetune_records

        records = build_finetune_records(
            demo_instances, demo_instances, prompt_builder, max_records=2
        )
        assert len(records) == 2

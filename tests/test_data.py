# coding=utf-8
"""数据层单元测试：数据模型、回复树展平、Twitter15/16 解析、配对数据集。

不依赖 torch 的部分（数据模型、展平、解析）会正常执行；
``PairDataset`` 相关用例需要 torch，未安装时自动跳过。
"""

from __future__ import annotations

import json
import os

import pytest

from tests.conftest import require_torch


# ===================================================================== #
# 标签体系
# ===================================================================== #
class TestLabels:
    def test_paper_label_order(self):
        """论文 Table 1 的类别顺序：NR / FR / TR / UR。"""
        from data.processors.data_model import ID_TO_LABEL, LABEL_TO_ID, LABELS

        assert LABELS == ["NR", "FR", "TR", "UR"]
        assert LABEL_TO_ID["NR"] == 0
        assert LABEL_TO_ID["UR"] == 3
        assert ID_TO_LABEL[2] == "TR"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("non-rumor", "NR"),
            ("Non-Rumor", "NR"),
            ("false", "FR"),
            ("FALSE", "FR"),
            ("true", "TR"),
            ("unverified", "UR"),
            ("UR", "ur"),
            (0, "NR"),
            (3, "UR"),
            ("2", "TR"),
        ],
    )
    def test_alias_normalization(self, raw, expected):
        from data.processors.data_model import normalize_label

        assert normalize_label(raw) == expected.upper()

    def test_unknown_label_raises(self):
        from data.processors.data_model import normalize_label

        with pytest.raises(ValueError):
            normalize_label("maybe")

    def test_out_of_range_index_raises(self):
        from data.processors.data_model import normalize_label

        with pytest.raises(ValueError):
            normalize_label(7)


# ===================================================================== #
# 数据模型
# ===================================================================== #
class TestDataModel:
    def test_reply_tree_is_recursive(self):
        from data.processors.data_model import Reply

        reply = Reply.from_record(
            {
                "uid": "r1",
                "string_value": "a",
                "replies": [{"uid": "r2", "string_value": "b", "replies": []}],
            }
        )
        assert reply.replies[0].uid == "r2"

    def test_reply_accepts_alternative_text_keys(self):
        from data.processors.data_model import Reply

        assert Reply.from_record({"uid": "r", "text": "hi"}).string_value == "hi"
        assert Reply.from_record({"id": "r", "content": "hi"}).string_value == "hi"

    def test_count_replies_counts_nested(self, demo_instances):
        instance = next(item for item in demo_instances if item.reply_count > 2)
        assert instance.reply_count >= 3

    def test_prompt_payload_excludes_label(self, demo_instances):
        payload = demo_instances[0].prompt_payload()
        assert set(payload.keys()) == {"uid", "string_value", "replies"}
        assert "label" not in payload

    def test_prompt_hash_is_stable(self, demo_instances):
        assert demo_instances[0].prompt_hash() == demo_instances[0].prompt_hash()

    def test_roundtrip_through_record(self, demo_instances):
        from data.processors.data_model import record_to_instance

        for instance in demo_instances:
            restored = record_to_instance(instance.to_record())
            assert restored.uid == instance.uid
            assert restored.label == instance.label
            assert restored.string_value == instance.string_value
            assert restored.reply_count == instance.reply_count
            assert restored.text == instance.text

    def test_record_without_label_raises(self):
        """论文 Fg.2 的裸结构不含标签，必须显式报错而不是静默填 0。"""
        from data.processors.data_model import record_to_instance

        with pytest.raises(KeyError, match="label"):
            record_to_instance({"uid": "u", "string_value": "x", "replies": []})

    def test_record_without_uid_raises(self):
        from data.processors.data_model import record_to_instance

        with pytest.raises(KeyError, match="uid"):
            record_to_instance({"string_value": "x", "label": "NR"})


# ===================================================================== #
# 回复展平与编码器文本
# ===================================================================== #
class TestReplyFlatten:
    def test_bfs_order_is_level_by_level(self):
        from data.processors.data_model import Reply
        from data.processors.reply_flatten import flatten_replies

        replies = [
            Reply("r1", "one", [Reply("r1-1", "one-one", []), Reply("r1-2", "one-two", [])]),
            Reply("r2", "two", []),
        ]
        order = [segment.uid for segment in flatten_replies(replies, order="bfs")]
        assert order == ["r1", "r2", "r1-1", "r1-2"]

    def test_dfs_order_goes_deep_first(self):
        from data.processors.data_model import Reply
        from data.processors.reply_flatten import flatten_replies

        replies = [
            Reply("r1", "one", [Reply("r1-1", "one-one", [])]),
            Reply("r2", "two", []),
        ]
        order = [segment.uid for segment in flatten_replies(replies, order="dfs")]
        assert order == ["r1", "r1-1", "r2"]

    def test_depth_is_recorded(self):
        from data.processors.data_model import Reply
        from data.processors.reply_flatten import flatten_replies

        replies = [Reply("r1", "one", [Reply("r1-1", "deep", [])])]
        depths = {segment.uid: segment.depth for segment in flatten_replies(replies)}
        assert depths == {"r1": 1, "r1-1": 2}

    def test_max_replies_limits_output(self):
        from data.processors.data_model import Reply
        from data.processors.reply_flatten import flatten_replies

        replies = [Reply(f"r{i}", f"text {i}", []) for i in range(10)]
        assert len(flatten_replies(replies, max_replies=3)) == 3

    def test_empty_replies_are_dropped(self):
        from data.processors.data_model import Reply
        from data.processors.reply_flatten import flatten_replies

        replies = [Reply("r1", "   ", []), Reply("r2", "real", [])]
        assert [segment.uid for segment in flatten_replies(replies)] == ["r2"]

    def test_invalid_order_raises(self, demo_instances):
        from data.processors.reply_flatten import flatten_replies

        with pytest.raises(ValueError):
            flatten_replies(demo_instances[0].replies, order="random")


class TestEncoderText:
    def test_source_only_ignores_replies(self, demo_instances):
        from data.processors.reply_flatten import build_encoder_text

        instance = next(item for item in demo_instances if item.replies)
        text = build_encoder_text(instance, text_mode="source_only")
        assert text == instance.string_value

    def test_source_replies_joins_with_separator(self, demo_instances):
        from data.processors.reply_flatten import SEPARATOR, build_encoder_text

        instance = next(item for item in demo_instances if item.replies)
        text = build_encoder_text(instance, text_mode="source_replies")
        assert instance.string_value in text
        assert SEPARATOR in text
        assert instance.replies[0].string_value in text

    def test_hierarchical_marks_roles(self, demo_instances):
        from data.processors.reply_flatten import SOURCE_PREFIX, build_encoder_text

        instance = next(item for item in demo_instances if item.replies)
        text = build_encoder_text(instance, text_mode="hierarchical")
        assert SOURCE_PREFIX in text
        assert "Reply:" in text

    def test_source_always_survives_budget(self, demo_instances):
        """即使回复很多，原帖也不能被挤掉（预算分配的核心约束）。"""
        from data.processors.reply_flatten import build_encoder_text

        instance = next(item for item in demo_instances if item.replies)
        text = build_encoder_text(instance, text_mode="source_replies", max_seq_length=32)
        assert text.split()[0] == instance.string_value.split()[0]

    def test_invalid_mode_raises(self, demo_instances):
        from data.processors.reply_flatten import build_encoder_text

        with pytest.raises(ValueError, match="text_mode"):
            build_encoder_text(demo_instances[0], text_mode="nope")

    def test_attach_encoder_text_fills_all(self, demo_instances):
        from data.processors.reply_flatten import attach_encoder_text

        for instance in demo_instances:
            instance.text = ""
        attach_encoder_text(demo_instances, text_mode="source_replies")
        assert all(item.text for item in demo_instances)


# ===================================================================== #
# 原始数据解析
# ===================================================================== #
class TestTwitterParser:
    def test_parse_labels(self, tmp_path):
        from data.processors.twitter_rumor import parse_labels

        path = tmp_path / "label.txt"
        path.write_text(
            "123\ttrue\n456\tfalse\n789\tunverified\n000\tnon-rumor\n", encoding="utf-8"
        )
        labels = parse_labels(str(path))
        assert labels == {"123": "TR", "456": "FR", "789": "UR", "000": "NR"}

    def test_parse_labels_reports_bad_row(self, tmp_path):
        from data.processors.twitter_rumor import parse_labels

        path = tmp_path / "label.txt"
        path.write_text("123\tnot-a-label\n", encoding="utf-8")
        with pytest.raises(ValueError, match="1"):
            parse_labels(str(path))

    def test_parse_source_tweets_handles_python_literal(self, tmp_path):
        """Twitter15 官方文件用的是 python 字面量（单引号），必须能解析。"""
        from data.processors.twitter_rumor import parse_source_tweets

        path = tmp_path / "source_tweets.txt"
        path.write_text("123\t{'text': 'hello world'}\n", encoding="utf-8")
        assert parse_source_tweets(str(path)) == {"123": "hello world"}

    def test_parse_source_tweets_handles_json(self, tmp_path):
        from data.processors.twitter_rumor import parse_source_tweets

        path = tmp_path / "source_tweets.txt"
        path.write_text('123\t{"full_text": "hi there"}\n', encoding="utf-8")
        assert parse_source_tweets(str(path)) == {"123": "hi there"}

    def test_parse_source_tweets_handles_plain_text(self, tmp_path):
        from data.processors.twitter_rumor import parse_source_tweets

        path = tmp_path / "tweets.txt"
        path.write_text("123\tjust plain text\n", encoding="utf-8")
        assert parse_source_tweets(str(path)) == {"123": "just plain text"}

    def test_parse_trees(self, tmp_path):
        from data.processors.twitter_rumor import parse_trees

        path = tmp_path / "tree.txt"
        path.write_text("s1\ts1\tr1\ns1\tr1\tr2\ns2\ts2\tr3\n", encoding="utf-8")
        trees = parse_trees(str(path))
        assert trees["s1"] == [("s1", "r1"), ("r1", "r2")]
        assert trees["s2"] == [("s2", "r3")]

    def test_build_reply_forest(self):
        from data.processors.twitter_rumor import _build_reply_forest

        edges = [("s1", "r1"), ("r1", "r2"), ("s1", "r3")]
        texts = {"s1": "source", "r1": "reply1", "r2": "reply2", "r3": "reply3"}
        forest = _build_reply_forest(edges, texts, "s1")
        assert len(forest) == 2
        assert forest[0].uid == "r1"
        assert forest[0].replies[0].uid == "r2"
        assert forest[0].string_value == "reply1"

    def test_build_reply_forest_handles_cycle(self):
        """脏数据里的环不能让解析死循环。"""
        from data.processors.twitter_rumor import _build_reply_forest

        edges = [("s1", "r1"), ("r1", "r2"), ("r2", "r1")]
        texts = {"r1": "a", "r2": "b"}
        forest = _build_reply_forest(edges, texts, "s1")
        assert forest[0].uid == "r1"
        assert forest[0].replies[0].uid == "r2"
        assert forest[0].replies[0].replies == []

    def test_load_raw_dataset_official_layout(self, tmp_path):
        from data.processors.twitter_rumor import load_raw_dataset

        dataset_dir = tmp_path / "twitter15"
        dataset_dir.mkdir()
        (dataset_dir / "label.txt").write_text(
            "".join(
                f"s{i}\t{label}\n"
                for i, label in enumerate(
                    ["true"] * 5 + ["false"] * 5 + ["unverified"] * 5 + ["non-rumor"] * 5
                )
            ),
            encoding="utf-8",
        )
        (dataset_dir / "source_tweets.txt").write_text(
            "".join(f"s{i}\t{{'text': 'source tweet number {i}'}}\n" for i in range(20)),
            encoding="utf-8",
        )
        (dataset_dir / "tree.txt").write_text(
            "".join(f"s{i}\ts{i}\tr{i}\t\n".replace("\t\n", "\n") for i in range(20)),
            encoding="utf-8",
        )

        instances = load_raw_dataset(
            raw_dir=str(tmp_path), name="twitter15", text_mode="source_only"
        )
        assert len(instances) == 20
        counts = {"train": 0, "dev": 0, "test": 0}
        for instance in instances:
            counts[instance.split] += 1
        assert counts["train"] > counts["test"] > 0
        assert all(item.text for item in instances)

    def test_load_raw_dataset_requires_label_file(self, tmp_path):
        from data.processors.twitter_rumor import load_raw_dataset

        (tmp_path / "twitter15").mkdir()
        with pytest.raises(FileNotFoundError, match="label"):
            load_raw_dataset(raw_dir=str(tmp_path), name="twitter15")


# ===================================================================== #
# 划分
# ===================================================================== #
class TestSplit:
    def test_stratified_split_respects_ratios(self, demo_instances):
        from data.processors.twitter_rumor import assign_splits

        counts = assign_splits(demo_instances, 0.7, 0.1, 0.2, seed=42)
        assert sum(counts.values()) == len(demo_instances)
        assert all(value >= 0 for value in counts.values())

    def test_official_split_is_preferred(self, demo_instances):
        from data.processors.twitter_rumor import assign_splits

        official = {item.uid: "test" for item in demo_instances[:3]}
        counts = assign_splits(demo_instances, 0.7, 0.1, 0.2, seed=42, official=official)
        assert counts["test"] >= 3
        for instance in demo_instances[:3]:
            assert instance.split == "test"

    def test_ratios_must_sum_to_one(self, demo_instances):
        from data.processors.twitter_rumor import assign_splits

        with pytest.raises(ValueError, match="划分比例"):
            assign_splits(demo_instances, 0.7, 0.1, 0.5)

    def test_split_is_deterministic(self, demo_records):
        from data.processors.data_model import record_to_instance
        from data.processors.twitter_rumor import assign_splits

        first = [record_to_instance(record) for record in demo_records]
        second = [record_to_instance(record) for record in demo_records]
        assign_splits(first, seed=7)
        assign_splits(second, seed=7)
        assert [item.split for item in first] == [item.split for item in second]


# ===================================================================== #
# 配对数据集（需要 torch）
# ===================================================================== #
@pytest.mark.torch
class TestPairDataset:
    def test_dataset_requires_torch(self):
        require_torch()

    def test_pairs_original_with_its_augmentation(self, demo_instances, demo_records):
        torch = require_torch()
        from data.dataset import PairDataset
        from data.processors.data_model import record_to_instance

        originals = demo_instances
        augmented = []
        for record in demo_records:
            payload = dict(record)
            payload["string_value"] = "rewritten " + record["string_value"]
            payload["augmented"] = True
            payload["augment_round"] = 1
            payload["original_uid"] = record["uid"]
            augmented.append(record_to_instance(payload))

        dataset = PairDataset(
            originals=originals,
            augmented=augmented,
            tokenizer=None,          # 用字符级伪 id，测试不依赖 transformers
            max_seq_length=16,
            augmented_round=1,
            per_sample=1,
        )
        assert len(dataset) == len(originals)
        sample = dataset[0]
        assert sample["n_augmented"] == 1
        assert sample["original"]["input_ids"].shape == (16,)
        assert sample["augmented"][0]["input_ids"].shape == (16,)

    def test_require_augmented_filters_unpaired(self, demo_instances):
        require_torch()
        from data.dataset import PairDataset

        dataset = PairDataset(
            originals=demo_instances,
            augmented=[],
            tokenizer=None,
            max_seq_length=8,
            require_augmented=True,
        )
        assert len(dataset) == 0
        assert dataset.skipped_no_aug == len(demo_instances)

    def test_augmented_round_filter(self, demo_records):
        require_torch()
        from data.dataset import PairDataset
        from data.processors.data_model import record_to_instance

        originals = [record_to_instance(record) for record in demo_records]
        augmented = []
        for round_index in (1, 2):
            for record in demo_records:
                payload = dict(record)
                payload["string_value"] = f"round{round_index} " + record["string_value"]
                payload["augmented"] = True
                payload["augment_round"] = round_index
                payload["original_uid"] = record["uid"]
                augmented.append(record_to_instance(payload))

        dataset = PairDataset(
            originals=originals,
            augmented=augmented,
            tokenizer=None,
            max_seq_length=8,
            augmented_round=2,
        )
        assert dataset[0]["n_augmented"] == 1

        all_rounds = PairDataset(
            originals=originals,
            augmented=augmented,
            tokenizer=None,
            max_seq_length=8,
            augmented_round=0,       # 0 = 使用全部轮次
            per_sample=0,
        )
        assert all_rounds[0]["n_augmented"] == 2

    def test_collate_pairs_uniform_shape(self, demo_instances, demo_records):
        require_torch()
        from data.dataset import PairDataset, collate_pairs
        from data.processors.data_model import record_to_instance

        augmented = []
        for record in demo_records:
            payload = dict(record)
            payload["string_value"] = "rewritten " + record["string_value"]
            payload["augmented"] = True
            payload["augment_round"] = 1
            payload["original_uid"] = record["uid"]
            augmented.append(record_to_instance(payload))

        dataset = PairDataset(
            originals=demo_instances, augmented=augmented,
            tokenizer=None, max_seq_length=10, augmented_round=1,
        )
        batch = collate_pairs([dataset[0], dataset[1], dataset[2]])
        assert batch["original"]["input_ids"].shape == (3, 10)
        # 份数一致 → 返回 [B, K, L]
        assert batch["augmented"]["input_ids"].shape == (3, 1, 10)
        assert batch["label"].shape == (3,)

    def test_collate_pairs_ragged_returns_list(self, demo_records):
        require_torch()
        from data.dataset import PairDataset, collate_pairs
        from data.processors.data_model import record_to_instance

        originals = [record_to_instance(record) for record in demo_records]
        augmented = []
        for record in demo_records[:2]:      # 只有前两条有增强
            payload = dict(record)
            payload["string_value"] = "rewritten " + record["string_value"]
            payload["augmented"] = True
            payload["augment_round"] = 1
            payload["original_uid"] = record["uid"]
            augmented.append(record_to_instance(payload))

        dataset = PairDataset(
            originals=originals, augmented=augmented,
            tokenizer=None, max_seq_length=8, augmented_round=1,
        )
        batch = collate_pairs([dataset[0], dataset[1], dataset[2]])
        assert isinstance(batch["augmented"], list)
        assert len(batch["augmented"]) == 1
        # 只有前两条样本参与该层
        assert batch["augmented"][0]["input_ids"].shape[0] == 2
        # 关键：必须同时给出"第 0 层由哪些样本组成"的下标
        assert batch["augmented_indices"] == [[0, 1]]

    def test_collate_pairs_indices_are_not_a_prefix(self, demo_records):
        """回归测试：层成员在 batch 中**不是前缀**时，下标必须如实反映。

        构造 ``n_augmented = [0, 1, 1, 1]``：第 0 层只含第 1、2、3 个样本。
        若下游用 ``[:count]`` 切片（即取第 0、1、2 个），锚点就会和别人的增强样本配对。
        """
        require_torch()
        from data.dataset import PairDataset, collate_pairs
        from data.processors.data_model import record_to_instance

        originals = [record_to_instance(record) for record in demo_records[:4]]
        # 只给后三条样本配增强样本 → 第一条（index 0）没有
        augmented = []
        for record in demo_records[1:4]:
            payload = dict(record)
            payload["string_value"] = "rewritten " + record["string_value"]
            payload["augmented"] = True
            payload["augment_round"] = 1
            payload["original_uid"] = record["uid"]
            augmented.append(record_to_instance(payload))

        dataset = PairDataset(
            originals=originals, augmented=augmented,
            tokenizer=None, max_seq_length=8, augmented_round=1,
        )
        batch = collate_pairs([dataset[index] for index in range(4)])
        assert batch["n_augmented"] == [0, 1, 1, 1]
        assert batch["augmented_indices"] == [[1, 2, 3]], batch["augmented_indices"]
        # 层内样本数 == 下标个数
        assert batch["augmented"][0]["input_ids"].shape[0] == len(
            batch["augmented_indices"][0]
        )

    def test_collate_pairs_uniform_has_no_indices(self, demo_records):
        """份数一致时返回 [B,K,L] 张量，indices 为 None（天然按下标对齐）。"""
        require_torch()
        from data.dataset import PairDataset, collate_pairs
        from data.processors.data_model import record_to_instance

        originals = [record_to_instance(record) for record in demo_records[:3]]
        augmented = []
        for record in demo_records[:3]:
            payload = dict(record)
            payload["string_value"] = "rewritten " + record["string_value"]
            payload["augmented"] = True
            payload["augment_round"] = 1
            payload["original_uid"] = record["uid"]
            augmented.append(record_to_instance(payload))

        dataset = PairDataset(
            originals=originals, augmented=augmented,
            tokenizer=None, max_seq_length=8, augmented_round=1,
        )
        batch = collate_pairs([dataset[index] for index in range(3)])
        assert batch["augmented_indices"] is None
        assert batch["augmented"]["input_ids"].shape == (3, 1, 8)

    def test_group_by_uid(self, demo_instances, demo_records):
        require_torch()
        from data.dataset import group_by_uid
        from data.processors.data_model import record_to_instance

        augmented = []
        for record in demo_records[:3]:
            payload = dict(record)
            payload["string_value"] = "rewritten"
            payload["augmented"] = True
            payload["augment_round"] = 1
            payload["original_uid"] = record["uid"]
            augmented.append(record_to_instance(payload))

        groups = group_by_uid(list(demo_instances) + augmented)
        assert len(groups) == len(demo_instances)
        first = groups[demo_instances[0].uid]
        assert first[0].augmented is False
        assert first[1].augmented is True


# ===================================================================== #
# 演示数据自洽
# ===================================================================== #
class TestDemoData:
    def test_demo_file_matches_generator_records(self, demo_path):
        """已提交的 demo_twitter15.jsonl 必须与 make_demo.DEMO_RECORDS 一致。"""
        from data.samples.make_demo import DEMO_RECORDS, build_instance

        with open(demo_path, "r", encoding="utf-8") as handle:
            actual = [json.loads(line) for line in handle if line.strip()]
        expected = [build_instance(record).to_record() for record in DEMO_RECORDS]
        assert actual == expected

    def test_demo_covers_all_four_labels(self, demo_instances):
        from data.processors.data_model import LABELS

        seen = {item.label for item in demo_instances}
        assert seen == set(LABELS)

    def test_demo_has_edge_case_without_replies(self, demo_instances):
        assert any(item.reply_count == 0 for item in demo_instances)

    def test_demo_has_nested_replies(self, demo_instances):
        assert any(
            any(reply.replies for reply in item.replies) for item in demo_instances
        )

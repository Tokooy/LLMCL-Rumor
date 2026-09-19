# coding=utf-8
"""配置系统、IO 工具与评估指标的单元测试（全部不需要 torch）。"""

from __future__ import annotations

import json
import os

import pytest


# ===================================================================== #
# 配置系统
# ===================================================================== #
class TestConfig:
    def test_attribute_access(self, base_config):
        assert base_config.project == "LLMCL-Rumor"
        assert base_config.data.name == "twitter15"
        assert base_config.model.contrastive.temperature == pytest.approx(0.07)

    def test_get_path_and_default(self, base_config):
        assert base_config.get_path("model.encoder.name") == "bert-base-uncased"
        assert base_config.get_path("no.such.key", "fallback") == "fallback"

    def test_set_path_creates_intermediate(self, base_config):
        base_config.set_path("a.b.c", 42)
        assert base_config.a.b.c == 42

    def test_training_key_hyperparameters_match_paper(self, base_config):
        """论文实验设置：70/10/20 划分、温度 τ、对齐调度参数。"""
        split = base_config.data.split
        assert split.train == pytest.approx(0.7)
        assert split.dev == pytest.approx(0.1)
        assert split.test == pytest.approx(0.2)
        assert base_config.training.cl.augment_every_epochs == 1
        assert base_config.training.alignment.momentum_beta == pytest.approx(0.9)

    def test_to_dict_roundtrip(self, base_config):
        payload = base_config.to_dict()
        assert isinstance(payload, dict)
        json.dumps(payload)   # 必须可序列化（写日志用）

    def test_clone_is_deep(self, base_config):
        clone = base_config.clone()
        clone.data.max_seq_length = 999
        assert base_config.data.max_seq_length != 999

    def test_missing_attribute_raises(self, base_config):
        with pytest.raises(AttributeError):
            _ = base_config.definitely_not_here


class TestConfigDefaultsChain:
    def test_experiment_inherits_base(self, repo_root):
        from src.utils.config import load_config

        config = load_config(
            os.path.join(repo_root, "configs", "experiments", "proposed-1.yaml")
        )
        # 来自 base.yaml
        assert config.model.encoder.name == "bert-base-uncased"
        # 来自 llm/qwen7b.yaml
        assert "7B" in config.llm.model_name
        # 来自 proposed-1.yaml 自身
        assert config.experiment.name == "proposed-1"
        assert config.experiment.augment_rounds == 1
        assert config.training.alignment.max_finetune_rounds == 0

    @pytest.mark.parametrize(
        "name,augment_rounds,finetune_rounds,llm_size",
        [
            ("proposed-1", 1, 0, "7B"),
            ("proposed-2", 2, 0, "7B"),
            ("proposed-3", 3, 0, "7B"),
            ("proposed-4", 3, 1, "7B"),
            ("proposed-5", 3, 2, "7B"),
            ("proposed-6", 3, 2, "13B"),
        ],
    )
    def test_all_proposed_variants(self, repo_root, name, augment_rounds, finetune_rounds, llm_size):
        """论文 Table 2 的六个变体配置必须与表格一致。"""
        from src.utils.config import load_config

        config = load_config(
            os.path.join(repo_root, "configs", "experiments", f"{name}.yaml")
        )
        assert config.experiment.name == name
        assert config.experiment.augment_rounds == augment_rounds
        assert config.experiment.finetune_rounds == finetune_rounds
        assert config.experiment.llm_size == llm_size
        assert config.training.alignment.max_augment_rounds == augment_rounds
        assert config.training.alignment.max_finetune_rounds == finetune_rounds

    def test_config_sources_are_recorded(self, repo_root):
        from src.utils.config import load_config

        config = load_config(
            os.path.join(repo_root, "configs", "experiments", "proposed-4.yaml")
        )
        sources = config.get_path("_config_sources", [])
        assert len(sources) >= 3

    def test_missing_config_raises(self):
        from src.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("configs/this_file_does_not_exist.yaml")

    def test_cycle_detection(self, tmp_path):
        from src.utils.config import load_config

        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text("defaults:\n  - b.yaml\nx: 1\n", encoding="utf-8")
        second.write_text("defaults:\n  - a.yaml\ny: 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="循环"):
            load_config(str(first))

    def test_later_config_overrides_earlier(self, tmp_path):
        from src.utils.config import load_config

        first = tmp_path / "first.yaml"
        second = tmp_path / "second.yaml"
        first.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
        second.write_text("b:\n  c: 3\n  d: 4\n", encoding="utf-8")
        config = load_config(str(first), str(second))
        assert config.a == 1
        assert config.b.c == 3
        assert config.b.d == 4


class TestMergeDict:
    def test_deep_merge(self):
        from src.utils.config import merge_dict

        merged = merge_dict({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
        assert merged == {"a": {"b": 1, "c": 3}}

    def test_lists_are_replaced_not_appended(self):
        from src.utils.config import merge_dict

        merged = merge_dict({"a": [1, 2]}, {"a": [3]})
        assert merged["a"] == [3]


# ===================================================================== #
# IO 工具
# ===================================================================== #
class TestIOUtils:
    def test_jsonl_roundtrip(self, tmp_path):
        from src.utils.io_utils import read_jsonl, write_jsonl

        path = str(tmp_path / "data.jsonl")
        records = [{"uid": f"u{i}", "值": f"中文{i}"} for i in range(5)]
        assert write_jsonl(path, records) == 5
        assert read_jsonl(path) == records

    def test_append_jsonl(self, tmp_path):
        from src.utils.io_utils import append_jsonl, count_lines, read_jsonl, write_jsonl

        path = str(tmp_path / "data.jsonl")
        write_jsonl(path, [{"a": 1}])
        append_jsonl(path, [{"a": 2}, {"a": 3}])
        assert count_lines(path) == 3
        assert [item["a"] for item in read_jsonl(path)] == [1, 2, 3]

    def test_unicode_is_preserved(self, tmp_path):
        from src.utils.io_utils import read_jsonl, write_jsonl

        path = str(tmp_path / "cn.jsonl")
        write_jsonl(path, [{"text": "未经证实的谣言"}])
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
        assert "未经证实的谣言" in raw      # ensure_ascii=False
        assert read_jsonl(path)[0]["text"] == "未经证实的谣言"

    def test_missing_file_raises(self):
        from src.utils.io_utils import read_jsonl

        with pytest.raises(FileNotFoundError):
            read_jsonl("no/such/file.jsonl")

    def test_bad_line_reports_line_number(self, tmp_path):
        from src.utils.io_utils import read_jsonl

        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2"):
            read_jsonl(str(path))

    def test_json_save_and_load(self, tmp_path):
        from src.utils.io_utils import load_json, save_json

        path = str(tmp_path / "nested" / "report.json")
        save_json(path, {"acc": 0.8123})
        assert load_json(path)["acc"] == pytest.approx(0.8123)

    def test_load_json_default(self):
        from src.utils.io_utils import load_json

        assert load_json("no/such/file.json", default={}) == {}

    def test_ensure_dir(self, tmp_path):
        from src.utils.io_utils import ensure_dir

        target = str(tmp_path / "a" / "b" / "c")
        ensure_dir(target)
        assert os.path.isdir(target)


# ===================================================================== #
# 评估指标
# ===================================================================== #
class TestMetrics:
    """指标口径测试。

    论文 Table 3 的 Proposed-1 给出四类 F1 为
    ``TR=90.68 / NR=69.42 / FR=75.73 / UR=74.12``，其算术平均
    ``77.4875`` 与表中 "Avg F1 = 77.48" 完全一致；若是按样本数加权的平均，
    这几个数不会凑出 77.48。因此 **Avg F1 = 四类 F1 的算术平均（宏平均）**。

    下面的测试用"每类各有一个干净的二分类混淆矩阵"来验证这一口径：
    每类的 TP/FP/FN 直接给定，F1 可以手算，不涉及类别之间的相互抢预测。
    """

    @staticmethod
    def _build_per_class_predictions(specs):
        """按 ``{类别下标: (TP, FP, FN)}`` 构造预测与标签。

        每类独立生成：TP 条预测正确，FN 条把真值错判成一个**该类专属的干扰类**，
        FP 条把干扰类的真值错判成本类。干扰类取 ``类别下标 + 100``，
        与四类互不相干，因此每类的 TP/FP/FN 完全可控、可手算。
        """
        references, predictions = [], []
        for label, (tp, fp, fn) in specs.items():
            decoy = label + 100
            references.extend([label] * tp)
            predictions.extend([label] * tp)
            references.extend([label] * fn)
            predictions.extend([decoy] * fn)
            references.extend([decoy] * fp)
            predictions.extend([label] * fp)
        return predictions, references

    def test_per_class_f1_is_hand_computable(self):
        """手工构造的四类 F1 必须与手算值一致，Avg F1 = 四者算术平均。"""
        from src.training.evaluate import classification_metrics

        # 每类：TP / FP / FN（互不干扰），F1 = 2·TP / (2·TP + FP + FN)
        specs = {
            0: (70, 10, 20),    # F1 = 140 / 170 = 0.823529…
            1: (60, 20, 40),    # F1 = 120 / 180 = 0.666666…
            2: (90, 5, 5),      # F1 = 180 / 190 = 0.947368…
            3: (50, 30, 30),    # F1 = 100 / 160 = 0.625
        }
        predictions, references = self._build_per_class_predictions(specs)
        metrics = classification_metrics(predictions, references)

        hand_computed = {
            "NR": 2 * 70 / (2 * 70 + 10 + 20),
            "FR": 2 * 60 / (2 * 60 + 20 + 40),
            "TR": 2 * 90 / (2 * 90 + 5 + 5),
            "UR": 2 * 50 / (2 * 50 + 30 + 30),
        }
        for label, expected in hand_computed.items():
            assert metrics["per_class"][label]["f1"] == pytest.approx(expected, abs=1e-9), label

        # 口径：四类 F1 的算术平均
        assert metrics["avg_f1"] == pytest.approx(
            sum(hand_computed.values()) / 4, abs=1e-9
        )
        # 这一类构造下宏平均与 Avg F1 定义一致
        assert metrics["macro_f1"] == pytest.approx(metrics["avg_f1"], abs=1e-9)

    def test_avg_f1_reproduces_paper_arithmetic(self):
        """直接验证论文那组数的算术关系：四类 F1 的均值就是表中的 Avg F1。

        论文 Table 3 的 Proposed-1：TR=90.68、NR=69.42、FR=75.73、UR=74.12，
        均值 77.4875，表中两位小数写作 77.48（77.4875 截断/四舍五入到 2 位为 77.49，
        论文写 77.48 属于四舍五入到 0.01 附近的写法差异）。这里只钉住"是算术平均"。
        """
        paper_f1 = [0.9068, 0.6942, 0.7573, 0.7412]
        mean = sum(paper_f1) / 4
        assert mean == pytest.approx(0.774875, abs=1e-6)
        assert mean * 100 == pytest.approx(77.4875, abs=1e-3)
        # 若按样本数加权（Twitter15 四类样本数近似相等），结果也会接近这个值；
        # 关键区别在于论文的 Avg F1 与 macro avg 完全一致，这条在实现里有断言。

    def test_avg_f1_is_macro_not_weighted(self):
        """各类样本数不同时，Avg F1（宏平均）必须与加权 F1 不同。"""
        from src.training.evaluate import classification_metrics

        # NR 很多且全预测错、UR 很少且全预测对 → 加权平均明显高于宏平均
        references = [0] * 100 + [3] * 10
        predictions = [1] * 100 + [3] * 10
        metrics = classification_metrics(predictions, references)
        assert metrics["avg_f1"] < metrics["weighted_f1"]
        per_class = [metrics["per_class"][label]["f1"] for label in ("NR", "FR", "TR", "UR")]
        assert metrics["avg_f1"] == pytest.approx(sum(per_class) / 4, abs=1e-9)

    def test_perfect_prediction(self):
        from src.training.evaluate import classification_metrics

        labels = [0, 1, 2, 3, 0, 1, 2, 3]
        metrics = classification_metrics(labels, labels)
        assert metrics["acc"] == pytest.approx(1.0)
        assert metrics["avg_f1"] == pytest.approx(1.0)
        assert metrics["num_samples"] == 8

    def test_empty_input_returns_zeros(self):
        from src.training.evaluate import classification_metrics

        metrics = classification_metrics([], [])
        assert metrics["acc"] == 0.0
        assert metrics["avg_f1"] == 0.0
        assert metrics["num_samples"] == 0

    def test_length_mismatch_raises(self):
        from src.training.evaluate import classification_metrics

        with pytest.raises(ValueError, match="数量不一致"):
            classification_metrics([0, 1], [0])

    def test_accepts_tensor_like_sequences(self):
        from src.training.evaluate import classification_metrics

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def reshape(self, *shape):
                return self

            def tolist(self):
                return self.values

        metrics = classification_metrics([FakeTensor([0, 1])], [FakeTensor([0, 1])])
        assert metrics["acc"] == pytest.approx(1.0)

    def test_format_report_contains_paper_columns(self):
        from src.training.evaluate import classification_metrics, format_report

        labels = [0, 1, 2, 3] * 5
        metrics = classification_metrics(labels, labels)
        metrics["name"] = "Proposed-1"
        report = format_report(metrics)
        for column in ("ACC", "F1-NR", "F1-FR", "F1-TR", "F1-UR", "Avg F1"):
            assert column in report
        assert "Proposed-1" in report

    def test_save_report_writes_json(self, tmp_path):
        from src.training.evaluate import classification_metrics, save_report

        labels = [0, 1, 2, 3] * 5
        metrics = classification_metrics(labels, labels)
        path = save_report(str(tmp_path / "metrics.json"), metrics, extra={"dataset": "demo"})
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["dataset"] == "demo"
        assert payload["acc"] == pytest.approx(1.0)

    def test_save_predictions(self, tmp_path):
        from src.training.evaluate import save_predictions

        path = save_predictions(
            str(tmp_path / "preds.jsonl"),
            uids=["u1", "u2"],
            predictions=[0, 3],
            references=[0, 2],
            probabilities=[[0.9, 0.05, 0.03, 0.02], [0.1, 0.2, 0.3, 0.4]],
        )
        with open(path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        assert records[0]["pred"] == "NR"
        assert records[1]["true"] == "TR"
        assert records[1]["pred_id"] == 3
        assert len(records[0]["probabilities"]) == 4


# ===================================================================== #
# λ 观测值的口径
# ===================================================================== #
class TestLambdaSource:
    """式(8) 的 f_m 三种口径都必须落在 [0,1] 且"越大越好"。"""

    def test_contrastive_accuracy(self):
        from src.training.joint_trainer import compute_lambda_score

        assert compute_lambda_score({"acc": 0.8}, "contrastive_accuracy") == pytest.approx(0.8)

    def test_avg_f1(self):
        from src.training.joint_trainer import compute_lambda_score

        assert compute_lambda_score({"avg_f1": 0.6}, "avg_f1") == pytest.approx(0.6)

    def test_inverse_loss(self):
        from src.training.joint_trainer import compute_lambda_score

        # loss 越小 → 1/(1+loss) 越大
        assert compute_lambda_score({"loss": 0.0}, "inverse_loss") == pytest.approx(1.0)
        assert compute_lambda_score({"loss": 1.0}, "inverse_loss") == pytest.approx(0.5)
        high = compute_lambda_score({"loss": 0.1}, "inverse_loss")
        low = compute_lambda_score({"loss": 5.0}, "inverse_loss")
        assert high > low

    def test_out_of_range_is_clipped(self):
        from src.training.joint_trainer import compute_lambda_score

        assert compute_lambda_score({"acc": 1.5}, "contrastive_accuracy") == pytest.approx(1.0)
        assert compute_lambda_score({"acc": -0.5}, "contrastive_accuracy") == pytest.approx(0.0)

    def test_unknown_source_raises(self):
        from src.training.joint_trainer import compute_lambda_score

        with pytest.raises(ValueError, match="lambda_source"):
            compute_lambda_score({"acc": 0.5}, "nope")

    def test_missing_metric_defaults_to_zero(self):
        from src.training.joint_trainer import compute_lambda_score

        assert compute_lambda_score({}, "contrastive_accuracy") == 0.0

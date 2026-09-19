# coding=utf-8
"""端到端静态/数据流验证：不需要 torch，也不需要 GPU。

**它做什么**

在仓库自带演示数据上，把"论文数据流"里**不依赖 torch** 的部分真正跑一遍：

1. 读取 ``configs/experiments/proposed-*.yaml``（走完整的 defaults 继承链），
   校验六个变体的 w / M / 基座规模与论文 Table 2 一致；
2. 从 ``data/samples/demo_twitter15.jsonl`` 读入数据实例，检查
   四类标签齐全、回复树结构完整、编码器文本已生成；
3. 用默认 Prompt 模板渲染 Prompt，检查四条设计目标对应的约束都出现在 Prompt 里、
   且**标签没有泄漏进 Prompt**；
4. 走一遍 demo 后端的"伪增强"，检查增强结果的结构一致性、uid 一致性、
   回复数量一致性，以及质量报告能否产出；
5. 检查任务向量容器与式(7)(8)(9) 的标量链路；
6. 检查 Algorithm 1 中不依赖 torch 的部分（缩放系数推导）。

**它不做什么**

不加载 BERT / Qwen 权重、不做真实 LLM 增强、不训练、不评测——
涉及张量的部分由 ``pytest tests`` 覆盖（在装了 torch 的机器上运行）。

用法::

    python scripts/verify_pipeline.py

退出码：0 = 全部通过；1 = 有检查项失败。
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.config import load_config  # noqa: E402
from src.utils.io_utils import read_jsonl  # noqa: E402

DEMO_PATH = os.path.join(REPO_ROOT, "data", "samples", "demo_twitter15.jsonl")


class Checker:
    """极简的检查器：累计通过/失败/跳过项，最后统一报告。

    区分"失败"与"跳过"很重要：本机可能没装 pyyaml / sklearn / torch，
    这时相关检查**无法执行**，应当报 skipped 并说明原因，
    而不是伪装成失败（那会让真正的失败被噪声淹没），也不能伪装成通过。
    """

    def __init__(self) -> None:
        self.passed = 0
        self.skipped: List[Tuple[str, str]] = []
        self.failures: List[Tuple[str, str]] = []

    def check(
        self,
        name: str,
        function: Callable[[], None],
        requires: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        """执行一项检查。

        Args:
            name: 检查项名称。
            function: 无参可调用；失败时抛异常。
            requires: 可选的前置条件；返回非空字符串表示"无法执行"，
                该字符串作为跳过原因。
        """
        if requires is not None:
            reason = requires()
            if reason:
                self.skipped.append((name, reason))
                print(f"  [SKIP] {name}\n         原因：{reason}")
                return
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - 这里就是要抓住所有失败
            self.failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}\n         {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            print(f"  [ OK ] {name}")

    def report(self) -> int:
        print("-" * 72)
        total = self.passed + len(self.skipped) + len(self.failures)
        print(
            f"验证完成：通过 {self.passed}，跳过 {len(self.skipped)}，失败 {len(self.failures)}"
            f"（共 {total} 项）"
        )
        if self.skipped:
            print("跳过项（缺少可选依赖，装齐 requirements.txt 后会被执行）：")
            for name, reason in self.skipped:
                print(f"  - {name}: {reason}")
        if self.failures:
            print("失败项：")
            for name, reason in self.failures:
                print(f"  - {name}: {reason}")
            return 1
        return 0


def expect(condition: bool, message: str) -> None:
    """断言辅助：条件不成立时抛出带说明的异常。"""
    if not condition:
        raise AssertionError(message)


def _need(module_name: str, friendly: str = "") -> Callable[[], Optional[str]]:
    """构造一个"依赖缺失就跳过"的前置条件。"""

    def _check() -> Optional[str]:
        import importlib.util

        if importlib.util.find_spec(module_name) is None:
            return f"未安装 {friendly or module_name}（pip install {friendly or module_name}）"
        return None

    return _check


# ---------------------------------------------------------------------- #
# 1. 配置
# ---------------------------------------------------------------------- #
def check_configs(checker: Checker) -> None:
    print("[1/6] 配置继承链与论文 Table 2")

    expected = {
        "proposed-1": (1, 0, "7B"),
        "proposed-2": (2, 0, "7B"),
        "proposed-3": (3, 0, "7B"),
        "proposed-4": (3, 1, "7B"),
        "proposed-5": (3, 2, "7B"),
        "proposed-6": (3, 2, "13B"),
    }

    for name, (w, m, size) in expected.items():
        def _check(name=name, w=w, m=m, size=size) -> None:
            config = load_config(
                os.path.join(REPO_ROOT, "configs", "experiments", f"{name}.yaml")
            )
            expect(config.experiment.name == name, f"experiment.name 应为 {name}")
            expect(
                config.training.alignment.max_augment_rounds == w,
                f"w 应为 {w}，实际 {config.training.alignment.max_augment_rounds}",
            )
            expect(
                config.training.alignment.max_finetune_rounds == m,
                f"M 应为 {m}，实际 {config.training.alignment.max_finetune_rounds}",
            )
            expect(config.experiment.llm_size == size, f"基座应为 {size}")
            # 继承链必须真的生效（来自 base.yaml 的值）
            expect(config.model.encoder.name == "bert-base-uncased", "未继承 base 的编码器配置")
            expect(
                abs(float(config.data.split.train) - 0.7) < 1e-9,
                "未继承 base 的 70/10/20 划分",
            )
            sources = config.get_path("_config_sources", [])
            expect(len(sources) >= 3, f"继承链应至少 3 份配置，实际 {len(sources)}")

        checker.check(f"{name}.yaml 的 w={w} M={m} 基座={size}", _check)


def check_yaml_parser(checker: Checker) -> None:
    """用真实 pyyaml 交叉验证内置极简解析器（装 pyyaml 时才有意义）。

    这个检查保证"没有 pyyaml 时的回退实现"不会悄悄跑偏：
    两份实现解析同一批配置文件，结果必须**完全相等**。
    """
    print("[0/6] 内置极简 YAML 解析器 vs pyyaml 交叉验证")

    def _cross_validate() -> None:
        from src.utils.minimal_yaml import load_yaml, simple_yaml_load

        paths = [
            os.path.join(REPO_ROOT, "configs", "base.yaml"),
            os.path.join(REPO_ROOT, "configs", "llm", "qwen7b.yaml"),
            os.path.join(REPO_ROOT, "configs", "llm", "qwen13b.yaml"),
        ]
        paths += [
            os.path.join(REPO_ROOT, "configs", "experiments", f"proposed-{index}.yaml")
            for index in range(1, 7)
        ]
        for path in paths:
            official, parser_name = load_yaml(path)
            expect(parser_name == "pyyaml", "有 pyyaml 时不应回退到极简解析器")
            minimal = simple_yaml_load(path)
            if minimal != official:
                only_official = {k: v for k, v in official.items() if minimal.get(k) != v}
                only_minimal = {k: v for k, v in minimal.items() if official.get(k) != v}
                raise AssertionError(
                    f"{os.path.basename(path)} 两份解析器结果不一致：\n"
                    f"    仅 pyyaml 有/不同：{only_official}\n"
                    f"    仅极简解析有/不同：{only_minimal}"
                )

    checker.check(
        "13 份配置在两份解析器下结果一致",
        _cross_validate,
        requires=_need("yaml", "pyyaml"),
    )


# ---------------------------------------------------------------------- #
# 2. 数据
# ---------------------------------------------------------------------- #
def check_data(checker: Checker) -> None:
    print("[2/6] 演示数据的结构与标签")
    from data.processors.data_model import LABELS, record_to_instance
    from data.processors.reply_flatten import build_encoder_text

    def _file_exists() -> None:
        expect(os.path.isfile(DEMO_PATH), f"演示数据不存在：{DEMO_PATH}")

    checker.check("演示数据文件存在", _file_exists)

    records = read_jsonl(DEMO_PATH)
    instances = [record_to_instance(record) for record in records]

    def _labels_complete() -> None:
        seen = {item.label for item in instances}
        expect(seen == set(LABELS), f"四类标签应齐全，实际 {sorted(seen)}")

    checker.check("覆盖 NR/FR/TR/UR 四类", _labels_complete)

    def _structure() -> None:
        expect(any(item.reply_count == 0 for item in instances), "缺少'无回复'边界样本")
        expect(
            any(any(node.replies for node in item.replies) for item in instances),
            "缺少两层嵌套回复样本",
        )

    checker.check("包含无回复与嵌套回复两种边界", _structure)

    def _text_generated() -> None:
        for item in instances:
            expect(bool(item.text), f"{item.uid} 的编码器文本为空")
            expect(item.string_value in item.text, f"{item.uid} 的原帖未进入编码器文本")
        expect(
            all(build_encoder_text(item, "source_replies") == item.text for item in instances),
            "已落盘的 text 与按规则现算的结果不一致",
        )

    checker.check("编码器文本已生成且可复算", _text_generated)

    def _label_not_in_prompt_payload() -> None:
        for item in instances:
            payload = item.prompt_payload()
            expect(
                set(payload.keys()) == {"uid", "string_value", "replies"},
                f"{item.uid} 的 Prompt 载荷字段应为三件套",
            )

    checker.check("Prompt 载荷不含标签", _label_not_in_prompt_payload)


# ---------------------------------------------------------------------- #
# 3. Prompt
# ---------------------------------------------------------------------- #
def check_prompt(checker: Checker) -> None:
    print("[3/6] Prompt 编排与论文 Fg.3 的四条设计目标")
    from data.processors.data_model import record_to_instance
    from src.llm.prompts import PromptBuilder

    instances = [record_to_instance(record) for record in read_jsonl(DEMO_PATH)]
    builder = PromptBuilder(
        template_file=os.path.join(REPO_ROOT, "configs", "prompts", "augment_default.txt"),
        label_descriptions={"NR": "non-rumor", "FR": "false rumor", "TR": "true rumor", "UR": "unverified rumor"},
    )
    spec = builder.build(instances[0])
    rendered = spec.as_prompt_text()

    def _placeholders_filled() -> None:
        for placeholder in ("{instance_json}", "{label_descriptions}", "{target_field}"):
            expect(placeholder not in rendered, f"占位符 {placeholder} 未替换")

    checker.check("占位符全部替换", _placeholders_filled)

    def _four_goals_present() -> None:
        for marker in ("C1.", "C2.", "C3.", "C4."):
            expect(marker in rendered, f"Prompt 缺少约束 {marker}")

    checker.check("四条约束 C1–C4 都在 Prompt 中", _four_goals_present)

    def _no_label_leak() -> None:
        expect('"label"' not in rendered, "Prompt 中出现了 label 字段，监督信号会泄漏")
        expect('"label_id"' not in rendered, "Prompt 中出现了 label_id 字段")

    checker.check("标签未泄漏进 Prompt", _no_label_leak)

    def _roles_split() -> None:
        expect(bool(spec.system), "未切出 system message")
        expect(spec.as_messages()[0]["role"] == "system", "首条消息应为 system")

    checker.check("system / user 消息正确切分", _roles_split)


# ---------------------------------------------------------------------- #
# 4. 增强与结构校验（demo 后端）
# ---------------------------------------------------------------------- #
def check_augmentation(checker: Checker) -> None:
    print("[4/6] demo 后端的伪增强与质量校验")
    from data.processors.data_model import record_to_instance
    from src.llm.augmentor import Augmentor
    from src.llm.demo_backend import DemoBackend
    from src.llm.parser import check_structure
    from src.llm.prompts import PromptBuilder

    instances = [record_to_instance(record) for record in read_jsonl(DEMO_PATH)]
    builder = PromptBuilder(
        template_file=os.path.join(REPO_ROOT, "configs", "prompts", "augment_default.txt"),
        label_descriptions={"NR": "non-rumor", "FR": "false rumor", "TR": "true rumor", "UR": "unverified rumor"},
    )
    augmentor = Augmentor(
        backend=DemoBackend(),
        prompt_builder=builder,
        cache_dir=None,
        strict_format=False,   # 伪增强不满足多样性约束，这里只验证结构
        concurrency=1,
    )
    augmented, stats = augmentor.augment(instances, augment_round=1)

    def _count() -> None:
        expect(len(augmented) == len(instances), "增强样本数应与原样本数一致")
        expect(stats.failed == 0, f"不应有失败样本，实际 {stats.failed}")

    checker.check("每条样本都产出增强副本", _count)

    def _pairing_fields() -> None:
        for item in augmented:
            expect(item.augmented, "augmented 标记缺失")
            expect(item.augment_round == 1, "augment_round 应为 1")
            expect(item.uid == item.original_uid, "uid 必须与原样本一致以便配对")

    checker.check("配对字段（augmented / augment_round / original_uid）正确", _pairing_fields)

    def _label_inherited() -> None:
        by_uid = {item.uid: item for item in instances}
        for item in augmented:
            original = by_uid[item.uid]
            expect(item.label == original.label, "增强样本必须沿用原样本标签")

    checker.check("标签沿用原样本", _label_inherited)

    def _structure_ok() -> None:
        by_uid = {item.uid: item for item in instances}
        payload = {
            "uid": augmented[0].uid,
            "string_value": augmented[0].string_value,
            "replies": [reply.to_record() for reply in augmented[0].replies],
        }
        ok, problems = check_structure(by_uid[augmented[0].uid], payload)
        expect(ok, f"结构校验应通过，实际问题：{problems}")

    checker.check("结构校验通过（uid/字段/回复数量一致）", _structure_ok)

    def _quality_report() -> None:
        for item in augmented:
            expect(item.quality is not None, f"{item.uid} 缺少 quality 报告")
            expect("word_overlap" in item.quality, "quality 缺少 word_overlap")

    checker.check("质量报告已生成", _quality_report)


# ---------------------------------------------------------------------- #
# 5. 任务向量与 λ / ω / scaling
# ---------------------------------------------------------------------- #
def check_merge_scalars(checker: Checker) -> None:
    print("[5/6] 式(7)(8)(9) 与任务向量容器")
    from src.llm.task_vector import TaskVector, subtract_parameters
    from src.llm.ties_merge import TiesMerger, resolve_scaling
    from src.training.joint_trainer import compute_lambda_score, compute_omega

    def _task_vector_container() -> None:
        # 用标量验证算子语义（不依赖 torch）：τ = θ_ft − θ_0
        base = {"a": 1.0, "b": 2.0}
        finetuned = {"a": 2.0, "b": 1.0}
        vector = subtract_parameters(finetuned, base)
        expect(vector == {"a": 1.0, "b": -1.0}, f"τ 计算错误：{vector}")

        container = TaskVector([{"a": 1.0}], weights=[0.3], rounds=[1])
        container.append({"a": 2.0}, weight=0.7)
        expect(container.weights == [0.3, 0.7], "ω 记录错误")
        expect(container.rounds == [1, 2], "轮次记录错误")
        expect(container.latest() == {"a": 2.0}, "latest() 应返回最近一轮")
        subset = container.subset([1])
        expect(subset.weights == [0.7] and subset.rounds == [2], "subset 未保留元信息")

    checker.check("任务向量 τ = θ_ft − θ_0 与 ω/轮次记录", _task_vector_container)

    def _container_rejects_mismatch() -> None:
        from src.llm.task_vector import TaskVector

        try:
            TaskVector([{"a": 1.0}, {"b": 1.0}])
        except ValueError:
            return
        raise AssertionError("参数集合不一致时应抛 ValueError")

    checker.check("任务向量容器拒绝参数集合不一致", _container_rejects_mismatch)

    def _momentum() -> None:
        merger = TiesMerger(lambda_init=1.0)
        lam = merger.update_lambda(0.8, beta=0.9)
        expect(abs(lam - 0.82) < 1e-9, f"式(8) 应为 0.82，实际 {lam}")
        expect(abs(merger.lambda_previous - 1.0) < 1e-9, "λ_previous 应保留上一轮值")

    checker.check("式(8) λ 动量更新", _momentum)

    def _omega() -> None:
        expect(abs(compute_omega(0.0, 0.05, 0.95) - 0.05) < 1e-12, "ω 下界失效")
        expect(abs(compute_omega(1.0, 0.05, 0.95) - 0.95) < 1e-12, "ω 上界失效")
        expect(abs(compute_omega(0.42, 0.05, 0.95) - 0.42) < 1e-12, "ω 未按 λ 取值")

    checker.check("式(9) ω 取值与截断", _omega)

    def _scaling() -> None:
        # scaling = (1-α)·λ_m + α·λ_{m-1}
        expect(abs(resolve_scaling(0.8, 1.0, 0.5) - 0.9) < 1e-12, "式(7) 推导结果错误")
        expect(abs(resolve_scaling(0.8, 1.0, None) - 0.8) < 1e-12, "alpha=None 应退化为 λ_m")
        expect(abs(resolve_scaling(0.63, 0.63, 0.5) - 0.63) < 1e-12, "λ 相等时应恒等")

    checker.check("式(7) scaling 推导（含退化情形）", _scaling)

    def _lambda_sources() -> None:
        expect(abs(compute_lambda_score({"acc": 0.8}, "contrastive_accuracy") - 0.8) < 1e-12, "acc 口径错误")
        expect(abs(compute_lambda_score({"loss": 1.0}, "inverse_loss") - 0.5) < 1e-12, "inverse_loss 口径错误")
        expect(compute_lambda_score({"acc": 1.6}, "contrastive_accuracy") == 1.0, "未做上界裁剪")

    checker.check("λ 的三种口径都归一化到 [0,1]", _lambda_sources)


# ---------------------------------------------------------------------- #
# 6. 指标口径
# ---------------------------------------------------------------------- #
def check_metrics(checker: Checker) -> None:
    print("[6/6] 评估指标（论文 Table 3 的 Avg F1 口径）")
    from src.training.evaluate import classification_metrics, format_report

    def _avg_f1_is_macro() -> None:
        # 构造四类，使逐类 F1 互不相同；Avg F1 必须是四类 F1 的算术平均
        references = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100
        predictions = (
            [0] * 90 + [3] * 10
            + [1] * 70 + [0] * 30
            + [2] * 60 + [1] * 40
            + [3] * 50 + [2] * 50
        )
        metrics = classification_metrics(predictions, references)
        per_class = [metrics["per_class"][label]["f1"] for label in ("NR", "FR", "TR", "UR")]
        expected = sum(per_class) / 4
        expect(
            abs(metrics["avg_f1"] - expected) < 1e-9,
            f"Avg F1 应为四类 F1 的算术平均 {expected:.6f}，实际 {metrics['avg_f1']:.6f}",
        )
        # 与加权平均必须不同（否则说明口径搞混了）
        expect(
            abs(metrics["avg_f1"] - metrics["weighted_f1"]) > 1e-6
            or abs(metrics["avg_f1"] - metrics["macro_f1"]) < 1e-9,
            "Avg F1 既不等于宏平均，需要人工确认口径",
        )

    checker.check("Avg F1 = 四类 F1 算术平均", _avg_f1_is_macro, requires=_need("sklearn", "scikit-learn"))

    def _perfect_prediction() -> None:
        labels = [0, 1, 2, 3] * 5
        metrics = classification_metrics(labels, labels)
        expect(abs(metrics["acc"] - 1.0) < 1e-12, "完全预测正确时 acc 应为 1")
        expect(abs(metrics["avg_f1"] - 1.0) < 1e-12, "完全预测正确时 avg_f1 应为 1")

    checker.check("完全正确时 ACC/AvgF1 = 1", _perfect_prediction, requires=_need("sklearn", "scikit-learn"))

    def _report_columns() -> None:
        labels = [0, 1, 2, 3] * 5
        metrics = classification_metrics(labels, labels)
        metrics["name"] = "Proposed-1"
        report = format_report(metrics)
        for column in ("ACC", "F1-NR", "F1-FR", "F1-TR", "F1-UR", "Avg F1"):
            expect(column in report, f"报告缺少列 {column}")

    checker.check(
        "报告列与论文 Table 3–8 同构", _report_columns, requires=_need("sklearn", "scikit-learn")
    )


def main() -> int:
    print("=" * 72)
    print("LLMCL-Rumor 数据流验证（不加载模型、不训练、不需要 GPU）")
    print("=" * 72)
    # 统一日志出口：无 pyyaml 时每个被读的配置文件都会发一条警告，
    # 这里让它们走 logger 而不是裸 print，输出才不会淹没检查结果。
    from src.utils.logger import setup_logging

    setup_logging()
    checker = Checker()
    check_yaml_parser(checker)
    check_configs(checker)
    check_data(checker)
    check_prompt(checker)
    check_augmentation(checker)
    check_merge_scalars(checker)
    check_metrics(checker)
    return checker.report()


if __name__ == "__main__":
    sys.exit(main())

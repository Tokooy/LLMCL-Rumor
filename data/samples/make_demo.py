# coding=utf-8
"""演示数据工具：生成/校验 ``data/samples/demo_twitter15.jsonl``。

``demo_twitter15.jsonl`` 是**手工构造的 10 条样例**（不是真实推文，也不摘自任何
真实账号），已随仓库提交。它的作用是在没有 Twitter15/16 数据、甚至没有 LLM
的情况下，把"数据解析 → 数据加载 → 对比学习配对 → 训练 → 评测"整条链路跑通。

本模块提供两个能力：

* :data:`DEMO_RECORDS` —— 样例的原始结构（与会话中的推导一致）；
* :func:`fake_paraphrase` / :func:`fake_augment` —— **伪增强**（机械式改写）。
  它只用于验证数据流与配对逻辑，**绝不代表论文的 LLM 增强效果**；
  真实增强必须调用 LLM，见 ``scripts/augment_data.py --backend demo|transformers|api``。

``scripts/augment_data.py --backend demo`` 会调用这里的伪增强，因此不需要 GPU。

直接运行本模块会重新生成 ``demo_twitter15.jsonl``（幂等，可用于校验手工版本）::

    python data/samples/make_demo.py --check   # 只校验，不写文件
    python data/samples/make_demo.py           # 重新生成
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.processors.data_model import DataInstance, Reply  # noqa: E402
from data.processors.reply_flatten import build_encoder_text  # noqa: E402

DEMO_FILENAME = "demo_twitter15.jsonl"

# ---------------------------------------------------------------------- #
# 10 条演示样本：覆盖 4 个类别、覆盖 1~2 层回复、覆盖"无回复"的边界情况
# ---------------------------------------------------------------------- #
DEMO_RECORDS: List[Dict[str, Any]] = [
    {
        "uid": "demo-0001",
        "label": "FR",
        "string_value": "an open letter to city voters from his top strategist-turned-defector",
        "replies": [
            {"uid": "demo-0001-r1", "string_value": "They love him", "replies": []},
            {
                "uid": "demo-0001-r2",
                "string_value": "She obviously didn't look at all the havoc he caused. He destroyed the local team!",
                "replies": [
                    {
                        "uid": "demo-0001-r2-1",
                        "string_value": "Exactly, nobody checks the record before posting",
                        "replies": [],
                    }
                ],
            },
        ],
    },
    {
        "uid": "demo-0002",
        "label": "TR",
        "string_value": "City council approves new transit line after three years of public consultation",
        "replies": [
            {"uid": "demo-0002-r1", "string_value": "Finally, this will cut my commute in half", "replies": []},
            {"uid": "demo-0002-r2", "string_value": "About time the budget was spent on something useful", "replies": []},
        ],
    },
    {
        "uid": "demo-0003",
        "label": "UR",
        "string_value": "Unconfirmed reports say the factory will close next month, workers say they were not told",
        "replies": [
            {"uid": "demo-0003-r1", "string_value": "My cousin works there and heard nothing about it", "replies": []},
            {
                "uid": "demo-0003-r2",
                "string_value": "Big if true, but nobody has shown a document yet",
                "replies": [
                    {
                        "uid": "demo-0003-r2-1",
                        "string_value": "Right, let's wait for an official statement",
                        "replies": [],
                    }
                ],
            },
        ],
    },
    {
        "uid": "demo-0004",
        "label": "NR",
        "string_value": "Local library extends weekend opening hours starting next Monday",
        "replies": [],
    },
    {
        "uid": "demo-0005",
        "label": "FR",
        "string_value": "The mayor secretly sold the public park to a hotel chain, according to an anonymous blog",
        "replies": [
            {"uid": "demo-0005-r1", "string_value": "That blog also said the moon landing was staged", "replies": []},
            {"uid": "demo-0005-r2", "string_value": "Do you have any proof at all?", "replies": []},
            {"uid": "demo-0005-r3", "string_value": "People will believe anything with a scary headline", "replies": []},
        ],
    },
    {
        "uid": "demo-0006",
        "label": "TR",
        "string_value": "Hospital confirms the new wing opens in April, hiring 120 nurses",
        "replies": [
            {"uid": "demo-0006-r1", "string_value": "Great news for the region", "replies": []},
            {"uid": "demo-0006-r2", "string_value": "Hopefully the waiting lists actually shrink", "replies": []},
        ],
    },
    {
        "uid": "demo-0007",
        "label": "UR",
        "string_value": "A video claims the bridge is unsafe, engineers have not yet responded",
        "replies": [
            {
                "uid": "demo-0007-r1",
                "string_value": "The video is blurry and cut in the middle",
                "replies": [
                    {
                        "uid": "demo-0007-r1-1",
                        "string_value": "That usually means the full clip says the opposite",
                        "replies": [],
                    },
                    {
                        "uid": "demo-0007-r1-2",
                        "string_value": "Or it means nothing at all, we can't tell",
                        "replies": [],
                    },
                ],
            }
        ],
    },
    {
        "uid": "demo-0008",
        "label": "NR",
        "string_value": "University publishes its annual research report, applications up nine percent",
        "replies": [
            {"uid": "demo-0008-r1", "string_value": "The report is public, I read it yesterday", "replies": []}
        ],
    },
    {
        "uid": "demo-0009",
        "label": "FR",
        "string_value": "Screenshot 'proves' the election result was changed, experts say the image is edited",
        "replies": [
            {"uid": "demo-0009-r1", "string_value": "You can literally see the font difference", "replies": []},
            {"uid": "demo-0009-r2", "string_value": "Stop spreading this, it has been debunked twice", "replies": []},
        ],
    },
    {
        "uid": "demo-0010",
        "label": "TR",
        "string_value": "Fire department reports the warehouse fire is contained, no injuries",
        "replies": [
            {"uid": "demo-0010-r1", "string_value": "Thank goodness, the smoke was visible from my street", "replies": []},
            {
                "uid": "demo-0010-r2",
                "string_value": "Crews are still on site as a precaution",
                "replies": [
                    {
                        "uid": "demo-0010-r2-1",
                        "string_value": "Good to hear they are staying until it is fully out",
                        "replies": [],
                    }
                ],
            },
        ],
    },
]

# ---------------------------------------------------------------------- #
# 伪增强（仅用于打通数据流）
# ---------------------------------------------------------------------- #
_REWRITE_HINTS = [
    ("an open letter", "a public letter"),
    ("They love him", "They're fully committed to him"),
    ("didn't look at", "clearly ignored"),
    ("havoc", "damage"),
    ("destroyed", "ruined"),
    ("approves", "signs off on"),
    ("Finally,", "At last,"),
    ("Unconfirmed reports say", "Reports that have not been confirmed suggest"),
    ("Big if true", "Significant if accurate"),
    ("extends weekend opening hours", "will open for longer on weekends"),
    ("secretly sold", "quietly handed over"),
    ("People will believe anything", "Some will accept any claim that"),
    ("confirms", "has confirmed"),
    ("Great news", "Excellent news"),
    ("A video claims", "A clip circulating online alleges"),
    ("usually means", "often indicates"),
    ("publishes", "has released"),
    ("Screenshot 'proves'", "A screenshot purporting to show"),
    ("has been debunked", "has already been disproved"),
    ("reports", "has announced"),
    ("Thank goodness", "Thank heavens"),
]


def fake_paraphrase(text: str) -> str:
    """机械替换，模拟"语义一致但表达不同"的增强结果。

    Warning:
        这不是论文的 LLM 增强，只是让数据流可以在无 GPU 环境下被验证。
    """
    result = text
    for source, target in _REWRITE_HINTS:
        if source in result:
            result = result.replace(source, target)
    if result == text:
        lowered = text[0].lower() + text[1:] if text else text
        result = f"According to the post, {lowered}"
    return result


def fake_augment(node: Dict[str, Any]) -> Dict[str, Any]:
    """递归地对一个节点（含所有子回复）做伪增强，结构保持不变。"""
    return {
        "uid": node["uid"],
        "string_value": fake_paraphrase(node["string_value"]),
        "replies": [fake_augment(child) for child in node.get("replies", [])],
    }


def to_reply(node: Dict[str, Any]) -> Reply:
    """递归把裸 dict 转成 :class:`Reply`，保持嵌套层级不变。"""
    return Reply(
        uid=node["uid"],
        string_value=node["string_value"],
        replies=[to_reply(child) for child in node.get("replies", [])],
    )


def build_instance(record: Dict[str, Any], **overrides: Any) -> DataInstance:
    """把一条裸记录转成 :class:`DataInstance` 并回填编码器文本。"""
    payload = {
        "uid": record["uid"],
        "string_value": record["string_value"],
        "label": record["label"],
        "replies": [to_reply(node) for node in record.get("replies", [])],
        "dataset": "demo",
    }
    payload.update(overrides)
    instance = DataInstance(**payload)
    instance.text = build_encoder_text(
        instance, text_mode="source_replies", max_seq_length=128
    )
    return instance


def build_instances(records: Sequence[Dict[str, Any]]) -> List[DataInstance]:
    """批量转换。"""
    return [build_instance(record) for record in records]


def build_demo_instances(text_mode: str = "source_replies", max_seq_length: int = 128) -> List[DataInstance]:
    """按指定 text_mode 构造全部演示实例（供 augment 脚本复用）。"""
    instances = []
    for record in DEMO_RECORDS:
        instance = DataInstance(
            uid=record["uid"],
            string_value=record["string_value"],
            label=record["label"],
            replies=[to_reply(node) for node in record.get("replies", [])],
            dataset="demo",
        )
        instance.text = build_encoder_text(
            instance, text_mode=text_mode, max_seq_length=max_seq_length
        )
        instances.append(instance)
    return instances


# ---------------------------------------------------------------------- #
# 生成 / 校验
# ---------------------------------------------------------------------- #
def dump_records(path: str) -> int:
    """把 :data:`DEMO_RECORDS` 写成 JSONL，返回条数。"""
    records = [build_instance(record).to_record() for record in DEMO_RECORDS]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def check_records(path: str) -> bool:
    """校验已提交的 JSONL 与 :data:`DEMO_RECORDS` 是否一致。"""
    if not os.path.isfile(path):
        print(f"[FAIL] 文件不存在：{path}")
        return False
    with open(path, "r", encoding="utf-8") as handle:
        actual = [json.loads(line) for line in handle if line.strip()]
    expected = [build_instance(record).to_record() for record in DEMO_RECORDS]
    if len(actual) != len(expected):
        print(f"[FAIL] 条数不一致：文件 {len(actual)} 条，期望 {len(expected)} 条")
        return False
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            print(f"[FAIL] 第 {index + 1} 条不一致（uid={want['uid']}）")
            for key in sorted(set(got) | set(want)):
                if got.get(key) != want.get(key):
                    print(f"   字段 {key}:\n     文件 = {got.get(key)!r}\n     期望 = {want.get(key)!r}")
            return False
    print(f"[ OK ] {path} 与 DEMO_RECORDS 完全一致（{len(actual)} 条）")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="演示数据生成/校验")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args(argv)

    path = os.path.join(HERE, DEMO_FILENAME)
    if args.check:
        return 0 if check_records(path) else 1

    count = dump_records(path)
    print(f"已写出 {count} 条演示样本：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

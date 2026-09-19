# coding=utf-8
"""数据处理层：原始社交网络数据 → 统一数据实例 → 编码器输入文本。

子模块
------
* :mod:`data.processors.data_model`     统一数据模型（uid / string_value / replies / label）
* :mod:`data.processors.reply_flatten`  回复树展平与 token 预算分配
* :mod:`data.processors.twitter_rumor`  Twitter15/16 原始格式解析

``data/dataset.py`` 里的 PyTorch Dataset 依赖本包，但本包**不依赖 torch**，
因此数据预处理脚本可以在没有 GPU 环境的机器上运行。
"""

from .data_model import (
    ID_TO_LABEL,
    LABEL_ALIASES,
    LABEL_TO_ID,
    LABELS,
    DataInstance,
    Reply,
    count_replies,
    instance_to_record,
    iter_replies,
    normalize_label,
    record_to_instance,
)
from .reply_flatten import build_encoder_text, flatten_replies
from .twitter_rumor import load_raw_dataset, parse_source_tweets, parse_trees

__all__ = [
    "LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "LABEL_ALIASES",
    "normalize_label",
    "Reply",
    "DataInstance",
    "iter_replies",
    "count_replies",
    "instance_to_record",
    "record_to_instance",
    "build_encoder_text",
    "flatten_replies",
    "load_raw_dataset",
    "parse_source_tweets",
    "parse_trees",
]

# coding=utf-8
"""通用工具层：配置解析、日志、随机种子、JSONL 与产物目录。

本层不依赖 torch / transformers，可以被任何其它层安全导入。
"""

from .io_utils import (
    append_jsonl,
    ensure_dir,
    iter_jsonl,
    load_json,
    read_jsonl,
    save_json,
    write_jsonl,
)
from .logger import get_logger, setup_logging
from .seed import set_seed
from .config import Config, load_config, merge_dict, to_dict

__all__ = [
    "Config",
    "load_config",
    "merge_dict",
    "to_dict",
    "get_logger",
    "setup_logging",
    "set_seed",
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "iter_jsonl",
    "load_json",
    "save_json",
    "ensure_dir",
]

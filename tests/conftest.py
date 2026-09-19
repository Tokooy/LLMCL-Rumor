# coding=utf-8
"""pytest 公共 fixture 与工具。

约定
----
* 所有测试可在 **无 GPU、无网络** 环境下运行；
* 依赖 torch 的用例先调用 :func:`require_torch`，缺失时自动 skip；
* 需要临时目录的用例统一用 ``tmp_path`` fixture。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def require_torch():
    """返回 torch 模块；未安装时跳过当前用例。"""
    return pytest.importorskip("torch", reason="该用例需要 PyTorch")


def require_transformers():
    """返回 transformers 模块；未安装时跳过当前用例。"""
    return pytest.importorskip("transformers", reason="该用例需要 transformers")


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


def demo_dataset_path() -> str:
    """返回仓库自带演示数据的路径（普通函数，非 fixture）。

    之所以同时提供函数与 fixture：fixture 版给用例用，函数版给其它 fixture
    （例如 :func:`demo_file`）在内部调用——fixture 不能直接调用另一个 fixture。
    """
    path = os.path.join(REPO_ROOT, "data", "samples", "demo_twitter15.jsonl")
    if not os.path.isfile(path):
        pytest.skip(f"演示数据不存在：{path}")
    return path


@pytest.fixture(scope="session")
def demo_path() -> str:
    """仓库自带演示数据集的路径。"""
    return demo_dataset_path()


@pytest.fixture()
def demo_records(demo_path: str) -> List[Dict[str, Any]]:
    """演示数据集的原始 JSON 记录列表。"""
    from src.utils.io_utils import read_jsonl

    return read_jsonl(demo_path)


@pytest.fixture()
def demo_instances(demo_records: List[Dict[str, Any]]) -> List[Any]:
    """演示数据集的 DataInstance 列表。

    注意：每个用例拿到的是**新构造**的一份（这个 fixture 是 function 作用域），
    用例可以放心就地修改（例如清空 ``text`` 后重新生成）。
    """
    from data.processors.data_model import record_to_instance

    return [record_to_instance(record) for record in demo_records]


@pytest.fixture()
def demo_file(tmp_path) -> str:
    """把演示数据复制到临时目录并返回路径。

    用于那些会**写文件**的用例（例如读取后再落盘）：直接对仓库里的
    ``demo_twitter15.jsonl`` 做原地读写会污染工作区，在只读沙箱下还会直接失败。
    """
    import shutil

    target = tmp_path / "demo_twitter15.jsonl"
    shutil.copyfile(demo_dataset_path(), target)
    return str(target)


@pytest.fixture()
def prompt_builder():
    """默认 Prompt 编排器（不加载任何模型）。"""
    from src.llm.prompts import PromptBuilder

    return PromptBuilder(
        template_file=os.path.join(
            REPO_ROOT, "configs", "prompts", "augment_default.txt"
        ),
        label_descriptions={
            "NR": "non-rumor",
            "FR": "false rumor",
            "TR": "true rumor",
            "UR": "unverified rumor",
        },
    )


@pytest.fixture()
def base_config():
    """基础配置对象。"""
    from src.utils.config import load_config

    return load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"))

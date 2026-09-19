# coding=utf-8
"""日志与仓库级工具的单测（不需要 torch）。

覆盖两处曾经出错的实现细节：

* ``get_logger`` 的**首次调用**就必须创建日志文件——曾经的写法只在具名 logger
  第二次被调用时才挂文件 handler，而每个脚本只调用一次，
  于是 ``outputs/logs/*.log`` 永远不会出现；
* ``scripts/check_tracked.py`` 的忽略规则检查（防止源码被 .gitignore 静默吞掉）。
"""

from __future__ import annotations

import logging
import os

import pytest


# ===================================================================== #
# 日志
# ===================================================================== #
class TestLoggerSetup:
    """``src.utils.logger`` 的行为。"""

    @staticmethod
    def _fresh_logger_name(suffix: str) -> str:
        """用一个从未配置过的名字，避免与其他测试共享 _CONFIGURED 状态。"""
        return f"llmcl_test_{suffix}_{os.getpid()}"

    def test_file_created_on_first_call(self, tmp_path):
        """**首次**调用就必须生成日志文件。"""
        from src.utils.logger import get_logger

        log_file = tmp_path / "run.log"
        logger = get_logger(self._fresh_logger_name("first"), log_file=str(log_file))
        logger.info("hello from the first call")

        for handler in logger.handlers:
            handler.flush()
        assert log_file.is_file(), "首次调用未创建日志文件"
        content = log_file.read_text(encoding="utf-8")
        assert "hello from the first call" in content

    def test_console_handler_attached_once(self, tmp_path):
        """重复获取同一个 logger 不应重复挂 handler（否则日志会打印多份）。"""
        from src.utils.logger import get_logger

        name = self._fresh_logger_name("once")
        log_file = tmp_path / "once.log"
        logger = get_logger(name, log_file=str(log_file))
        console_before = [
            handler for handler in logger.handlers if not isinstance(handler, logging.FileHandler)
        ]
        get_logger(name, log_file=str(log_file))
        console_after = [
            handler for handler in logger.handlers if not isinstance(handler, logging.FileHandler)
        ]
        assert len(console_before) == len(console_after) == 1

    def test_file_handler_not_duplicated(self, tmp_path):
        """同一路径的日志文件只应有一个 handler，重复调用不会写两遍。"""
        from src.utils.logger import get_logger

        name = self._fresh_logger_name("dup")
        log_file = tmp_path / "dup.log"
        logger = get_logger(name, log_file=str(log_file))
        get_logger(name, log_file=str(log_file))
        get_logger(name, log_file=str(log_file))

        file_handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

        logger.info("written once")
        for handler in file_handlers:
            handler.flush()
        assert log_file.read_text(encoding="utf-8").count("written once") == 1

    def test_no_log_file_configured_is_fine(self):
        """不传 log_file 时不应报错，也不应创建文件。"""
        from src.utils.logger import get_logger

        logger = get_logger(self._fresh_logger_name("nofile"))
        logger.info("no file expected")
        assert not any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        )

    def test_nested_directory_is_created(self, tmp_path):
        from src.utils.logger import get_logger

        log_file = tmp_path / "a" / "b" / "c" / "deep.log"
        logger = get_logger(self._fresh_logger_name("deep"), log_file=str(log_file))
        logger.info("nested")
        for handler in logger.handlers:
            handler.flush()
        assert log_file.is_file()


# ===================================================================== #
# 仓库完整性检查脚本
# ===================================================================== #
class TestCheckTracked:
    """``scripts/check_tracked.py`` 的核心判定逻辑。"""

    @staticmethod
    def _module():
        import importlib.util

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "scripts", "check_tracked.py")
        spec = importlib.util.spec_from_file_location("check_tracked_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_source_like_detection(self):
        """应被追踪的扩展名与允许忽略的目录。"""
        module = self._module()
        assert module._is_source_like("src/models/encoder.py")
        assert module._is_source_like("configs/base.yaml")
        assert module._is_source_like("tests/test_data.py")
        # 数据与产物目录属于预期忽略
        assert not module._is_source_like("data/raw/twitter15/label.txt")
        assert not module._is_source_like("data/processed/demo/train.jsonl")
        assert not module._is_source_like("outputs/checkpoints/best.pt")
        # 非源码扩展名
        assert not module._is_source_like("outputs/figures/tsne.png")

    def test_current_repo_has_no_ignored_sources(self):
        """当前仓库不应有被 .gitignore 误伤的源码（回归测试）。

        这条用例正是为了记住 ``models/`` 那个坑：它曾经连 ``src/models/`` 一起忽略，
        导致整个模型层源码没进版本库，而 git status 毫无提示。
        """
        module = self._module()
        ignored = module.ignored_source_files()
        assert ignored == [], f"以下源码被 .gitignore 误伤：{ignored}"

    def test_ignored_and_untracked_are_disjoint_checks(self):
        """两个检查项的含义不同，返回值都应是列表。"""
        module = self._module()
        assert isinstance(module.ignored_source_files(), list)
        assert isinstance(module.untracked_source_files(), list)


# ===================================================================== #
# 极简 YAML 解析器
# ===================================================================== #
class TestMinimalYaml:
    """``src.utils.minimal_yaml`` 的解析能力（无 pyyaml 时的回退实现）。"""

    def test_scalars(self):
        from src.utils.minimal_yaml import _parse_scalar

        assert _parse_scalar("123") == 123
        assert _parse_scalar("-4.5") == -4.5
        assert _parse_scalar("true") is True
        assert _parse_scalar("False") is False
        assert _parse_scalar("null") is None
        assert _parse_scalar("plain text") == "plain text"
        assert _parse_scalar('"quoted"') == "quoted"
        assert _parse_scalar("[1, 2, 3]") == [1, 2, 3]

    def test_flow_list_with_strings(self):
        from src.utils.minimal_yaml import _parse_scalar

        assert _parse_scalar("[a, b, c]") == ["a", "b", "c"]
        assert _parse_scalar("[]") == []

    def test_nested_mapping_and_list(self, tmp_path):
        from src.utils.minimal_yaml import simple_yaml_load

        path = tmp_path / "sample.yaml"
        path.write_text(
            "# comment\n"
            "top: 1\n"
            "nested:\n"
            "  a: true\n"
            "  b: 2.5\n"
            "list:\n"
            "  - one\n"
            "  - two\n"
            "inline: [x, y]\n"
            "empty_string: \"\"\n",
            encoding="utf-8",
        )
        parsed = simple_yaml_load(str(path))
        assert parsed["top"] == 1
        assert parsed["nested"] == {"a": True, "b": 2.5}
        assert parsed["list"] == ["one", "two"]
        assert parsed["inline"] == ["x", "y"]
        assert parsed["empty_string"] == ""

    def test_trailing_comment_is_stripped(self, tmp_path):
        from src.utils.minimal_yaml import simple_yaml_load

        path = tmp_path / "comment.yaml"
        path.write_text("key: 42  # the answer\n", encoding="utf-8")
        assert simple_yaml_load(str(path)) == {"key": 42}

    def test_hash_inside_quotes_is_kept(self, tmp_path):
        from src.utils.minimal_yaml import simple_yaml_load

        path = tmp_path / "hash.yaml"
        path.write_text('key: "a # b"\n', encoding="utf-8")
        assert simple_yaml_load(str(path)) == {"key": "a # b"}

    def test_real_config_files_parse(self):
        """仓库里真实的配置都要能被回退解析器读出来（键齐全）。"""
        from src.utils.minimal_yaml import simple_yaml_load

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = simple_yaml_load(os.path.join(repo_root, "configs", "base.yaml"))
        for section in ("paths", "data", "model", "llm", "training", "evaluation"):
            assert section in base, f"base.yaml 缺少 {section} 节"
        assert base["data"]["split"]["train"] == 0.7
        assert base["training"]["cl"]["augment_every_epochs"] == 1

        proposed = simple_yaml_load(
            os.path.join(repo_root, "configs", "experiments", "proposed-4.yaml")
        )
        assert proposed["experiment"]["name"] == "proposed-4"
        assert proposed["training"]["alignment"]["max_finetune_rounds"] == 1
        assert proposed["defaults"] == ["../llm/qwen7b.yaml"]

    def test_cross_validation_when_pyyaml_available(self):
        """装了 pyyaml 时，回退解析器必须与官方实现结果完全一致。"""
        pytest.importorskip("yaml", reason="需要 pyyaml 才能做交叉验证")
        from src.utils.minimal_yaml import load_yaml, simple_yaml_load

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for relative in (
            "configs/base.yaml",
            "configs/llm/qwen7b.yaml",
            "configs/llm/qwen13b.yaml",
            "configs/experiments/proposed-1.yaml",
            "configs/experiments/proposed-6.yaml",
        ):
            path = os.path.join(repo_root, relative)
            official, parser_name = load_yaml(path)
            assert parser_name == "pyyaml"
            assert simple_yaml_load(path) == official, f"{relative} 两份解析器结果不一致"

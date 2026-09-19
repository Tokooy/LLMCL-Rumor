# coding=utf-8
"""纯 Python 用例实跑器：在没有 pytest / torch 的机器上也能验证核心逻辑。

**为什么需要它**

本项目的目标环境要装 torch + transformers + sklearn，但在开发/交付阶段常常拿不到
这些依赖（本机就是如此）。此时 ``pytest tests`` 直接 ``ModuleNotFoundError: pytest``，
一个用例都跑不了，"逻辑到底改通没有"就无从验证。

本脚本注入一个**最小 pytest 替身**，并把真实 pytest 里最常用的几件事补上：

* ``importorskip`` / ``skip`` → 记为 **skip**（缺依赖）而不是失败；
* ``tmp_path`` → 传一个仓库内的临时目录（每个用例独立）；
* **conftest 里的 fixture** → 从 ``tests/conftest.py`` 解析并按参数名注入
  （``demo_path`` / ``demo_instances`` / ``prompt_builder`` / ``base_config`` /
  ``demo_records`` / ``repo_root``）；
* ``@pytest.mark.parametrize`` → 记录在函数属性上，由本脚本展开成多个用例。

**它不运行任何项目代码**：不加载模型、不读数据集、不训练。被测对象是配置系统、
解析器、日志、数据模型、任务向量算子、Algorithm 2 调度与评估指标这些纯计算逻辑。

用法::

    python scripts/run_pure_python_tests.py              # 跑全部可跑的用例
    python scripts/run_pure_python_tests.py --list       # 只列出会跑哪些模块
    python scripts/run_pure_python_tests.py --only test_utils_and_logger --verbose

退出码：0 = 无失败（允许 skip）；1 = 存在失败。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import pathlib
import re
import shutil
import sys
import traceback
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: 默认要尝试的测试模块（按依赖从轻到重）
DEFAULT_MODULES = [
    "tests.test_utils_and_logger",
    "tests.test_prompts_and_parser",
    "tests.test_config_and_metrics",
    "tests.test_data",
    "tests.test_joint_trainer",
    "tests.test_ties_merge",
    "tests.test_losses",
]

TMP_ROOT_NAME = ".tmp_pure_tests"

#: parametrize 参数集挂在函数上的属性名（由本脚本的 mark 替身写入）
PARAM_ATTR = "__pure_parametrize__"


class _Skipped(Exception):
    """缺依赖导致的跳过（对应 pytest.importorskip / pytest.skip）。"""


class _Approx:
    """``pytest.approx`` 的最小实现。"""

    def __init__(self, expected: Any, rel: Optional[float] = None, abs: Optional[float] = None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other: Any) -> bool:
        try:
            if self.abs is not None:
                return abs(other - self.expected) <= self.abs
            rel = self.rel if self.rel is not None else 1e-6
            return abs(other - self.expected) <= max(abs(self.expected) * rel, 1e-12)
        except TypeError:  # pragma: no cover
            return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


class _Raises:
    """``pytest.raises`` 的最小实现（含 ``match=`` 正则校验）。

    ``match`` 必须实现：本项目有 20 处 ``pytest.raises(..., match="...")``，
    只校验异常类型的话，"异常消息写错了"这类问题在本地跑不出来，
    但换到真 pytest 下就会失败——那会让这份实跑结果的可信度虚高。
    """

    def __init__(self, expected: Any, match: Optional[str] = None):
        self.expected = expected
        self.match = match
        self.value: Optional[BaseException] = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected}")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None and not re.search(self.match, str(exc)):
            raise AssertionError(
                f"异常消息不匹配 match={self.match!r}；实际消息：{str(exc)!r}"
            )
        self.value = exc
        return True


def install_pytest_stub() -> bool:
    """注入 pytest 替身。

    Returns:
        True 表示注入了替身；False 表示环境里已有真正的 pytest。
    """
    try:
        import pytest  # noqa: F401

        return False
    except ImportError:
        pass

    stub = types.ModuleType("pytest")

    def importorskip(name: str, reason: str = "", **kwargs: Any) -> Any:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise _Skipped(f"{name}（{reason or '缺少依赖'}）") from exc

    def skip(reason: str = "", **kwargs: Any) -> None:
        raise _Skipped(reason or "skip")

    def fixture(*args: Any, **kwargs: Any):
        def decorator(function: Callable) -> Callable:
            return function

        if args and callable(args[0]):
            return args[0]
        return decorator

    class _Mark:
        """支持 ``@pytest.mark.torch`` / ``@pytest.mark.parametrize(...)``。"""

        def __getattr__(self, name: str):
            def decorator(*args: Any, **kwargs: Any):
                # 直接装饰（@pytest.mark.torch）
                if args and callable(args[0]) and not kwargs and name != "parametrize":
                    return args[0]

                def wrapper(function: Callable) -> Callable:
                    if name == "parametrize" and args:
                        # 参数名可能是单个字符串或逗号分隔的字符串
                        names = args[0]
                        if isinstance(names, str):
                            names = [part.strip() for part in names.split(",") if part.strip()]
                        sets = list(args[1]) if len(args) > 1 else []
                        existing = getattr(function, PARAM_ATTR, [])
                        function.__dict__[PARAM_ATTR] = existing + [(list(names), list(sets))]
                    return function

                return wrapper

            return decorator

    stub.importorskip = importorskip
    stub.skip = skip
    stub.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel, abs)
    stub.raises = lambda expected, match=None, **kwargs: _Raises(expected, match=match)
    stub.fixture = fixture
    stub.mark = _Mark()
    stub.main = lambda *a, **k: 0

    sys.modules["pytest"] = stub
    return True


# ---------------------------------------------------------------------- #
# conftest fixture 支持
# ---------------------------------------------------------------------- #
class FixtureResolver:
    """从 ``tests/conftest.py`` 解析 fixture 并按需构造。

    fixture 之间存在依赖（例如 ``demo_instances`` 依赖 ``demo_records``），
    因此这里做一次简单的递归解析，并把结果按"每用例一次"缓存
    （``demo_instances`` 会被用例修改，不能跨用例复用）。
    """

    def __init__(self, conftest_path: str, tmp_root: pathlib.Path):
        self.tmp_root = tmp_root
        self.functions: Dict[str, Callable] = {}
        self._load(conftest_path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        namespace: Dict[str, Any] = {"__file__": path, "__name__": "conftest_for_runner"}
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        exec(compile(source, path, "exec"), namespace)  # noqa: S102 - 只执行仓库自带的 conftest
        for name, value in namespace.items():
            if callable(value) and not name.startswith("_"):
                self.functions[name] = value

    def has(self, name: str) -> bool:
        return name in self.functions

    def resolve(self, name: str, case_dir: pathlib.Path, _seen: Optional[set] = None) -> Any:
        """构造名为 ``name`` 的 fixture 值。"""
        if name == "tmp_path":
            case_dir.mkdir(parents=True, exist_ok=True)
            return case_dir
        if name not in self.functions:
            raise _Skipped(f"fixture {name!r} 不在 conftest 中，本脚本无法提供")

        seen = set(_seen or ())
        if name in seen:
            raise _Skipped(f"fixture {name!r} 存在循环依赖")
        seen.add(name)

        function = self.functions[name]
        signature = inspect.signature(function)
        kwargs: Dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue
            kwargs[parameter.name] = self.resolve(parameter.name, case_dir, seen)
        return function(**kwargs)


# ---------------------------------------------------------------------- #
# 执行
# ---------------------------------------------------------------------- #
def _iter_cases(method: Callable) -> List[Tuple[str, Dict[str, Any]]]:
    """把一个测试方法展开成 ``(用例后缀, 参数)`` 列表（处理 parametrize）。"""
    parametrize_sets = getattr(method, PARAM_ATTR, None)
    if not parametrize_sets:
        return [("", {})]

    # 只处理单层 parametrize；多层是极少数情况，遇到就跳过
    if len(parametrize_sets) > 1:
        return [("", {})]

    names, sets = parametrize_sets[0]
    cases: List[Tuple[str, Dict[str, Any]]] = []
    for values in sets:
        if len(names) == 1:
            payload = {names[0]: values}
            suffix = f"[{values!r}]"
        else:
            payload = dict(zip(names, values))
            suffix = "[" + ",".join(repr(value) for value in values) + "]"
        cases.append((suffix, payload))
    return cases


def _needs_tmp_path(method: Callable) -> bool:
    signature = inspect.signature(method)
    return "tmp_path" in signature.parameters


#: 可选依赖：这些包缺失时，相关用例应当记为 skip（等价于 pytest 的 importorskip）
OPTIONAL_DEPENDENCIES = ("torch", "sklearn", "yaml", "matplotlib", "transformers", "numpy", "peft")


def _optional_dependency_of(exc: BaseException) -> Optional[str]:
    """如果异常是"缺少某个可选依赖"，返回该依赖名；否则返回 None。"""
    if not isinstance(exc, ModuleNotFoundError):
        return None
    name = (getattr(exc, "name", "") or "").split(".")[0]
    return name if name in OPTIONAL_DEPENDENCIES else None


def run_module(
    module_name: str,
    fixtures: FixtureResolver,
    tmp_root: pathlib.Path,
    verbose: bool = False,
) -> Tuple[int, int, int, List[str]]:
    """跑一个测试模块里所有可执行的用例。

    Returns:
        ``(passed, skipped, failed, failures)``。
    """
    try:
        module = importlib.import_module(module_name)
    except _Skipped as exc:
        return 0, 1, 0, []
    except ImportError:
        return 0, 1, 0, []

    passed = skipped = failed = 0
    failures: List[str] = []
    case_dir = tmp_root / module_name.replace(".", "_")
    case_dir.mkdir(parents=True, exist_ok=True)

    for class_name, klass in sorted(vars(module).items()):
        if not inspect.isclass(klass) or not class_name.startswith("Test"):
            continue
        try:
            instance = klass()
        except Exception as exc:  # pragma: no cover
            failed += 1
            failures.append(f"{module_name}::{class_name} 实例化失败：{exc}")
            continue

        for method_name in sorted(name for name in dir(instance) if name.startswith("test_")):
            method = getattr(instance, method_name)
            for suffix, extra in _iter_cases(method):
                label = f"{class_name}::{method_name}{suffix}"
                work_dir = case_dir / f"{method_name}{suffix}".replace("[", "_").replace("]", "").replace(",", "_")
                try:
                    signature = inspect.signature(method)
                    kwargs = dict(extra)
                    for parameter in signature.parameters.values():
                        if parameter.name == "self" or parameter.name in kwargs:
                            continue
                        kwargs[parameter.name] = fixtures.resolve(parameter.name, work_dir)
                    method(**kwargs)
                except _Skipped as exc:
                    skipped += 1
                    if verbose:
                        print(f"  [SKIP] {label}（{exc}）")
                except Exception as exc:  # noqa: BLE001 - 收集所有失败
                    # 缺少可选依赖（torch/sklearn/pyyaml…）时按 skip 处理，
                    # 与 pytest 的 importorskip 语义一致：无法执行 ≠ 失败。
                    missing = _optional_dependency_of(exc)
                    if missing:
                        skipped += 1
                        if verbose:
                            print(f"  [SKIP] {label}（未安装 {missing}）")
                        continue
                    failed += 1
                    detail = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    failures.append(f"{module_name}::{label}: {detail}")
                    print(f"  [FAIL] {module_name}::{label}\n         {detail}")
                else:
                    passed += 1
                    if verbose:
                        print(f"  [ OK ] {label}")

    return passed, skipped, failed, failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="纯 Python 用例实跑器（无需 pytest/torch）")
    parser.add_argument("--only", action="append", default=None,
                        help="只跑指定模块（可重复），例如 --only test_utils_and_logger")
    parser.add_argument("--list", action="store_true", help="只列出会跑哪些模块")
    parser.add_argument("--verbose", action="store_true", help="打印每个用例的结果")
    args = parser.parse_args(argv)

    modules = DEFAULT_MODULES
    if args.only:
        modules = [
            name if name.startswith("tests.") else f"tests.{name}" for name in args.only
        ]

    if args.list:
        print("将尝试以下测试模块（缺少依赖的自动跳过）：")
        for name in modules:
            print(f"  - {name}")
        return 0

    injected = install_pytest_stub()
    # 用 NOTSET：部分用例会断言日志文件内容，屏蔽 INFO 会让它们假失败
    logging.disable(logging.NOTSET)

    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False

    tmp_root = pathlib.Path(REPO_ROOT) / TMP_ROOT_NAME
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("纯 Python 用例实跑（不加载模型、不读数据集、不训练）")
    print(
        f"pytest：{'环境已安装' if not injected else '使用内置替身'}"
        f" | torch：{'已安装' if has_torch else '未安装（张量用例将跳过）'}"
    )
    print("=" * 72)

    total_passed = total_skipped = total_failed = 0
    all_failures: List[str] = []
    for module_name in modules:
        fixtures = FixtureResolver(
            os.path.join(REPO_ROOT, "tests", "conftest.py"), tmp_root
        )
        print(f"[模块] {module_name}")
        passed, skipped, failed, failures = run_module(
            module_name, fixtures, tmp_root, verbose=args.verbose
        )
        print(f"       通过 {passed}，跳过 {skipped}，失败 {failed}")
        total_passed += passed
        total_skipped += skipped
        total_failed += failed
        all_failures.extend(failures)

    shutil.rmtree(tmp_root, ignore_errors=True)

    print("-" * 72)
    print(f"合计：通过 {total_passed}，跳过 {total_skipped}，失败 {total_failed}")
    if total_skipped:
        print(
            "跳过的用例依赖 torch / scikit-learn / pyyaml 或未实现的 fixture；"
            "装齐 requirements.txt 后请用 `pytest tests -q` 做完整执行。"
        )
    if all_failures:
        print("失败明细：")
        for item in all_failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

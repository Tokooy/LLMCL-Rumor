# coding=utf-8
"""静态语法自检：对仓库内所有 Python 文件做 AST 解析。

**这不是运行项目**。`ast.parse` 只做语法分析，不导入任何模块、不加载权重、
不触发任何训练或数据流；它的唯一目的是在没有安装 torch/transformers 的机器上
也能确认"每个文件的语法树是完整的"（拼写、缩进、括号、f-string 等硬错误）。

用法::

    python scripts/check_syntax.py            # 检查 src/ scripts/ tests/ data/
    python scripts/check_syntax.py --verbose  # 同时打印每个文件的行数

退出码：0 = 全部通过；1 = 存在语法错误。
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGETS = ["src", "scripts", "tests", "data"]


def iter_python_files(targets: List[str]) -> List[str]:
    """收集目标目录下的所有 .py 文件（跳过缓存目录与虚拟环境）。"""
    skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", "build", "dist"}
    files: List[str] = []
    for target in targets:
        path = target if os.path.isabs(target) else os.path.join(REPO_ROOT, target)
        if os.path.isfile(path) and path.endswith(".py"):
            files.append(path)
            continue
        if not os.path.isdir(path):
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
    return sorted(set(files))


def check_file(path: str) -> Tuple[bool, str]:
    """解析单个文件，返回 (是否通过, 错误信息或行数说明)。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        return False, f"读取失败：{exc}"

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, col {exc.offset}"
        return False, f"SyntaxError @ {location}: {exc.msg}"

    n_lines = source.count("\n") + 1
    n_defs = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    return True, f"{n_lines} lines, {n_defs} defs/classes"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AST 语法自检（不运行代码）")
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS,
                        help=f"要检查的目录或文件，默认 {DEFAULT_TARGETS}")
    parser.add_argument("--verbose", action="store_true", help="打印每个文件的详情")
    args = parser.parse_args(argv)

    files = iter_python_files(args.targets or DEFAULT_TARGETS)
    if not files:
        print("没有找到任何 .py 文件，请检查路径。")
        return 1

    failures: List[Tuple[str, str]] = []
    for path in files:
        ok, message = check_file(path)
        rel = os.path.relpath(path, REPO_ROOT)
        if ok:
            if args.verbose:
                print(f"  [ OK ] {rel:<60} {message}")
        else:
            failures.append((rel, message))
            print(f"  [FAIL] {rel:<60} {message}")

    print("-" * 72)
    print(f"检查完成：{len(files) - len(failures)}/{len(files)} 个文件通过")
    if failures:
        print("存在语法错误的文件：")
        for rel, message in failures:
            print(f"  - {rel}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

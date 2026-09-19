# coding=utf-8
"""静态一致性自检：不导入项目模块，只用 AST 找"低级错误"。

**这不是运行项目**：脚本自身只读文件、解析语法树，不 import 任何被检查的模块，
不加载模型、不触发训练或数据流。

检查项
------
1. ``__all__`` 中列出但模块内未定义/未导入的名字（包 ``__init__.py`` 里的
   子模块名会被正确识别，不会误报）；
2. 函数体内引用的裸名既不是局部变量（含闭包捕获）、不是模块级名字、
   也不是内置名或 ``self``/``cls``（典型的"忘了 import"或"变量名写错"）；
3. 同一模块内重复定义的顶层函数/类名；
4. **跨模块导入交叉检查**：``from src.x.y import name`` 里的模块文件是否存在、
   ``name`` 是否真的在目标模块里定义（这类错误会让测试在收集阶段就 ImportError，
   而本机没装 torch 时根本跑不到那一步）。

用法::

    python scripts/check_static.py            # 检查 src/ scripts/ tests/ data/
    python scripts/check_static.py --strict   # 有任何提示即返回非 0

退出码：默认 0（提示仅供人工确认）；``--strict`` 时有问题返回 1。
"""

from __future__ import annotations

import argparse
import ast
import builtins
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGETS = ["src", "scripts", "tests", "data"]
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "build", "dist", "node_modules"}
IMPLICIT_LOCALS = {
    "self",
    "cls",
    "__class__",
    # 模块级 dunder：由导入系统注入，AST 上看不到赋值语句
    "__file__",
    "__name__",
    "__doc__",
    "__package__",
    "__spec__",
    "__loader__",
    "__builtins__",
}
#: 参与跨模块导入检查的包前缀
CHECKED_PACKAGES = ("src", "data")


def iter_python_files(targets: List[str]) -> List[str]:
    files: List[str] = []
    for target in targets:
        path = target if os.path.isabs(target) else os.path.join(REPO_ROOT, target)
        if os.path.isfile(path) and path.endswith(".py"):
            files.append(path)
            continue
        if not os.path.isdir(path):
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
    return sorted(set(files))


def _bindings(node: ast.AST) -> Set[str]:
    """递归收集一个作用域内所有被绑定的名字。

    同时收集**嵌套作用域**里的绑定，这样闭包变量（外层函数定义的变量被内层
    函数引用）不会被误判成"未定义"；代价是嵌套作用域的同名遮蔽可能被漏检，
    这属于可接受的误放行方向（宁可少报，不可乱报）。
    """
    names: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)  # type: ignore[arg-type]
        elif isinstance(sub, ast.Lambda):
            pass
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
        elif isinstance(sub, ast.comprehension):
            for target in ast.walk(sub.target):
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                if item.optional_vars is not None:
                    for target in ast.walk(item.optional_vars):
                        if isinstance(target, ast.Name):
                            names.add(target.id)
        elif isinstance(sub, (ast.For, ast.AsyncFor)):
            for target in ast.walk(sub.target):
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(sub, ast.arg):
            names.add(sub.arg)
        elif isinstance(sub, ast.Global) or isinstance(sub, ast.Nonlocal):
            names.update(sub.names)
    return names


def _module_bindings(tree: ast.Module) -> Tuple[Set[str], Set[str]]:
    """返回（模块级已定义/绑定名, 已导入名）。"""
    defined: Set[str] = set()
    imported: Set[str] = set()
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                for alias in sub.names:
                    imported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(sub.name)  # type: ignore[arg-type]
            elif isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            defined.add(name.id)
    return defined, imported


def _submodules(package_init: str, tree: ast.Module) -> Set[str]:
    """包 ``__init__.py`` 中可用的子模块名（同目录下的 *.py 与子包目录）。"""
    directory = os.path.dirname(package_init)
    modules: Set[str] = set()
    if not os.path.isdir(directory):
        return modules
    for entry in os.listdir(directory):
        full = os.path.join(directory, entry)
        if entry.endswith(".py") and entry != "__init__.py":
            modules.add(entry[:-3])
        elif os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")):
            modules.add(entry)
    return modules


def _module_public_names(path: str) -> Set[str]:
    """收集一个模块里被定义/导入的名字集合（用于跨模块导入检查）。"""
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), path)
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def check_cross_module_imports(path: str, source: str) -> List[str]:
    """检查 ``from src./data. import ...`` 的模块与名字是否真实存在。

    这类错误在没装 torch 的机器上跑 pytest 是查不出来的：测试收集阶段就会
    ImportError，但因为 ``importorskip`` 之前就崩了，报错信息也不明显。
    纯 AST 检查可以提前把这类问题抓出来。
    """
    problems: List[str] = []
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return problems  # 语法错误由 check_syntax.py 负责报

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith(tuple(f"{name}." for name in CHECKED_PACKAGES)):
            continue

        module_path = os.path.join(REPO_ROOT, node.module.replace(".", os.sep) + ".py")
        if not os.path.isfile(module_path):
            # 也可能是包（目录 + __init__.py）
            package_path = os.path.join(
                REPO_ROOT, node.module.replace(".", os.sep), "__init__.py"
            )
            if not os.path.isfile(package_path):
                problems.append(
                    f"{path}:{node.lineno}: 模块 {node.module!r} 不存在"
                    f"（既不是 {os.path.relpath(module_path, REPO_ROOT)} 也不是包）"
                )
                continue
            module_path = package_path

        if not os.path.isfile(module_path):
            continue

        available = _module_public_names(module_path)
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name not in available:
                problems.append(
                    f"{path}:{node.lineno}: {node.module} 中不存在 {alias.name!r}"
                )
    return problems


def check_module(path: str) -> List[str]:
    """返回该模块的所有静态提示。"""
    problems: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        return [f"读取失败：{exc}"]

    tree = ast.parse(source, path)
    defined, imported = _module_bindings(tree)
    known = set(defined) | set(imported) | set(dir(builtins)) | IMPLICIT_LOCALS

    # ---- 1) __all__ 覆盖检查 ----
    exported: Optional[Set[str]] = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "__all__" for target in node.targets
        ):
            exported = {
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    if exported is not None:
        allowed = set(defined) | set(imported)
        if os.path.basename(path) == "__init__.py":
            allowed |= _submodules(path, tree)
        for name in sorted(exported):
            if name not in allowed:
                problems.append(
                    f"{path}: __all__ 列出的 {name!r} 未在模块中定义、导入，也不是子模块"
                )

    # ---- 2) 重复的顶层定义 ----
    seen: Dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                problems.append(
                    f"{path}:{node.lineno}: 顶层名字 {node.name!r} 重复定义"
                    f"（首次出现于第 {seen[node.name]} 行）"
                )
            seen[node.name] = node.lineno

    # ---- 3) 函数体内可能未定义的名字 ----
    # 闭包会从**外层函数**捕获变量：嵌套函数 def _log() 里引用外层 load_raw_dataset()
    # 的参数 logger 是合法的。这里先把所有顶层函数的局部绑定汇总成"外层可见名"，
    # 供嵌套函数检查时一并放行。
    outer_bindings: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            outer_bindings |= _bindings(node)

    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._check(node)
            self.generic_visit(node)

        def _check(self, node) -> None:
            local = _bindings(node) | outer_bindings
            reported: Set[Tuple[int, str]] = set()
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Name) or not isinstance(sub.ctx, ast.Load):
                    continue
                if sub.id in local or sub.id in known:
                    continue
                key = (sub.lineno, sub.id)
                if key in reported:
                    continue
                reported.add(key)
                problems.append(
                    f"{path}:{sub.lineno}: 函数 {node.name}() 中引用的 {sub.id!r} "
                    "既非局部变量也非模块级/内置名（疑似漏 import 或拼写错误）"
                )

    # 只对模块顶层函数做检查；嵌套函数与 lambda 已由 _bindings 覆盖
    visitor = FunctionVisitor()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visitor.visit(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visitor._check(item)

    # ---- 4) 跨模块导入交叉检查 ----
    problems.extend(check_cross_module_imports(path, source))

    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AST 静态一致性自检（不运行代码）")
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--strict", action="store_true", help="有提示即返回非 0")
    args = parser.parse_args(argv)

    files = iter_python_files(args.targets or DEFAULT_TARGETS)
    total = 0
    for path in files:
        problems = check_module(path)
        rel = os.path.relpath(path, REPO_ROOT)
        if problems:
            total += len(problems)
            print(f"[提示] {rel}")
            for problem in problems:
                print(f"    - {problem}")
        else:
            print(f"[ OK ] {rel}")

    print("-" * 72)
    print(f"检查完成：{len(files)} 个文件，{total} 条提示")
    if total and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

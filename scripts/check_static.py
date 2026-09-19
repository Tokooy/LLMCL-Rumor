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
   而本机没装 torch 时根本跑不到那一步）；
5. **跨模块调用签名检查**：调用 ``f(a, b, keyword=c)`` 时，``keyword`` 是否是
   目标函数接受的参数、必需参数是否漏传。这类缺陷**不会抛任何异常**——
   例如"某函数新增了一个能改变语义的可选参数，但调用点没传"，
   语法检查、类型检查、甚至跑一遍都可能看不出来，因此需要工具兜底。

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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


def _collect_signatures(path: str, module_name: str) -> Dict[str, ast.arguments]:
    """收集一个模块里所有可调用对象的参数表，键形如 ``func`` / ``Class.method``。

    只做浅层解析（模块级函数 + 类内方法），这已经覆盖了本项目里绝大多数
    "跨文件调用"的场景。
    """
    signatures: Dict[str, ast.arguments] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), path)
    except (OSError, SyntaxError):
        return signatures

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = node.args
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    signatures[f"{node.name}.{item.name}"] = item.args
    return signatures


def _iter_star_args(arguments: ast.arguments) -> Any:
    return arguments.vararg, arguments.kwarg


def _signature_problems(
    arguments: ast.arguments,
    call: ast.Call,
    display_name: str,
    is_method: bool = False,
) -> List[str]:
    """对比一次调用与目标函数签名，返回参数层面的问题描述。

    Args:
        arguments: 目标函数的参数表。
        call: 调用节点。
        display_name: 用于报错展示的名字。
        is_method: 目标是否是类方法（``Class.method`` 形式）。为 True 时
            需要跳过隐式绑定的第一个参数（``self`` / ``cls``），
            否则 ``Reply.from_record(record)`` 会被误报成"缺少参数"。
    """
    problems: List[str] = []

    positional = list(arguments.posonlyargs) + list(arguments.args)
    if is_method and positional:
        # 实例方法/类方法的第一个位置参数由绑定隐式提供
        positional = positional[1:]
    all_params = {arg.arg for arg in positional}
    all_params |= {arg.arg for arg in arguments.kwonlyargs}
    vararg, kwarg = _iter_star_args(arguments)
    accepts_kwargs = kwarg is not None
    accepts_varargs = vararg is not None
    required_positional = len(positional)

    # 1) 关键字参数名是否存在
    seen_keywords: Dict[str, ast.keyword] = {}
    for keyword in call.keywords:
        if keyword.arg is None:      # **kwargs 展开，无法静态判断
            accepts_kwargs = True
            continue
        seen_keywords[keyword.arg] = keyword
        if keyword.arg not in all_params and not accepts_kwargs:
            problems.append(
                f"{display_name} 不接受关键字参数 {keyword.arg!r}"
                f"（可选：{sorted(all_params)}）"
            )

    # 2) 缺少必需参数（按位置与关键字一起算）
    if not accepts_varargs and not accepts_kwargs:
        supplied = len(call.args)
        for index, arg in enumerate(positional):
            has_default = index >= len(positional) - len(arguments.defaults)
            if has_default or index < supplied or arg.arg in seen_keywords:
                continue
            problems.append(f"{display_name} 缺少必需参数 {arg.arg!r}")
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if default is None and arg.arg not in seen_keywords:
                problems.append(f"{display_name} 缺少必需的关键字参数 {arg.arg!r}")

    # 3) 位置参数给多了
    if not accepts_varargs and len(call.args) > required_positional:
        problems.append(
            f"{display_name} 最多接受 {required_positional} 个位置参数，"
            f"但传入了 {len(call.args)} 个"
        )

    return problems


class SignatureIndex:
    """跨模块调用签名索引：``模块名 -> {符号名: 参数表}``。

    用于检查"调用点是否漏传/错传参数"——这类错误（例如漏传一个能改变语义的
    关键字参数）**不会**引发任何运行时异常，静态语法检查也看不到，
    正是最需要工具兜底的一类缺陷。
    """

    def __init__(self) -> None:
        self.modules: Dict[str, Dict[str, ast.arguments]] = {}
        self.imports: Dict[str, Dict[str, str]] = {}

    def build(self, files: Sequence[str]) -> None:
        """建索引。

        **索引范围始终覆盖整个仓库**（``src/ data/ scripts/ tests/``），
        与"这次要检查哪些目标"无关：否则只检查某个子目录时，
        被调用方不在索引里，跨模块调用就会静默跳过检查（漏报）。
        """
        scanned: List[str] = []
        for target in DEFAULT_TARGETS:
            scanned.extend(iter_python_files([target]))
        scanned.extend(files)
        for path in sorted(set(scanned)):
            if not path.endswith(".py"):
                continue
            relative = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
            if relative.startswith("reference/") or relative.startswith(".tmp"):
                continue
            module_name = relative[:-3].replace("/", ".")
            self.modules[module_name] = _collect_signatures(path, module_name)

            # 记录该模块里 from ... import ... 的别名（用于解析 Class.method）
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), path)
            except (OSError, SyntaxError):
                continue
            mapping: Dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        mapping[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            self.imports[module_name] = mapping

    def resolve(self, current_module: str, dotted: str) -> Optional[ast.arguments]:
        """把一次调用的名字解析成目标函数的参数表；无法确定时返回 None。"""
        parts = dotted.split(".")
        # 情况一：本模块内的函数/方法
        for cut in range(len(parts), 0, -1):
            candidate = ".".join(parts[:cut])
            if candidate in self.modules.get(current_module, {}):
                return self.modules[current_module][candidate]

        # 情况二：module.attr.method / module.func
        for cut in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:cut])
            if module_name in self.modules:
                rest = ".".join(parts[cut:])
                signature = self.modules[module_name].get(rest)
                if signature is not None:
                    return signature

        # 情况三：from x import Y 后的 Y.method
        if len(parts) >= 2 and parts[0] in self.imports.get(current_module, {}):
            target = self.imports[current_module][parts[0]]
            module_name, _, attr = target.rpartition(".")
            rest = ".".join([attr] + parts[1:])
            if module_name in self.modules:
                signature = self.modules[module_name].get(rest)
                if signature is not None:
                    return signature
        return None


def check_call_signatures(
    path: str,
    source: str,
    index: SignatureIndex,
) -> List[str]:
    """检查跨模块调用的参数是否与目标函数签名一致。"""
    problems: List[str] = []
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return problems

    relative = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    current_module = relative[:-3].replace("/", ".") if relative.endswith(".py") else relative

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name = self._dotted_name(node.func)
            if name:
                signature = index.resolve(current_module, name)
                if signature is not None:
                    # 形如 A.b(...) 的目标是类方法/实例方法，跳过隐式首参
                    is_method = "." in name
                    for problem in _signature_problems(
                        signature, node, name, is_method=is_method
                    ):
                        problems.append(f"{path}:{node.lineno}: {problem}")
            self.generic_visit(node)

        @staticmethod
        def _dotted_name(node: ast.AST) -> Optional[str]:
            """把 ``a.b.c`` 形式的调用目标还原成字符串；其它形式返回 None。"""
            parts: List[str] = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            else:
                return None
            return ".".join(reversed(parts))

    CallVisitor().visit(tree)
    return problems


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


def check_module(path: str, index: Optional["SignatureIndex"] = None) -> List[str]:
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

    # ---- 5) 跨模块调用签名检查（漏传/错传参数）----
    if index is not None:
        problems.extend(check_call_signatures(path, source, index))

    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AST 静态一致性自检（不运行代码）")
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--strict", action="store_true", help="有提示即返回非 0")
    parser.add_argument("--no-signature-check", action="store_true",
                        help="关闭跨模块调用签名检查")
    args = parser.parse_args(argv)

    files = iter_python_files(args.targets or DEFAULT_TARGETS)

    # 先建全局签名索引，再做逐文件检查（签名检查需要看到全部模块）
    index: Optional[SignatureIndex] = None
    if not args.no_signature_check:
        index = SignatureIndex()
        index.build(files)

    total = 0
    for path in files:
        problems = check_module(path, index)
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

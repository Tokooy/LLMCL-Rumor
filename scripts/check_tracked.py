# coding=utf-8
"""仓库完整性自检：确认所有源码文件都被 git 追踪。

**为什么需要这个脚本**

``.gitignore`` 里一条看似无害的模式就能静默吞掉整个源码目录。本仓库就踩过一次：

```gitignore
models/          # 意图：忽略下载的 LLM 权重目录
```

gitignore 的模式**不含斜杠时匹配任意层级**，因此这条规则同时忽略了
``src/models/``——整个"模型层"源码（encoder / projector / classifier / 总模型）
从未进入版本库，而 ``git status`` 不会给出任何提示。
等到 push 之后才发现远端缺文件，返工成本很高。

本脚本做两件事：

1. **检查忽略规则**：列出所有被 ``.gitignore`` 忽略、但**扩展名像源码**的文件
   （``.py`` / ``.yaml`` / ``.txt`` / ``.md`` / ``.jsonl`` / ``.ini`` / ``.toml``），
   这些几乎总是误伤；
2. **检查追踪状态**：列出工作区存在、但未被 git 追踪的同类文件
   （未 ``git add`` 或落在忽略规则里）。

用法::

    python scripts/check_tracked.py            # 检查（有遗漏时返回 1）
    python scripts/check_tracked.py --list-ignored   # 只列出忽略规则命中的文件

注意：本脚本只读 git 状态，不修改任何文件，不运行项目代码。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 应当被追踪的文件扩展名（"源码/配置/文档"，不含数据与二进制产物）
SOURCE_SUFFIXES = (
    ".py",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".cfg",
    ".txt",
    ".md",
    ".jsonl",
)

#: 允许被忽略的目录/文件（数据、产物与测试临时目录，属于预期行为）
ALLOWED_IGNORED_DIRS = ("data/raw/", "data/processed/", "outputs/", "logs/")
#: 允许被忽略的路径模式（测试/自检脚本的临时目录，见 .gitignore 的说明）
ALLOWED_IGNORED_PATTERNS = (".tmp",)


def _run_git(args: List[str]) -> Tuple[int, str]:
    """执行 git 命令，返回 ``(返回码, 标准输出)``。"""
    try:
        completed = subprocess.run(
            ["git"] + args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:  # pragma: no cover
        return 127, ""
    return completed.returncode, completed.stdout or ""


def _is_source_like(path: str) -> bool:
    lowered = path.lower()
    if not lowered.endswith(SOURCE_SUFFIXES):
        return False
    if any(lowered.startswith(prefix) for prefix in ALLOWED_IGNORED_DIRS):
        return False
    # 测试/自检脚本的临时目录（.tmp*/ 与任意层级的 .tmp*/）
    components = lowered.split("/")
    if any(
        part.startswith(pattern)
        for part in components
        for pattern in ALLOWED_IGNORED_PATTERNS
    ):
        return False
    return True


def ignored_source_files() -> List[str]:
    """返回被 .gitignore 忽略、但看起来是源码的文件列表。"""
    code, output = _run_git(["status", "--ignored", "--porcelain"])
    if code != 0:
        return []
    files: List[str] = []
    for line in output.splitlines():
        if not line.startswith("!! "):
            continue
        path = line[3:].strip().strip('"').replace("\\", "/")
        if path.endswith("/"):
            # 整个目录被忽略：逐个文件检查
            directory = os.path.join(REPO_ROOT, path)
            if not os.path.isdir(directory):
                continue
            for dirpath, dirnames, filenames in os.walk(directory):
                dirnames[:] = [name for name in dirnames if name != ".git"]
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
                    if _is_source_like(rel):
                        files.append(rel)
            continue
        if _is_source_like(path):
            files.append(path)
    return sorted(set(files))


def untracked_source_files() -> List[str]:
    """返回工作区存在、但未被 git 追踪（且未被忽略）的源码文件。"""
    code, output = _run_git(["ls-files", "--others", "--exclude-standard"])
    if code != 0:
        return []
    files = [
        line.strip().replace("\\", "/")
        for line in output.splitlines()
        if line.strip() and _is_source_like(line.strip())
    ]
    return sorted(set(files))


def tracked_files() -> Set[str]:
    code, output = _run_git(["ls-files"])
    if code != 0:
        return set()
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def working_source_files() -> List[str]:
    """工作区里所有应被追踪的源码文件（跳过 .git 与已允许忽略的目录）。"""
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
        ]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
            if _is_source_like(rel):
                files.append(rel)
    return sorted(set(files))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="确认源码文件都被 git 追踪")
    parser.add_argument("--list-ignored", action="store_true",
                        help="只列出被忽略的源码文件后退出")
    args = parser.parse_args(argv)

    ignored = ignored_source_files()
    untracked = untracked_source_files()

    if args.list_ignored:
        if not ignored:
            print("没有被误伤的源码文件。")
            return 0
        print("被 .gitignore 忽略的源码文件：")
        for path in ignored:
            print(f"  - {path}")
        return 1

    problems = 0

    if ignored:
        problems += len(ignored)
        print("[错误] 以下源码文件被 .gitignore 误伤（应修复忽略规则，"
              "给模式加前导斜杠锚定到仓库根）：")
        for path in ignored:
            print(f"    - {path}")
    else:
        print("[ OK ] 没有被 .gitignore 误伤的源码文件")

    if untracked:
        problems += len(untracked)
        print("[警告] 以下源码文件尚未被 git 追踪（忘记 git add？）：")
        for path in untracked:
            print(f"    - {path}")
    else:
        print("[ OK ] 所有源码文件都已纳入版本控制")

    tracked = tracked_files()
    working = set(working_source_files())
    print("-" * 72)
    print(f"工作区源码/配置文件 {len(working)} 个，已被追踪 {len(tracked & working)} 个")

    if problems:
        print(f"发现 {problems} 个问题：请修复后再 push，否则远端会缺少文件。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

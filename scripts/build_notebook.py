#!/usr/bin/env python
"""从 ``examples/*.py`` 生成 Colab notebook。

计划书 §12.3 把"最小复现脚本"列为 CI 检查项，用意是拦截"论文说能跑但仓库跑不通"。
如果 notebook 是手写副本，它迟早会跟脚本漂移——而那正是最难发现的一类问题：CI 跑
脚本是绿的，用户点开 notebook 却是坏的。

所以这里反过来：**脚本是唯一真相源**，notebook 由它生成。
``--check`` 模式用于 CI，发现不同步就报错。只依赖标准库。

用法::

    python scripts/build_notebook.py                    # 生成全部
    python scripts/build_notebook.py --check            # 校验是否同步（CI）
    python scripts/build_notebook.py examples/foo.py    # 只处理一个文件

脚本里的单元用 ``# %%`` 分隔（Jupytext 的 light 格式）::

    # %% [markdown]
    # ## 标题
    # 正文

    # %%
    print("代码单元")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"

NOTEBOOK_METADATA: dict[str, Any] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    # 让 Colab 打开时展开目录、显示"在 Colab 中打开"徽章
    "colab": {"provenance": [], "toc_visible": True},
}


class NotebookBuildError(Exception):
    """脚本无法转换为 notebook。"""


def _to_source(lines: list[str]) -> list[str]:
    """nbformat 的 source 用行列表表示，除最后一行外都带换行符。"""
    if not lines:
        return []
    source = [f"{line}\n" for line in lines[:-1]]
    source.append(lines[-1])
    return source


def _strip_markdown_prefix(line: str) -> str:
    if line.strip() == "#":
        return ""
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


def split_cells(script: str) -> list[tuple[str, list[str]]]:
    """把脚本切成 ``(cell_type, lines)`` 列表。"""
    cells: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if current_type is None:
            buffer = []
            return
        # 去掉单元首尾的空行，notebook 里不需要
        while buffer and not buffer[0].strip():
            buffer.pop(0)
        while buffer and not buffer[-1].strip():
            buffer.pop()
        cells.append((current_type, buffer))
        buffer = []

    for line in script.splitlines():
        stripped = line.rstrip()
        if stripped.startswith(CELL_MARKER):
            flush()
            current_type = "markdown" if stripped.startswith(MARKDOWN_MARKER) else "code"
            continue
        if current_type is None:
            # 标记之前的内容按代码处理（正常情况下不该有）
            current_type = "code"
        buffer.append(_strip_markdown_prefix(stripped) if current_type == "markdown" else line)
    flush()
    return cells


def build_notebook(script_path: Path) -> dict[str, Any]:
    script = script_path.read_text(encoding="utf-8")
    cells = split_cells(script)
    if not cells:
        raise NotebookBuildError(f"{script_path} 里没有找到任何单元（缺少 {CELL_MARKER} 标记）。")

    notebook_cells: list[dict[str, Any]] = []
    for index, (cell_type, lines) in enumerate(cells):
        cell: dict[str, Any] = {
            # nbformat 4.5 起 cell id 是必填项。这里按序号生成而不是随机生成，
            # 否则每次构建都会产生一份全新的 notebook，--check 永远不可能通过。
            "id": f"cell-{index:03d}",
            "cell_type": cell_type,
            "metadata": {},
            "source": _to_source(lines),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        notebook_cells.append(cell)

    return {
        "cells": notebook_cells,
        "metadata": json.loads(json.dumps(NOTEBOOK_METADATA)),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(notebook: dict[str, Any]) -> str:
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def target_for(script_path: Path) -> Path:
    return script_path.with_suffix(".ipynb")


def discover(targets: list[Path]) -> list[Path]:
    if targets:
        return targets
    return sorted(p for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "scripts", nargs="*", type=Path, help="要处理的 .py 文件（默认 examples/*.py）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验 notebook 是否与脚本同步，不写文件。不同步则退出码为 1。",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    for script_path in discover(args.scripts):
        if not script_path.exists():
            print(f"[错误] 找不到 {script_path}", file=sys.stderr)
            failures.append(str(script_path))
            continue
        try:
            content = render(build_notebook(script_path))
        except NotebookBuildError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            failures.append(str(script_path))
            continue

        notebook_path = target_for(script_path)
        if args.check:
            if not notebook_path.exists():
                print(f"[不同步] {notebook_path.name} 不存在", file=sys.stderr)
                failures.append(str(notebook_path))
            elif notebook_path.read_text(encoding="utf-8") != content:
                print(
                    f"[不同步] {notebook_path.name} 与 {script_path.name} 不一致。"
                    f"请运行：python scripts/build_notebook.py",
                    file=sys.stderr,
                )
                failures.append(str(notebook_path))
            else:
                print(f"[同步] {notebook_path.name}")
        else:
            # newline="\n" 是必须的：Windows 上默认的文本模式会把 \n 翻译成 \r\n，
            # 于是生成器写出 CRLF、pre-commit 的 mixed-line-ending 钩子又把它改回 LF，
            # 两者每次提交都互相打架，--check 也永远通不过。
            notebook_path.write_text(content, encoding="utf-8", newline="\n")
            print(f"[生成] {notebook_path.relative_to(REPO_ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

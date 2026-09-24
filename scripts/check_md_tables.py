#!/usr/bin/env python
"""Markdown 表格结构检查。

GFM 里表格的 ``|`` **即使在反引号内也仍是列分隔符**，要写成 ``\\|`` 才是字面竖线。
漏转义会让该行多出单元格，而 GFM 渲染时**把超出表头格数的部分直接丢弃**——文件里
还在，渲染出来的表里没有。

这类错误不看渲染结果就发现不了：``cat`` 正常、``git diff`` 正常、逐行审查也正常。
2026-09-25 就是这样丢掉过 877 个源字符（``docs/references.md`` 引用 [4] 那行的
Table 4/5 数值与两句逐字引文），于是把它做成检查。

本脚本查两类问题：

1. **整行格数与表头不符**。多出来的部分会被渲染丢弃，是内容凭空消失的直接原因；
   少则会被补空单元格，通常意味着漏写了一个分隔竖线。
2. **反引号内的未转义竖线**。代码段里的 ``|`` 一定会切开单元格，几乎总是笔误；
   即使格数碰巧与表头对得上，内容也已经串到别的格里了，而这一类比第 1 类更隐蔽。

分工：本脚本只看**结构**，不看内容。引用清单的编号 / URL / 状态由
``scripts/check_references.py`` 负责，链接存活由 CI 里的 lychee 负责。

用法::

    python scripts/check_md_tables.py [路径...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 不扫的目录。
#:
#: ``data`` 是本地抓取与核验的草稿区（第三方全文副本按许可约束不入库，见
#: ``docs/references.md`` 的「核验用的全文副本」一节），里面的文件不受本检查管辖。
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "data"}

#: 分隔行里的单个单元格，例如 ``---`` / ``:--`` / ``--:``。
SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")

#: 反引号代码段。GFM 里 N 个反引号开、同样 N 个反引号闭。
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")

PREVIEW_WIDTH = 96


class TableRow:
    """一条表格行：原文本，以及**未转义**竖线的位置。

    转义过的 ``\\|`` 不算分隔符——这正是本检查要区分的两件事。所以这里逐字符扫描
    而不是 ``str.split("|")``：后者会把 ``\\|`` 也切开，一个正确的表格反而被判成错的。
    """

    def __init__(self, line: str) -> None:
        self.line = line
        body = line.strip()
        self.body = body

        # 首尾的边框竖线不算分隔单元格的竖线；这里把它们从扫描范围里去掉。
        start = 1 if body.startswith("|") else 0
        end = len(body)
        if body.endswith("|") and not body.endswith("\\|"):
            end -= 1

        #: 分隔竖线在 ``body`` 中的列号。``breaks[k]`` 是第 k 格与第 k+1 格之间那条。
        self.breaks: list[int] = []
        index = start
        while index < end:
            if body[index] == "\\":
                index += 2  # 被转义的字符整体跳过
                continue
            if body[index] == "|":
                self.breaks.append(index)
            index += 1

    @property
    def cell_count(self) -> int:
        return len(self.breaks) + 1

    def is_separator(self) -> bool:
        """分隔行：每一格都是 ``---`` / ``:--`` 之类。"""
        if not self.breaks and self.body.strip("|-: ") == "":
            return True
        cells = re.split(r"\|", self.body.strip("|"))
        return all(SEPARATOR_CELL_RE.match(cell.strip()) for cell in cells if cell.strip())

    def dropped_length(self, keep: int) -> int:
        """超出 ``keep`` 格的那部分有多少字符。

        GFM 只渲染前 ``keep`` 格，后面的连同竖线一起丢弃，所以这个数就是"渲染出来
        看不到"的字符数——用来说明后果有多严重。
        """
        if self.cell_count <= keep:
            return 0
        return len(self.body) - self.breaks[keep - 1] - 1

    def unescaped_pipes_in_code_spans(self) -> list[int]:
        """反引号代码段里未转义的竖线位置（相对整行的列号）。"""
        found: list[int] = []
        for match in CODE_SPAN_RE.finditer(self.line):
            span_start = match.start(2)
            content = match.group(2)
            for offset, char in enumerate(content):
                if char != "|":
                    continue
                if offset > 0 and content[offset - 1] == "\\":
                    continue
                found.append(span_start + offset)
        return found


def is_table_line(line: str) -> bool:
    """GFM 表格行：缩进不超过 3 格、以 ``|`` 开头。"""
    stripped = line.lstrip(" ")
    return len(line) - len(stripped) <= 3 and stripped.startswith("|")


def preview(text: str) -> str:
    text = text.strip()
    if len(text) <= PREVIEW_WIDTH:
        return text
    return text[:PREVIEW_WIDTH] + "…"


def check_file(findings: list[str], path: Path, root: Path) -> int:
    """检查一个文件，返回发现的表格张数。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        label = path.relative_to(root).as_posix()
    except ValueError:
        label = path.as_posix()

    tables = 0
    index = 0
    while index < len(lines) - 1:
        header = TableRow(lines[index])
        if not is_table_line(lines[index]) or not TableRow(lines[index + 1]).is_separator():
            index += 1
            continue

        tables += 1
        want = header.cell_count
        if want < 2:
            findings.append(f"{label}:{index + 1} 表头只有 1 格，不像表格——分隔行可能写错了。")

        row = index + 2
        while row < len(lines) and is_table_line(lines[row]):
            current = TableRow(lines[row])
            got = current.cell_count
            if got > want:
                findings.append(
                    f"{label}:{row + 1} 该行有 {got} 格，表头只有 {want} 格："
                    f"多出的 {got - want} 格在渲染时会被丢弃，"
                    f"即从第 {want + 1} 格起共 {current.dropped_length(want)} 个字符不会显示。\n"
                    f"        漏转义的竖线要写成 `\\|`（反引号内也一样）。\n"
                    f"        {preview(current.body)}"
                )
            elif got < want:
                findings.append(
                    f"{label}:{row + 1} 该行只有 {got} 格，表头有 {want} 格："
                    f"渲染时会被补上空的单元格，内容也可能已经并进相邻格。\n"
                    f"        {preview(current.body)}"
                )

            # 第 2 类：格数对得上，但代码段里的竖线仍在切单元格——内容会串位。
            for column in current.unescaped_pipes_in_code_spans():
                findings.append(
                    f"{label}:{row + 1} 第 {column} 列：反引号代码段里有未转义的 `|`，"
                    f"它仍会切开单元格，内容会串到别的格。应写成 `\\|`。"
                )
            row += 1

        index = row
    return tables


def collect(root: Path, given: list[Path]) -> list[Path]:
    targets = given or [root]
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(
                path
                for path in sorted(target.rglob("*.md"))
                if not any(part in SKIP_DIRS for part in path.parts)
            )
        elif target.suffix == ".md":
            files.append(target)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        default=None,
        help="要检查的 Markdown 文件或目录（默认整个仓库）",
    )
    args = parser.parse_args(argv)

    files = collect(REPO_ROOT, args.path)
    if not files:
        print("没有找到任何 Markdown 文件。", file=sys.stderr)
        return 1

    findings: list[str] = []
    tables = 0
    for path in files:
        tables += check_file(findings, path, REPO_ROOT)

    for finding in findings:
        print(f"[错误] {finding}", file=sys.stderr)

    if findings:
        print(f"Markdown 表格结构检查未通过：{len(findings)} 个问题。")
        return 1
    print(f"Markdown 表格结构检查通过（{len(files)} 个文件、{tables} 张表）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

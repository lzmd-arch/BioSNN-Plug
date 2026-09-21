#!/usr/bin/env python
"""引用清单结构检查。

计划书 §12.3 把"引用链接存活检查"列为 CI 检查项，并点明它的用途是拦截
"Khacef 署名"类错误的入口——那个错误在仓库里真实发生过：参考文献 [8] 的作者被
误署，v6.1 才更正为 Hajizada et al.。

分工：

* **本脚本**校验 ``docs/references.md`` 的**结构**：编号连续、URL 格式合法、
  每条都有核验状态、没有重复条目。它不联网，因此在本地和 CI 里都很快、很稳。
* **lychee** 在 CI 里负责真正的 HTTP 存活检查（见 ``.github/workflows/ci.yml``）。

把联网的部分单独交给 lychee，是为了让本脚本没有网络依赖——引用清单的结构问题
应该在提交那一刻就被拦下，而不是等到 CI 联网跑完。

用法::

    python scripts/check_references.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES_FILE = REPO_ROOT / "docs" / "references.md"

#: 允许的核验状态。
#:
#: ``verified`` 表示逐条核验过元数据与引用数字；``partial`` 表示文献存在、元数据正确，
#: 但部分归因数字未在原文中确认；``metadata-error`` 表示文献存在、但登记的作者/年份/
#: 卷期页/文章号有误。把 ``metadata-error`` 与 ``partial`` 分开，是因为两者要做的事
#: 完全不同——前者要去改书目，后者只需要补核数字，混在一起看状态列分辨不出来。
ALLOWED_STATUS = {"verified", "partial", "metadata-error", "unverified"}

COLUMNS = ["编号", "文献", "URL", "核验状态"]

#: 表头第一列的可接受写法。
#:
#: 引用清单有三语版本（`references.md` / `.en.md` / `.ja.md`），译文里表头会被翻译。
#: 第一版按中文 `编号` 精确匹配，译文一翻译表头，`parse_rows` 就返回空、并报
#: "没有解析到任何引用条目"——**静默失败**，比报错更难查。所以这里接受各语言的写法。
#: 第二列到第四列不参与锚定（只有第一列用于定位表格），故无需罗列。
HEADER_FIRST_CELLS = {"编号", "No.", "No", "#", "番号", "序号"}

ROW_RE = re.compile(r"^\|(?P<cells>.+)\|\s*$")


def split_cells(line: str) -> list[str] | None:
    match = ROW_RE.match(line.rstrip())
    if not match:
        return None
    return [cell.strip() for cell in match.group("cells").split("|")]


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and set(cells[0]) <= set("-: ")


def parse_rows(text: str) -> list[tuple[int, list[str]]]:
    """返回引用清单的 ``(行号, 单元格)``。

    必须**锚定到表头**再往下读，而不是全文找表格行——这份文档里还有别的表格
    （比如"核验状态"那张两列表），全文扫描会把它们也当成引用条目收进来。
    """
    lines = text.splitlines()

    header_index = None
    for index, line in enumerate(lines):
        cells = split_cells(line)
        if cells and cells[0] in HEADER_FIRST_CELLS:
            header_index = index
            break
    if header_index is None:
        return []

    rows: list[tuple[int, list[str]]] = []
    for offset, line in enumerate(lines[header_index + 1 :]):
        cells = split_cells(line)
        if cells is None:  # 表格结束
            break
        if not cells or not cells[0] or is_separator(cells):
            continue
        rows.append((header_index + 2 + offset, cells))
    return rows


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def strip_markdown_link(cell: str) -> str:
    """``[文字](url)`` 或裸 URL 都接受，返回其中的 URL。"""
    match = re.search(r"\]\((?P<url>[^)\s]+)\)", cell)
    if match:
        return match.group("url")
    return cell.strip("<>").strip()


def check(findings: list[str], references_file: Path) -> None:
    if not references_file.exists():
        findings.append(
            f"找不到 {references_file}。引用清单是引用检查的唯一真相源，缺失则整个检查失去意义。"
        )
        return

    text = references_file.read_text(encoding="utf-8")
    rows = parse_rows(text)

    if not rows:
        findings.append(f"{references_file} 里没有解析到任何引用条目。")
        return

    seen_urls: dict[str, int] = {}
    numbers: list[int] = []

    for line_number, cells in rows:
        if len(cells) < len(COLUMNS):
            findings.append(
                f"第 {line_number} 行只有 {len(cells)} 列，至少需要 {len(COLUMNS)} 列"
                f"（{'、'.join(COLUMNS)}）。"
            )
            continue

        raw_number, _reference, raw_url, status = cells[:4]
        marker = f"第 {line_number} 行（编号 {raw_number!r}）"

        try:
            numbers.append(int(raw_number))
        except ValueError:
            findings.append(f"{marker} 的编号不是整数。")

        url = strip_markdown_link(raw_url)
        if not is_valid_url(url):
            findings.append(
                f"{marker} 的 URL 不合法：{raw_url!r}（需要 http(s):// 开头的完整地址）。"
            )
        elif url in seen_urls:
            findings.append(
                f"{marker} 的 URL 与第 {seen_urls[url]} 行重复：{url}。"
                f"重复条目通常意味着某条被复制粘贴后忘了改。"
            )
        else:
            seen_urls[url] = line_number

        if status not in ALLOWED_STATUS:
            findings.append(
                f"{marker} 的核验状态 {status!r} 非法，只能是 "
                f"{'、'.join(sorted(ALLOWED_STATUS))} 之一。"
            )

    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        findings.append(
            f"引用编号必须从 1 连续递增，实际得到 {numbers}。缺号或跳号会让正文里的 [n] 指错文献。"
        )


def default_targets() -> list[Path]:
    """默认检查引用清单的**全部语言版本**。

    译文（`references.en.md` / `.ja.md`）的书目条目与原文相同，但表头与说明文字是
    翻译的——编号连续性、URL 格式、状态字面量这些结构问题在译文里同样会犯，不检查
    就等于译文没人管。第一版只查中文那一份。
    """
    found = sorted(DEFAULT_REFERENCES_FILE.parent.glob(f"{DEFAULT_REFERENCES_FILE.stem}*.md"))
    return found or [DEFAULT_REFERENCES_FILE]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        default=None,
        help="引用清单路径（默认检查 docs/references 的全部语言版本）",
    )
    args = parser.parse_args(argv)

    targets = args.path or default_targets()
    findings: list[str] = []
    for target in targets:
        check(findings, target)

    for finding in findings:
        print(f"[错误] {finding}", file=sys.stderr)

    if findings:
        print(f"引用清单检查未通过：{len(findings)} 个问题。")
        return 1
    print(
        f"引用清单结构检查通过（{len(targets)} 份语言版本；HTTP 存活检查由 CI 里的 lychee 负责）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

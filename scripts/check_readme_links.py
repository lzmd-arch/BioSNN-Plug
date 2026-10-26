#!/usr/bin/env python3
"""README 内部链接的**本地解析**检查——lychee 那条豁免的补偿。

## 它补的是哪一块

`.github/workflows/ci.yml` 里的 lychee 豁免了 `packages/biosnn-bus/README*.md` 中
**指向本仓库的绝对 URL**。原因写在 ci.yml 的注释里，是结构性的：那个文件是
**PyPI 页面的 long_description 来源**，那里**必须**用绝对 URL（相对路径在 PyPI 上
会 404），于是 lychee 必然走网络访问 github.com——而 GitHub 对这类**自指链接**
间歇返回 503（已复现四次，每次重跑即绿）。

⚠️ **豁免挪走的是「网络可达性」，不是「链接正确性」。** 本脚本把那部分拿回来，
而且是在**本地域**做——比 HTTP 200 更严：

* HTTP 200 只说明「此刻可达」，抓不到「文件被移走、URL 却仍返回 200」的软 404；
* 本地解析直接断言**目标文件在仓库里存在**，文件一改名就红。

## 它检查什么

| 链接形态 | 处置 |
| :--- | :--- |
| 相对路径（`../../x/y.py`、`README.ja.md`） | 相对**该文件所在目录**解析，断言存在 |
| `https://github.com/lzmd-arch/BioSNN-Plug/blob/main/<path>` | 断言仓库内 `<path>` 存在 |
| 其它 http(s) URL、`mailto:`、纯锚点 | **跳过**（外部可达性交给 lychee） |

跳过项会一并报数——「跳过了多少」必须看见，否则一个什么都没检查的绿色输出
和「全都检查过」长得一模一样。

用法::

    python scripts/check_readme_links.py                  # 默认目标（biosnn-bus 三语 README）
    python scripts/check_readme_links.py path/to/README.md
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 指向本仓库文件的绝对 URL 前缀。命中它就把路径抠出来做本地断言。
SELF_BLOB_PREFIX = "https://github.com/lzmd-arch/BioSNN-Plug/blob/main/"

#: 扫描时跳过的目录——与 lychee 的 ``--exclude-path`` 对齐，另加 ``data``
#: （数据集与草稿都在那儿，不是文档）。
#:
#: **为什么默认扫全仓的 ``.md``**（与 lychee 的 ``'./**/*.md'`` 同范围），而不是只扫
#: biosnn-bus 那三个 README：豁免是 **URL 级**的，lychee 没有「按文件豁免某条 URL」的
#: 能力，所以豁免实际影响到**所有用了那条 URL 的文件**——实测除了那三个 README，
#: ``.github/releases/v0.1.0.md`` 也在内。补偿范围若小于豁免范围，差额就是净损失，
#: 「把检查从网络域挪到本地域」这句话就成了空话。
#: 扫全仓还顺带覆盖了其它文件的相对路径链接（纯本地解析，零成本）。
SKIP_DIRS = {".git", ".venv", "node_modules", "data", ".vscode", "ToolsRuby"}

#: 图片也算链接（``![alt](src)``），所以前缀的 ``!`` 可选。
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")


def strip_anchor(target: str) -> str:
    """去掉 ``#fragment``——锚点不影响文件是否存在。"""
    return target.split("#", 1)[0]


def label_of(path: Path) -> str:
    """报告里用的路径标签。

    仓库外的文件（测试用 ``tmp_path`` 就是这种）报**绝对路径**——报告工具不该因为
    文件不在仓库里就抛 ``ValueError`` 崩掉。
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check_file(path: Path, findings: list[str]) -> tuple[int, int]:
    """检查一个文件，返回 ``(本地断言过的链接数, 跳过的链接数)``。"""
    text = path.read_text(encoding="utf-8")
    label = label_of(path)
    checked = skipped = 0

    for match in LINK_RE.finditer(text):
        raw = match.group(1)
        target = strip_anchor(raw)
        line = text[: match.start()].count("\n") + 1
        if not target:  # 纯锚点链接，如 [x](#section)
            skipped += 1
            continue

        if target.startswith(("http://", "https://")):
            if not target.startswith(SELF_BLOB_PREFIX):
                skipped += 1  # 外部链接：可达性由 lychee 负责
                continue
            relative = target[len(SELF_BLOB_PREFIX) :]
            checked += 1
            if not (REPO_ROOT / relative).exists():
                findings.append(
                    f"{label}:{line} 自指链接指向仓库内**不存在**的文件：{relative}"
                    "（这条本来由 lychee 走网络查，已豁免——所以必须在这里拦住）"
                )
            continue

        if target.startswith("mailto:"):
            skipped += 1
            continue

        # 相对路径：相对该文件所在目录。
        checked += 1
        if not (path.parent / target).exists():
            findings.append(f"{label}:{line} 相对链接指向不存在的文件：{target}")

    return checked, skipped


def resolve_targets(given: list[str]) -> list[Path]:
    """给了参数就用参数；否则扫全仓的 ``.md``。

    用 ``os.walk`` 并**在遍历时剪掉** ``SKIP_DIRS``，而不是先 ``rglob`` 再过滤——
    后者会先把 ``.venv`` 里的每一个 ``.md`` 枚举出来再丢掉，慢得没必要。
    """
    if given:
        return [REPO_ROOT / name for name in given]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        found.extend(Path(dirpath) / name for name in filenames if name.endswith(".md"))
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    targets = resolve_targets(list(argv if argv is not None else sys.argv[1:]))
    findings: list[str] = []

    missing = [p for p in targets if not p.is_file()]
    if missing:
        for path in missing:
            print(f"[错误] 找不到 {label_of(path)}", file=sys.stderr)
        return 1

    checked = skipped = 0
    for path in targets:
        file_checked, file_skipped = check_file(path, findings)
        checked += file_checked
        skipped += file_skipped

    if findings:
        for finding in findings:
            print(f"[错误] {finding}", file=sys.stderr)
        print(f"\nREADME 内部链接检查未通过：{len(findings)} 个问题。", file=sys.stderr)
        return 1

    print(
        f"README 内部链接检查通过：本地断言 {checked} 条（相对路径 + 自指绝对 URL），"
        f"跳过 {skipped} 条（外部链接交给 lychee）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

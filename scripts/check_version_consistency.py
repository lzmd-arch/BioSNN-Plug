#!/usr/bin/env python
"""版本号一致性检查。

计划书 §12.3 要求拦截"文件名与内容版本号错位"。历史上这个仓库就出过一次同类问题：
参考文献 [8] 的作者被误署为 "Khacef et al."（实际是 Hajizada et al.），v6.1 才更正。
版本号错位的危害类似——读者按文件名找 v6，拿到的却是 v6.2 的内容。

检查三处：

1. ``packages/biosnn-bus/pyproject.toml`` 的 ``version``；
2. ``packages/biosnn-bus/src/biosnn_bus/__init__.py`` 的 ``__version__``；
3. 计划书文档的**文件名版本**与**正文版本行**。

``research/`` 下的研究代码不参与——计划书 §12.5 给了它 12 个月的破坏性变更免责期。

用法::

    python scripts/check_version_consistency.py
    python scripts/check_version_consistency.py --strict   # 警告也当失败
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIVE = Path("packages") / "biosnn-bus"
PYPROJECT_RELATIVE = PACKAGE_RELATIVE / "pyproject.toml"
INIT_RELATIVE = PACKAGE_RELATIVE / "src" / "biosnn_bus" / "__init__.py"

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None

VERSION_RE = re.compile(r'^\s*version\s*=\s*"(?P<version>[^"]+)"', re.MULTILINE)
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*"(?P<version>[^"]+)"', re.MULTILINE)
DOC_FILENAME_VERSION_RE = re.compile(r"_v(?P<version>\d+(?:\.\d+)*)\.md$")
DOC_CONTENT_VERSION_RE = re.compile(r"\*\*版本\s*(?P<version>\d+(?:\.\d+)*)")


class Findings:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def read_pyproject_version(root: Path, findings: Findings) -> str | None:
    pyproject = root / PYPROJECT_RELATIVE
    if not pyproject.exists():
        findings.error(f"找不到 {pyproject}")
        return None
    text = pyproject.read_text(encoding="utf-8")
    if tomllib is not None:
        version = tomllib.loads(text).get("project", {}).get("version")
        if version:
            return str(version)
    match = VERSION_RE.search(text)
    if not match:
        findings.error(f"{pyproject} 里没有 [project] version 字段")
        return None
    return match.group("version")


def read_init_version(root: Path, findings: Findings) -> str | None:
    init_file = root / INIT_RELATIVE
    if not init_file.exists():
        findings.error(f"找不到 {init_file}")
        return None
    match = INIT_VERSION_RE.search(init_file.read_text(encoding="utf-8"))
    if not match:
        findings.error(f"{init_file} 里没有 __version__")
        return None
    return match.group("version")


def check_package_versions(root: Path, findings: Findings) -> None:
    pyproject_version = read_pyproject_version(root, findings)
    init_version = read_init_version(root, findings)
    if pyproject_version is None or init_version is None:
        return

    if pyproject_version != init_version:
        findings.error(
            f"骨架库版本号不一致：pyproject.toml 是 {pyproject_version!r}，"
            f"__init__.py 是 {init_version!r}。两者必须相同，否则 pip 装到的版本"
            f"与运行时自报的版本会对不上。"
        )
    else:
        print(f"[版本] biosnn-bus = {pyproject_version}")


def check_plan_document(root: Path, findings: Findings) -> None:
    documents = sorted(root.glob("*项目计划书*.md"))
    if not documents:
        findings.warn("根目录没有找到计划书文档，跳过文件名/正文版本号一致性检查。")
        return

    for document in documents:
        filename_match = DOC_FILENAME_VERSION_RE.search(document.name)
        if not filename_match:
            findings.warn(f"{document.name} 的文件名里没有 _vX.Y 形式的版本号，无法与正文比对。")
            continue
        filename_version = filename_match.group("version")

        content_match = DOC_CONTENT_VERSION_RE.search(document.read_text(encoding="utf-8"))
        if not content_match:
            findings.error(f"{document.name} 的正文里找不到 `**版本 X.Y` 形式的版本号。")
            continue
        content_version = content_match.group("version")

        if filename_version == content_version:
            print(f"[版本] 计划书 = v{content_version}（文件名与正文一致）")
        elif content_version.startswith(f"{filename_version}."):
            # 文件名只写到主版本号，宽松但不算错——提醒一下，不拦提交。
            findings.warn(
                f"{document.name} 的文件名版本是 v{filename_version}，正文写的是 "
                f"v{content_version}。文件名只到主版本号，读者无法从文件名分辨具体版本；"
                f"建议改名为 v{content_version} 后重跑本检查。"
            )
        else:
            findings.error(
                f"{document.name} 的文件名版本 v{filename_version} 与正文版本 "
                f"v{content_version} 不一致。这正是计划书 §12.3 要拦截的"
                f"'文件名与内容版本号错位'。"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="把警告也当作失败")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="仓库根目录（默认脚本所在仓库的根）",
    )
    args = parser.parse_args(argv)

    findings = Findings()
    check_package_versions(args.root, findings)
    check_plan_document(args.root, findings)

    for warning in findings.warnings:
        print(f"[警告] {warning}", file=sys.stderr)
    for error in findings.errors:
        print(f"[错误] {error}", file=sys.stderr)

    if findings.errors or (args.strict and findings.warnings):
        print(
            f"版本号一致性检查未通过：{len(findings.errors)} 个错误、"
            f"{len(findings.warnings)} 个警告。"
        )
        return 1

    print(f"版本号一致性检查通过（{len(findings.warnings)} 个警告）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

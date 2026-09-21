#!/usr/bin/env python
"""依赖许可审计（计划书 §12.1）。

计划书点名要核查 eprop-PyTorch 与 Lava/Loihi 相关组件的许可证条款。本脚本做基线：
把全部依赖的许可证列出来，并在出现**强 copyleft** 时失败。

用法::

    uv run --with pip-licenses pip-licenses --format=json > /tmp/licenses.json
    uv run python scripts/check_licenses.py /tmp/licenses.json

## 为什么单独写成脚本，而不是在 CI 里内联 grep

因为它**必须能在本地跑**。第一版是 CI 里的一条
``grep -Ei '\\| *(A?GPL|SSPL|BUSL)' dependency-licenses.md``，它在首次 CI 运行时就
误报了：``pip-licenses --with-license-file`` 会把许可证**正文**整段灌进表格，而
PSF 许可证正文里有一句

    GPL-compatible licenses make it possible to combine Python with ...

按整行 grep 直接命中，于是 CI 红了一个根本没有强 copyleft 依赖的仓库。
纯文本 grep 分不清"许可证名称"与"许可证正文里提到 GPL"——这一条在本地跑一次就
能发现，而它在 CI 里烧掉了一整轮。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: 强 copyleft：出现即失败。这些许可证会传染到使用方，与 Apache-2.0 不兼容。
FORBIDDEN_PREFIXES = ("AGPL", "GPL", "SSPL", "BUSL", "CPAL", "OSL")

#: 弱 copyleft / 有条件许可证：只提醒，不拦。Python 生态里常见，且多数在"仅作为
#: 依赖使用"的场景下没有问题——但值得人看一眼。
CAUTION_PREFIXES = ("LGPL", "EUPL", "MPL", "CDDL", "EPL")

#: 许可证字段缺失或无法识别时的占位值。
UNKNOWN_MARKERS = ("", "UNKNOWN", "NONE")


def classify(license_name: str) -> str:
    """返回 ``forbidden`` / ``caution`` / ``unknown`` / ``ok``。"""
    name = license_name.strip()
    if name.upper() in UNKNOWN_MARKERS:
        return "unknown"
    # 只看许可证名称的开头，不做子串匹配——正文里的"提到 GPL"不算。
    for prefix in FORBIDDEN_PREFIXES:
        if name.upper().startswith(prefix):
            return "forbidden"
    for prefix in CAUTION_PREFIXES:
        if name.upper().startswith(prefix):
            return "caution"
    return "ok"


def load_packages(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(
            f"{path} 的顶层结构应为 JSON 数组（pip-licenses --format=json），收到 {type(data).__name__}。"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("licenses_json", type=Path, help="pip-licenses --format=json 的输出")
    parser.add_argument("--strict", action="store_true", help="把 caution 与 unknown 也当作失败")
    args = parser.parse_args(argv)

    packages = load_packages(args.licenses_json)
    if not packages:
        print("依赖清单为空——审计没有实际检查任何东西，这本身就是一个问题。", file=sys.stderr)
        return 1

    forbidden, caution, unknown = [], [], []
    for pkg in sorted(packages, key=lambda p: p.get("Name", "")):
        verdict = classify(pkg.get("License") or "")
        row = (
            pkg.get("Name", "?"),
            pkg.get("Version", "?"),
            (pkg.get("License") or "").strip() or "(空)",
        )
        if verdict == "forbidden":
            forbidden.append(row)
        elif verdict == "caution":
            caution.append(row)
        elif verdict == "unknown":
            unknown.append(row)

    print(
        f"审计 {len(packages)} 个依赖："
        f"强 copyleft {len(forbidden)}、需留意 {len(caution)}、许可证未知 {len(unknown)}"
    )

    for label, rows in (("强 copyleft", forbidden), ("需留意", caution), ("许可证未知", unknown)):
        if rows:
            print(f"\n[{label}]")
            for name, version, license_name in rows:
                print(f"  {name} {version}  ->  {license_name}")

    if forbidden:
        print(
            f"\n错误：{len(forbidden)} 个依赖使用强 copyleft 许可证，与骨架库的 Apache-2.0 不兼容。",
            file=sys.stderr,
        )
        return 1
    if args.strict and (caution or unknown):
        print("\n--strict：caution / unknown 也视为失败。", file=sys.stderr)
        return 1

    print("\n未发现强 copyleft 依赖。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
